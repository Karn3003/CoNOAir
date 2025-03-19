import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.adam import Adam
from utils.params import get_args
from model_dict import get_model
import math
import os

# import wandb
# wandb.login()

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()

TRAIN_PATH_1 = os.path.join(args.data_path, './train1.mat')
TRAIN_PATH_2 = os.path.join(args.data_path, './train2.mat')
TEST_PATH = os.path.join(args.data_path, './test_data.mat')

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
T_in = args.T_in
T_out = args.T_out

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

# run = wandb.init(
#     # Set the project where this run will be logged
#     project = args.project_name,
#     # Provide your desired run name here
#     name = args.run_name
#     )

################################################################
# models
################################################################
model = get_model(args)
print(model)
print(get_num_params(model))

################################################################
# load data and data normalization
################################################################

# Train data

reader = MatReader(TRAIN_PATH_1)
train_a_1 = reader.read_field('a')[:, :, ::r1, ::r2]

reader = MatReader(TRAIN_PATH_2)
train_a_2 = reader.read_field('a')[:, :, ::r1, ::r2]

train_a = torch.cat((train_a_1, train_a_2), 0)
train_a = train_a.permute(0, 2, 3, 1)

# Test data

reader = MatReader(TEST_PATH)
test_a = reader.read_field('a')[:, :, ::r1, ::r2]
test_a = test_a.permute(0, 2, 3, 1)

print(train_a.shape)
print(test_a.shape)

T = T_in + T_out
step = 1

train_X = torch.tensor([])
train_Y = torch.tensor([])
test_X = torch.tensor([])
test_Y = torch.tensor([])

for i in range(T - T_in):
    x = train_a[:, :, :, i: i + T_in]
    train_X = torch.cat((train_X, x), dim = 0)
    
for i in range(T - T_in):
    y = train_a[:, :, :, i + step]
    train_Y = torch.cat((train_Y, y), dim = 0)
    
for i in range(T - T_in):
    x = test_a[:, :, :, i: i + T_in]
    test_X = torch.cat((test_X, x), dim = 0)
    
for i in range(T - T_in):
    y = test_a[:, :, :, i + step]
    test_Y = torch.cat((test_Y, y), dim = 0)
    
n_train = train_X.shape[0]
n_test = test_X.shape[0]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_Y), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X, test_Y), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        batch_size = x.shape[0]
        
        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s1, s2)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()
    model.eval()

    test_l2 = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            
            out = model(x).reshape(batch_size, s1, s2)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    t2 = default_timer()
    
    train_l2 /= ntrain
    test_l2 /= ntest
    
    print(ep, t2 - t1, train_l2, test_l2)
    # wandb.log({"epoch": ep, "train_l2": train_l2, "test_l2": test_l2, "time": t2-t1})
    
    if ep % step_size == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))