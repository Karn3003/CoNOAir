export CUDA_VISIBLE_DEVICES=1

python /home/ece/hdd/Karn/carbon_monoxide/exp_carbon.py \
  --data-path /home/ece/hdd/Karn/carbon_monoxide/dataset \
  --ntrain 3000 \
  --ntest 500 \
  --ntotal 3500 \
  --in_dim 10 \
  --out_dim 1 \
  --h 140 \
  --w 124 \
  --h-down 1 \
  --w-down 1 \
  --T-in 10 \
  --T-out 6 \
  --batch-size 20 \
  --learning-rate 0.001 \
  --model FNO_2D \
  --d-model 20 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 4,4 \
  --padding 0,0 \
  --model-save-path ./checkpoints/carbon \
  --model-save-name fno_2D_channels_20_Tin_10_Tout_6.pt \
  --project-name Carbon-Monoxide \
  --run-name FNO-2D_Tin_10_Tout_6