export CUDA_VISIBLE_DEVICES=0

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
  --T-out 16 \
  --batch-size 20 \
  --learning-rate 0.001 \
  --model complete_complex_FFNO_2D \
  --d-model 20 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 4,4 \
  --padding 11,11 \
  --model-save-path ./checkpoints/carbon_new \
  --model-save-name complete_complex_ffno_2D_channels_20_Tin_10_T_out_16.pt \
  --project-name Carbon-Monoxide \
  --run-name complete_complex_FFNO-20-Tin-10-Tout-16