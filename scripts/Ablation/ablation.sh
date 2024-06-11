
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Ablation" ]; then
    mkdir ./logs/Ablation
fi

seq_len=336
model_name=LightGBM
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --num_leaves 7 >logs/Ablation/$model_name'_'Etth1_$seq_len'_'$pred_len.log
done

model_name=NLinear
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/Ablation/$model_name'_'Etth1_$seq_len'_'$pred_len.log
done

# RLinear
model_name=Linear
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --use_revin \
  --loss MAE --batch_size 32 --learning_rate 0.005 >logs/Ablation/$model_name'_R_'Etth1_$seq_len'_'$pred_len.log
done

# RIN-Linear
model_name=NLinear
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --use_revin \
  --loss MAE --batch_size 32 --learning_rate 0.005 >logs/Ablation/$model_name'_RIN_'Etth1_$seq_len'_'$pred_len.log
done

# LTBoost -NNorm
model_name=LTBoost
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --num_leaves 7 \
  --use_revin \
  --normalize \
  --loss MAE \
  --batch_size 32 \
  --learning_rate 0.005 \
  --tree_lb 7 --lb_data 0 >logs/Ablation/$model_name'_raw_'Etth1_$seq_len'_'$pred_len.log
done

# LTBoost
model_name=LTBoost
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --num_leaves 7 \
  --use_revin \
  --normalize \
  --loss MAE \
  --batch_size 32 \
  --learning_rate 0.005 \
  --tree_lb 7 --lb_data N >logs/Ablation/$model_name'_Nnorm_'Etth1_$seq_len'_'$pred_len.log
done