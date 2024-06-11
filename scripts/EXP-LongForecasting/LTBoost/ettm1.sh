
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=LTBoost
seq_len=720

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --loss MAE \
  --tree_loss Huber \
  --train_epochs 20 \
  --num_leaves 15 \
  --tree_lr 0.01 \
  --tree_iter 200 \
  --normalize \
  --use_revin \
  --tree_lb $seq_len \
  --lb_data N \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'ETTm1_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --loss MAE \
  --tree_loss MSE \
  --train_epochs 20 \
  --num_leaves 2 \
  --tree_lr 0.01 \
  --tree_iter 200 \
  --normalize \
  --use_revin \
  --tree_lb $seq_len \
  --lb_data RIN \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'ETTm1_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --loss MAE \
  --tree_loss MSE \
  --train_epochs 20 \
  --num_leaves 2 \
  --tree_lr 0.01 \
  --tree_iter 200 \
  --normalize \
  --use_revin \
  --tree_lb $seq_len \
  --lb_data RIN \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'ETTm1_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --loss MAE \
  --tree_loss MSE \
  --train_epochs 20 \
  --num_leaves 2 \
  --tree_lr 0.01 \
  --tree_iter 200 \
  --normalize \
  --use_revin \
  --tree_lb $seq_len \
  --lb_data RIN \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'ETTm1_$seq_len'_'720.log
