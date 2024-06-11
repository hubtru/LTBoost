
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=LTBoost
seq_len=720

for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --des 'Exp' \
  --loss Custom \
  --tree_loss Mixed \
  --num_leaves 2 \
  --tree_lr 0.01 \
  --tree_iter 200 \
  --normalize \
  --use_revin \
  --tree_lb $seq_len \
  --lb_data RIN \  
  --itr 1 --batch_size 16 --learning_rate 0.05 >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'$pred_len.log 
done