
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
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --loss Custom \
  --tree_loss Mixed \
  --num_leaves 2 \
  --tree_lr 0.05 \
  --tree_iter 300 \
  --psmooth 15 \
  --normalize \
  --use_revin \
  --use_sigmoid \
  --tree_lb $seq_len \
  --lb_data N \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'$pred_len.log
done