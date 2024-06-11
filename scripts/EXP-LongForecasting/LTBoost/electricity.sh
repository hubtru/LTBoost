
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
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --des 'Exp' \
  --loss Custom \
  --tree_loss Mixed \
  --num_leaves 2 \
  --tree_lr 0.005 \
  --tree_iter 400 \
  --normalize \
  --use_revin \
  --tree_lb 7 \
  --lb_data RIN \
  --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log 
done
