
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=LTBoost
seq_len=104

for pred_len in 24 36
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --loss MAE \
  --tree_loss MSE \
  --num_leaves 3 \
  --tree_lr 0.05 \
  --tree_iter 100 \
  --psmooth 15 \
  --normalize \
  --use_revin \
  --tree_lb $seq_len \
  --lb_data N \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'$pred_len.log
done

for pred_len in 48 60
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --loss MAE \
  --tree_loss Mixed \
  --num_leaves 3 \
  --tree_lr 0.05 \
  --tree_iter 100 \
  --psmooth 15 \
  --normalize \
  --use_revin \
  --tree_lb $seq_len \
  --lb_data N \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'$pred_len.log
done