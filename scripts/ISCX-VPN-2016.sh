
model_name=TimeMixer

seq_len=48
pred_len=0
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.003
d_model=128
d_ff=256
batch_size=128
train_epochs=40
patience=10
root_path='../datasets/'
data_path='vpn-iscx-m.csv'
task_name='classification'

python -u run.py \
 --task_name $task_name \
 --is_training 1 \
 --root_path $root_path \
 --data_path $data_path \
 --model_id $task_name'_'$data_path \
 --model $model_name \
 --data custom \
 --features MS\
 --seq_len 96 \
 --label_len 0 \
 --pred_len $pred_len \
 --e_layers 5 \
 --d_layers 2 \
 --factor 3 \
 --enc_in 79 \
 --dec_in 79 \
 --c_out 1 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
 --target Label \
 --channel_independence 0 \
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 128 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --decomp_method dft_decomp \
 --down_sampling_method conv \
 --down_sampling_window $down_sampling_window 
 
#  python -u run.py \
#  --task_name $task_name \
#  --is_training 1 \
#  --root_path $root_path \
#  --data_path $data_path \
#  --model_id $task_name'_'$data_path \
#  --model $model_name \
#  --data custom \
#  --features MS\
#  --seq_len 192 \
#  --label_len 0 \
#  --pred_len $pred_len \
#  --e_layers 5 \
#  --d_layers 2 \
#  --factor 3 \
#  --enc_in 79 \
#  --dec_in 79 \
#  --c_out 1 \
#  --des 'Exp' \
#  --itr 1 \
#  --use_norm 0 \
#  --target Label \
#  --channel_independence 0 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size 16 \
#  --learning_rate $learning_rate \
#  --train_epochs $train_epochs \
#  --patience $patience \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window 
 
#  python -u run.py \
#  --task_name $task_name \
#  --is_training 1 \
#  --root_path $root_path \
#  --data_path $data_path \
#  --model_id $task_name'_'$data_path \
#  --model $model_name \
#  --data custom \
#  --features MS\
#  --seq_len 384 \
#  --label_len 0 \
#  --pred_len $pred_len \
#  --e_layers 5 \
#  --d_layers 2 \
#  --factor 3 \
#  --enc_in 79 \
#  --dec_in 79 \
#  --c_out 1 \
#  --des 'Exp' \
#  --itr 1 \
#  --use_norm 0 \
#  --target Label \
#  --channel_independence 0 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size 16 \
#  --learning_rate $learning_rate \
#  --train_epochs $train_epochs \
#  --patience $patience \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window 
 
 