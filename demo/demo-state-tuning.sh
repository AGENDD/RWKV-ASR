
# 3B
load_model='RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth' 
#7B
# load_model='RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'

#model output dir
proj_dir='output'
data_file='temp_datasets'

# 3B
n_layer=32
n_embd=2560

# 7B
# n_layer=32
# n_embd=4096

micro_bsz=4
epoch_steps=18089
ctx_len=1024
device=4
epoch_save=1
#1 for training, 2 for prediction, 3 for wer on libri test set
OP=1

QUANT='nf4' 
export HF_ENDPOINT=https://hf-mirror.com
python train.py --load_model $load_model --devices $device \
--proj_dir $proj_dir --data_file $data_file \
--data_type binidx --vocab_size 65536 \
--ctx_len $ctx_len --epoch_steps $epoch_steps --epoch_count 1000 --epoch_begin 0 --epoch_save $epoch_save --micro_bsz $micro_bsz \
--n_layer $n_layer --n_embd $n_embd \
--pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 100 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --strategy deepspeed_stage_1 --grad_cp 1 --op $OP \
--precision bf16 \
--my_testing "x060" \
--train_type "state"  --dataload pad
# --quant $QUANT
