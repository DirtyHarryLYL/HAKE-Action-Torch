exp_dir='runs/vitb16_pastanet'

if [[ ! -d $exp_dir ]]; then
    mkdir -p $exp_dir
fi

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --output_dir ${exp_dir} | tee -a ${exp_dir}/log.txt

