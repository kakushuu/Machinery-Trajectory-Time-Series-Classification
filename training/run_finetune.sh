#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-[5,12]
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="ast_agriculture"
#SBATCH --output=./slurm_log/log_%j.txt

# 设置CUDA库路径
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# conda activate ast

# 设置CUDA库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

pretrain_exp=post-m108
pretrain_model=Agriculture
# pretrain_path=/home/research/optim-ssast/exp/exp_pretrained_model/audioset_10_10_0.4593.pth
pretrain_path=/home/research/optim-ssast/Paper_Exp_Summary/Self-Distilled/teacher_model/teacher_best_audio_model.pth
dataset=Agriculture
set=full

dataset_mean=-2.3221399784088135
dataset_std=5.202194690704346


target_length=288
noise=True
use_pretrained=True  # 是否使用预训练模型，默认为True

task=ft_avgtok
model_size=base
head_lr=1
warmup=True

if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=40
  tr_data=/home/research/optim-ssast/data/labeled_data/datafiles/label_train_data.json
elif [ $set == full ]
then
  bal=bal
  lr=1e-5
  epoch=40
  tr_data=/home/research/optim-ssast/data/labeled_data/datafiles/label_train_data.json
fi

te_data=/home/research/optim-ssast/data/labeled_data/datafiles/label_eval_data.json
freqm=24
timem=96
mixup=0.0
fstride=10
tstride=10
fshape=16
tshape=16
batch_size=48
exp_dir=/home/research/optim-ssast/Paper_Exp_Summary_Finetune/finetune_distilled/teacher/archive/post-m60/

# 打印环境信息
echo "Using conda environment: $CONDA_PREFIX"1
echo "CUDA Library Path: $LD_LIBRARY_PATH"

# 检查CUDA可用性
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available()); print('CUDA设备数量:', torch.cuda.device_count())"

# 创建实验目录
mkdir -p $exp_dir

# 保存训练参数到文件
cat > $exp_dir/training_params.txt << EOF
训练参数信息:
===========================================
时间: $(date)
数据集: ${dataset}
数据集版本: ${set}
训练数据: ${tr_data}
测试数据: ${te_data}
===========================================
模型信息:
预训练模型: ${pretrain_model}
预训练路径: ${pretrain_path}
模型大小: ${model_size}
任务类型: ${task}
使用预训练: ${use_pretrained}
===========================================
训练超参数:
学习率: ${lr}
头部学习率: ${head_lr}
批量大小: ${batch_size}
训练轮数: ${epoch}
频率遮蔽: ${freqm}
时间遮蔽: ${timem}
混合比例: ${mixup}
频率步长: ${fstride}
时间步长: ${tstride}
频率形状: ${fshape}
时间形状: ${tshape}
目标长度: ${target_length}
数据集均值: ${dataset_mean}
数据集标准差: ${dataset_std}
平衡策略: ${bal}
噪声增强: ${noise}
===========================================
其他设置:
学习率调度起始: 5
学习率调度步长: 1
学习率调度衰减: 0.85
权重平均: False
权重平均起始: 6
权重平均结束: 25
损失函数: CE
评估指标: mAP
===========================================
EOF

# 将所有命令行参数也保存到参数文件
echo -e "\n完整命令行参数:" >> $exp_dir/training_params.txt
echo "CUDA_VISIBLE_DEVICES=0 python -W ignore /home/research/optim-ssast/src/run.py \
  --dataset ${dataset} \
  --data-train ${tr_data} \
  --data-val ${te_data} \
  --exp-dir $exp_dir \
  --label-csv /home/research/optim-ssast/data/labeled_data/datafiles/machinery_class_labels_indices.csv \
  --n_class 10 \
  --lr $lr \
  --n-epochs ${epoch} \
  --batch-size $batch_size \
  --save_model True \
  --freqm $freqm \
  --timem $timem \
  --mixup ${mixup} \
  --bal ${bal} \
  --tstride $tstride \
  --fstride $fstride \
  --fshape ${fshape} \
  --tshape ${tshape} \
  --warmup False \
  --task ${task} \
  --model_size ${model_size} \
  --adaptschedule False \
  --pretrained_mdl_path ${pretrain_path} \
  --dataset_mean ${dataset_mean} \
  --dataset_std ${dataset_std} \
  --target_length ${target_length} \
  --num_mel_bins 128 \
  --head_lr ${head_lr} \
  --noise ${noise} \
  --use_pretrained ${use_pretrained} \
  --lrscheduler_start 5 \
  --lrscheduler_step 1 \
  --lrscheduler_decay 0.85 \
  --wa False \
  --wa_start 6 \
  --wa_end 25 \
  --loss CE \
  --metrics acc" >> $exp_dir/training_params.txt

CUDA_VISIBLE_DEVICES=0 python -W ignore /home/research/optim-ssast/src/run.py \
  --dataset ${dataset} \
  --data-train ${tr_data} \
  --data-val ${te_data} \
  --exp-dir $exp_dir \
  --label-csv /home/research/optim-ssast/data/labeled_data/datafiles/machinery_class_labels_indices.csv \
  --n_class 10 \
  --lr $lr \
  --n-epochs ${epoch} \
  --batch-size $batch_size \
  --save_model True \
  --freqm $freqm \
  --timem $timem \
  --mixup ${mixup} \
  --bal ${bal} \
  --tstride $tstride \
  --fstride $fstride \
  --fshape ${fshape} \
  --tshape ${tshape} \
  --warmup False \
  --task ${task} \
  --model_size ${model_size} \
  --adaptschedule False \
  --pretrained_mdl_path ${pretrain_path} \
  --dataset_mean ${dataset_mean} \
  --dataset_std ${dataset_std} \
  --target_length ${target_length} \
  --num_mel_bins 128 \
  --head_lr ${head_lr} \
  --noise ${noise} \
  --use_pretrained ${use_pretrained} \
  --lrscheduler_start 5 \
  --lrscheduler_step 1 \
  --lrscheduler_decay 0.85 \
  --wa False \
  --wa_start 6 \
  --wa_end 25 \
  --loss CE \
  --metrics acc

python /home/research/optim-ssast/src/finetune/agricuture/get_machinery_result.py --exp_path ${exp_dir}