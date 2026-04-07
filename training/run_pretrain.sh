#!/bin/bash

# 设置CUDA库路径
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# SSAST预训练脚本 - FDCAM版本 (频域对比注意力掩码策略)
task=pretrain_joint
mask_patch=60

# 频域对比注意力掩码策略(FDCAM)配置 - 默认启用
use_fdcam=False            # 是否使用频域对比注意力掩码策略
# mask_strategy="freq_diff" # 掩码策略: freq_diff - 基于频域差异性
# mask_strategy="contrastive" # 掩码策略: contrastive - 基于对比性  
mask_strategy="class_guided" # 掩码策略: class_guided - 基于类别引导
# 数据集配置
dataset=Agriculture
# tr_data=/home/research/optim-ssast/data/labeled_data/datafiles/label_train_data.json
# te_data=/home/research/optim-ssast/data/labeled_data/datafiles/label_eval_data.json

tr_data=/home/research/optim-ssast/data/unlabeled_data/datafiles/unlabeled_train_data.json
te_data=/home/research/optim-ssast/data/unlabeled_data/datafiles/unlabeled_eval_data.json


dataset_mean=-2.3221399784088135
dataset_std=5.202194690704346

# dataset_mean=-1.9482910587549986
# dataset_std=5.147459523328597
target_length=300
num_mel_bins=128

# 模型配置
model_size=base
# 不使用patch重叠
fshape=16
tshape=16
fstride=${fshape}
tstride=${tshape}

# 不使用类别平衡采样，因为预训练是自监督的
bal=none

# 优化器配置
batch_size=24  # 每个GPU的批次大小
lr=1e-4        # 降低基础学习率
lr_patience=2  # 学习率调整耐心参数
epoch=200      # 预训练轮数
epoch_iter=4000 # 每个轮次的迭代次数

# 数据增强配置
freqm=0  # 频率掩码
timem=0  # 时间掩码
mixup=0  # mixup增强

# 性能优化配置 - 针对2x RTX 3090
use_fp16=False                # 使用FP16混合精度训练
gradient_accumulation_steps=4 # 梯度累积步数，增大有效批次大小

# 计算有效批次大小
effective_batch=$((batch_size * 2 * gradient_accumulation_steps))
echo "使用2个RTX 3090 GPU, 每GPU批次大小=${batch_size}, 梯度累积=${gradient_accumulation_steps}"
echo "有效批次大小 = ${effective_batch}"

# 为FDCAM添加标记
fdcam_tag="-fdcam-${mask_strategy}"
echo "启用频域对比注意力掩码策略(FDCAM)，掩码策略: ${mask_strategy}"

# 实验结果保存目录
exp_dir=/home/research/optim-ssast/exp/exp_pretrained/mask${mask_patch}-base-${mask_patch}-${model_size}-f${fshape}-t${tshape}-b${effective_batch}-lr${lr}-${task}-${dataset}${fdcam_tag}

# 确保目录存在
mkdir -p ${exp_dir}/models
mkdir -p ${exp_dir}/checkpoints

# 检查点恢复逻辑
# 首先检查是否有checkpoint目录，如果没有则创建
if [ ! -d "${exp_dir}/checkpoints" ]; then
    mkdir -p ${exp_dir}/checkpoints
    echo "创建检查点目录: ${exp_dir}/checkpoints"
    resume_training=False  # 新目录，从头开始训练
else
    # 检查是否已有检查点
    if [ -f "${exp_dir}/checkpoints/checkpoint_metadata.json" ]; then
        echo "已找到元数据文件，将尝试恢复训练"
        resume_training=True
    else
        # 直接检查检查点文件
        latest_checkpoint=$(find ${exp_dir}/checkpoints -name "checkpoint_step_*.pt" -type f | sort -V | tail -n 1)
        if [ -n "$latest_checkpoint" ]; then
            echo "找到最新检查点: $latest_checkpoint"
            resume_training=True
        else
            echo "未找到检查点，从头开始训练"
            resume_training=False
        fi
    fi
fi

# 训练命令
CUDA_VISIBLE_DEVICES=0 python -W ignore /home/research/optim-ssast/src/run.py \
--dataset ${dataset} \
--data-train ${tr_data} \
--data-val ${te_data} \
--exp-dir $exp_dir \
--label-csv /home/research/optim-ssast/data/unlabeled_data/unlabled_class_labels_indices_50.csv \
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
--dataset_mean ${dataset_mean} \
--dataset_std ${dataset_std} \
--target_length ${target_length} \
--num_mel_bins ${num_mel_bins} \
--model_size ${model_size} \
--mask_patch ${mask_patch} \
--n-print-steps 50 \
--task ${task} \
--use_fp16 ${use_fp16} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--resume_training ${resume_training} \
--lr_patience ${lr_patience} \
--epoch_iter ${epoch_iter} \
--num-workers 12 \
--use_fdcam ${use_fdcam} \
--mask_strategy ${mask_strategy} 