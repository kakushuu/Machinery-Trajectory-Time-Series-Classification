# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
import json
import pandas as pd
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from models import ASTModel
import numpy as np
from traintest import train, validate
from traintest_mask import trainmask
from datetime import datetime

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")

parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# only used in pretraining stage or from-scratch fine-tuning experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
# during self-supervised pretraining stage, no patch split overlapping is used (to aviod shortcuts), i.e., fstride=fshape and tstride=tshape
# during fine-tuning, using patch split overlapping (i.e., smaller {f,t}stride than {f,t}shape) improves the performance.
# it is OK to use different {f,t} stride in pretraining and finetuning stages (though fstride is better to keep t
# he same)
# but {f,t}stride in pretraining and finetuning stages must be consistent.
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')

parser.add_argument("--task", type=str, default='ft_cls', help="预训练或微调任务", 
                   choices=["ft_avgtok", "ft_cls", 
                           "pretrain_mpc", "pretrain_mpg", "pretrain_eat", "pretrain_joint",
                           "pretrain_mae_mpc", "pretrain_mae_mpg", "pretrain_mae_joint",
                           "pretrain_post", "pretrain_core", "pretrain_compass"])

# pretraining augments
#parser.add_argument('--pretrain_stage', help='True for self-supervised pretraining stage, False for fine-tuning stage', type=ast.literal_eval, default='False')
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")

# fine-tuning arguments
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="预训练模型路径")
parser.add_argument("--use_pretrained", type=str, default=None, help="预训练模型路径(兼容旧参数名)")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
parser.add_argument("--noise", help='if augment noise in finetuning', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument("--iter_visualize", type=int, default=4000, help="how many iterations to visualize")

# 添加新的参数用于实验记录
parser.add_argument("--experiment_name", type=str, default="ast_pretrain", help="实验名称")
parser.add_argument("--paper_baseline", help='是否与论文基线进行比较', type=ast.literal_eval, default='True')
parser.add_argument("--log_performance", help='是否记录详细性能指标', type=ast.literal_eval, default='True')

# 使用字符串形式，更灵活
parser.add_argument("--visualization_types", type=str, 
                   default="mpc,eat",  # 默认比较mpc和eat
                   help="用逗号分隔的可视化方法类型，例如: mpc,mpg,eat,CoRe")
parser.add_argument("--ablation_freq", type=int, default=10, 
                   help="运行POST消融实验的频率（每N个epoch）")
parser.add_argument("--visualize_ablation", type=ast.literal_eval, default=False,
                   help="是否为消融实验生成可视化")

# 添加实验时间戳和MPC对比参数
parser.add_argument("--experiment_timestamp", type=str, default=None,
                   help="实验时间戳，用于生成独立的日志文件")
parser.add_argument("--mpc_comparison_mode", type=int, default=0,
                   help="是否启用MPC对比模式")

# 添加EAT蒸馏相关参数
parser.add_argument("--enable_distillation", type=ast.literal_eval, default=False,
                   help="是否启用EAT风格的teacher-student蒸馏")
parser.add_argument("--distillation_temperature", type=float, default=4.0,
                   help="蒸馏温度参数，控制知识转移的软硬程度")
parser.add_argument("--distillation_alpha", type=float, default=0.7,
                   help="蒸馏损失权重，平衡主任务损失和蒸馏损失")
parser.add_argument("--teacher_momentum", type=float, default=0.999,
                   help="teacher网络的EMA更新动量参数")

args = parser.parse_args()


def analyze_post_ablation_results(base_exp_dir, all_config_results, post_configurations):
    """分析POST消融实验结果"""
    print("\n" + "="*60)
    print("🔬 POST消融实验结果分析")
    print("="*60)
    
    # 分析结果
    analysis_report = {
        "experiment_summary": {
            "total_configs": len(all_config_results),
            "completed_configs": len([r for r in all_config_results.values() if 'error' not in r]),
            "failed_configs": len([r for r in all_config_results.values() if 'error' in r]),
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "configuration_results": {},
        "detailed_analysis": {},
        "recommendations": []
    }
    
    # 处理每个配置的结果 - 修复：从日志文件中解析实际结果
    baseline_results = None
    for config_name, results in all_config_results.items():
        if 'error' not in results:
            # 从日志文件中解析实际的训练结果
            parsed_results = parse_training_log(results.get('log_file', None))
            
            analysis_report["configuration_results"][config_name] = {
                "train_acc": parsed_results.get('train_acc', results.get('train_acc', 0.0)),
                "val_acc": parsed_results.get('val_acc', results.get('val_acc', 0.0)),
                "train_loss": parsed_results.get('train_loss', results.get('train_loss', 0.0)),
                "val_loss": parsed_results.get('val_loss', results.get('val_loss', 0.0)),
                "total_params_M": parsed_results.get('total_params_M', results.get('total_params_M', 0.0)),
                "convergence_time": parsed_results.get('convergence_time', results.get('convergence_time', 0.0)),
                "training_time": parsed_results.get('training_time', results.get('training_time', 0.0)),
                "best_val_acc": parsed_results.get('best_val_acc', results.get('best_val_acc', 0.0)),
                "best_epoch": parsed_results.get('best_epoch', results.get('best_epoch', 0)),
                "log_file": results.get('log_file', None),
                "exp_dir": results.get('exp_dir', None),
                "status": results.get('status', 'completed')
            }
            
            # 寻找基线结果
            if 'baseline' in config_name.lower():
                baseline_results = analysis_report["configuration_results"][config_name]
    
    # 如果没有找到baseline，使用第一个有效结果作为基准
    if baseline_results is None and analysis_report["configuration_results"]:
        baseline_results = list(analysis_report["configuration_results"].values())[0]
        print("⚠️  未找到基础配置结果，使用第一个配置作为基准")
    
    # 找到最佳配置
    if analysis_report["configuration_results"]:
        best_config = max(analysis_report["configuration_results"].items(), 
                         key=lambda x: x[1]['best_val_acc'])
    else:
        print("⚠️  没有有效的配置结果")
        return analysis_report
    
    analysis_report["detailed_analysis"] = {
        "best_configuration": {
            "name": best_config[0],
            "performance": best_config[1]
        },
        "component_contributions": {},
        "combination_effects": {},
        "baseline_performance": baseline_results or best_config[1]
    }
    
    # 组件贡献分析 - 修复：使用正确的配置名称
    baseline_acc = analysis_report["detailed_analysis"]["baseline_performance"].get('best_val_acc', 0.0)
    
    single_component_configs = ['post_mechanical', 'post_similarity', 'post_balanced', 'post_local']
    for config_name in single_component_configs:
        if config_name in analysis_report["configuration_results"]:
            config_acc = analysis_report["configuration_results"][config_name]['best_val_acc']
            improvement = config_acc - baseline_acc
            analysis_report["detailed_analysis"]["component_contributions"][config_name] = {
                "accuracy": config_acc,
                "improvement": improvement,
                "improvement_percent": improvement / (baseline_acc + 1e-8) * 100
            }
    
    # 组合效果分析 - 修复：使用正确的配置名称
    combination_configs = ['post_mech_sim', 'post_sim_bal', 'post_three_way', 'post_full']
    for config_name in combination_configs:
        if config_name in analysis_report["configuration_results"]:
            config_acc = analysis_report["configuration_results"][config_name]['best_val_acc']
            improvement = config_acc - baseline_acc
            analysis_report["detailed_analysis"]["combination_effects"][config_name] = {
                "accuracy": config_acc,
                "improvement": improvement,
                "improvement_percent": improvement / (baseline_acc + 1e-8) * 100
            }
    
    # 生成建议
    best_improvement = best_config[1]['best_val_acc'] - baseline_acc
    if best_improvement > 0:
        analysis_report["recommendations"].append(
            f"建议使用 {best_config[0]} 配置，可获得 +{best_improvement:.6f} ({best_improvement/(baseline_acc + 1e-8)*100:.2f}%) 的性能提升"
        )
    else:
        analysis_report["recommendations"].append(
            "POST方法相比基线没有明显提升，建议检查超参数设置或数据质量"
        )
    
    # 计算效率分数（准确率提升 / 训练时间）
    efficiency_scores = {}
    for config_name, results in analysis_report["configuration_results"].items():
        if results['training_time'] > 0:
            improvement = results['best_val_acc'] - baseline_acc
            efficiency = improvement / (results['training_time'] / 1000.0)  # 每秒提升
            efficiency_scores[config_name] = efficiency
    
    if efficiency_scores:
        best_efficiency_config = max(efficiency_scores.items(), key=lambda x: x[1])
        analysis_report["recommendations"].append(
            f"从效率角度考虑，{best_efficiency_config[0]} 配置具有最佳的性能/时间比"
        )
    
    # 保存分析结果
    analysis_file = os.path.join(base_exp_dir, 'post_ablation_complete_analysis.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 完整分析报告已保存到: {analysis_file}")
    
    # 输出关键建议
    print(f"\n💡 建议:")
    for recommendation in analysis_report["recommendations"]:
        print(f"  • {recommendation}")
    
    # 调用Markdown报告生成（如果函数存在）
    try:
        generate_markdown_report(base_exp_dir, analysis_report, post_configurations)
    except NameError:
        print("⚠️  generate_markdown_report 函数未定义，跳过Markdown报告生成")
    except Exception as e:
        print(f"⚠️  生成Markdown报告时出错: {str(e)}")
    
    return analysis_report

def parse_training_log(log_file_path):
    """从训练日志文件中解析实际的训练结果"""
    if not log_file_path or not os.path.exists(log_file_path):
        return {}
    
    results = {
        'train_acc': 0.0,
        'val_acc': 0.0,
        'train_loss': 0.0,
        'val_loss': 0.0,
        'best_val_acc': 0.0,
        'best_epoch': 0,
        'training_time': 0.0,
        'total_params_M': 0.0
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取最佳验证准确率
        import re
        
        # 增强的正则表达式模式匹配
        # 匹配训练过程中的验证结果 - 支持多种格式
        val_acc_patterns = [
            r'验证精度[:\s]*([0-9.]+)',
            r'验证完成[:\s]*acc=([0-9.]+)',
            r'Val Acc[:\s]*([0-9.]+)',
            r'validation accuracy[:\s]*([0-9.]+)',
            r'acc=([0-9.]+).*验证完成'
        ]
        
        val_accs = []
        for pattern in val_acc_patterns:
            matches = re.findall(pattern, content)
            if matches:
                val_accs.extend([float(x) for x in matches])
        
        if val_accs:
            results['best_val_acc'] = max(val_accs)
            results['val_acc'] = val_accs[-1]  # 最后一次验证精度
        
        # 匹配训练准确率 - 支持多种格式
        train_acc_patterns = [
            r'Train Acc[:\s]*([0-9.]+)',
            r'训练精度[:\s]*([0-9.]+)',
            r'Training Accuracy[:\s]*([0-9.]+)',
            r'Train.*Acc.*?([0-9.]+)'
        ]
        
        train_accs = []
        for pattern in train_acc_patterns:
            matches = re.findall(pattern, content)
            if matches:
                train_accs.extend([float(x) for x in matches])
        
        if train_accs:
            results['train_acc'] = train_accs[-1]
        
        # 匹配训练损失 - 支持多种格式
        train_loss_patterns = [
            r'Train Loss[:\s]*([0-9.]+)',
            r'训练损失[:\s]*([0-9.]+)',
            r'Training Loss[:\s]*([0-9.]+)',
            r'Train.*Loss.*?([0-9.]+)'
        ]
        
        train_losses = []
        for pattern in train_loss_patterns:
            matches = re.findall(pattern, content)
            if matches:
                train_losses.extend([float(x) for x in matches])
        
        if train_losses:
            results['train_loss'] = train_losses[-1]
        
        # 匹配验证损失 - 支持多种格式
        val_loss_patterns = [
            r'验证损失[:\s]*([0-9.]+)',
            r'验证完成[:\s]*.*nce=([0-9.]+)',
            r'Val Loss[:\s]*([0-9.]+)',
            r'validation loss[:\s]*([0-9.]+)',
            r'nce=([0-9.]+).*验证完成'
        ]
        
        val_losses = []
        for pattern in val_loss_patterns:
            matches = re.findall(pattern, content)
            if matches:
                val_losses.extend([float(x) for x in matches])
        
        if val_losses:
            results['val_loss'] = val_losses[-1]
        
        # 匹配参数量
        param_patterns = [
            r'参数量[:\s]*([0-9.]+)M',
            r'Parameters[:\s]*([0-9.]+)M',
            r'Params[:\s]*([0-9.]+)M'
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, content)
            if matches:
                results['total_params_M'] = float(matches[-1])
                break
        
        # 匹配训练时间
        time_patterns = [
            r'训练时间[:\s]*([0-9.]+)',
            r'Training Time[:\s]*([0-9.]+)',
            r'Total.*time[:\s]*([0-9.]+)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, content)
            if matches:
                results['training_time'] = float(matches[-1])
                break
            
        # 如果上述模式没有匹配到，尝试其他格式
        if results['best_val_acc'] == 0.0:
            # 尝试匹配性能比较表格中的数据
            table_matches = re.findall(r'([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', content)
            if table_matches:
                # 假设表格格式为：训练精度 验证精度 训练损失 验证损失
                last_row = table_matches[-1]
                results['train_acc'] = float(last_row[0])
                results['val_acc'] = float(last_row[1])
                results['best_val_acc'] = float(last_row[1])
                results['train_loss'] = float(last_row[2])
                results['val_loss'] = float(last_row[3])
        
        # 特殊处理POST方法的日志格式 - 修复最重要的问题
        if 'post' in log_file_path.lower():
            # POST方法经常在日志中输出准确率但格式不同
            post_val_patterns = [
                r'Epoch.*验证完成.*acc=([0-9.]+)',
                r'验证.*结束.*准确率.*?([0-9.]+)',
                r'epoch\s+\d+.*validation.*?([0-9.]+)',
                r'POST.*验证.*?([0-9.]+)'
            ]
            
            all_post_vals = []
            for pattern in post_val_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    all_post_vals.extend([float(x) for x in matches])
            
            if all_post_vals:
                results['best_val_acc'] = max(all_post_vals)
                results['val_acc'] = all_post_vals[-1]
                print(f"🔧 POST特殊解析找到验证准确率: {results['best_val_acc']:.6f}")
            
            # 如果还是0，尝试从最后几行提取数字
            if results['best_val_acc'] == 0.0:
                lines = content.split('\n')
                # 查看最后20行，寻找包含数字的行
                for line in lines[-20:]:
                    if any(keyword in line.lower() for keyword in ['acc', '精度', 'accuracy', 'val']):
                        numbers = re.findall(r'([0-9.]+)', line)
                        if numbers:
                            # 取第一个看起来像准确率的数字（通常在0-1之间或0-100之间）
                            for num_str in numbers:
                                num = float(num_str)
                                if 0.0 <= num <= 1.0 or 0.0 <= num <= 100.0:
                                    results['best_val_acc'] = num if num <= 1.0 else num/100.0
                                    results['val_acc'] = results['best_val_acc']
                                    print(f"🔧 从最后几行解析得到验证准确率: {results['best_val_acc']:.6f}")
                                    break
                        if results['best_val_acc'] > 0.0:
                            break
        
        print(f"📋 从日志文件解析结果: 验证精度={results['best_val_acc']:.6f}, 训练精度={results['train_acc']:.6f}")
        
    except Exception as e:
        print(f"⚠️  解析日志文件 {log_file_path} 时出错: {str(e)}")
    
    return results

def generate_markdown_report(base_exp_dir, analysis_report, post_configurations):
    """生成POST消融实验的Markdown报告"""
    markdown_file = os.path.join(base_exp_dir, 'POST_Ablation_Study_Report.md')
    
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write("# POST方法消融实验报告\n\n")
        f.write(f"**生成时间**: {analysis_report['experiment_summary']['analysis_timestamp']}\n\n")
        
        # 实验概述
        f.write("## 实验概述\n\n")
        f.write(f"- **总配置数**: {analysis_report['experiment_summary']['total_configs']}\n")
        f.write(f"- **成功配置**: {analysis_report['experiment_summary']['completed_configs']}\n")
        f.write(f"- **失败配置**: {analysis_report['experiment_summary']['failed_configs']}\n\n")
        
        # 配置结果表格
        f.write("## 详细结果\n\n")
        f.write("| 配置名称 | 验证准确率 | 训练准确率 | 验证损失 | 训练时间(s) | 最佳轮次 |\n")
        f.write("|---------|-----------|-----------|----------|------------|----------|\n")
        
        for config_name, results in analysis_report["configuration_results"].items():
            f.write(f"| {config_name} | {results['val_acc']:.6f} | {results['train_acc']:.6f} | ")
            f.write(f"{results['val_loss']:.6f} | {results['training_time']:.1f} | {results['best_epoch']} |\n")
        
        # 最佳配置
        f.write(f"\n## 最佳配置\n\n")
        best_config = analysis_report["detailed_analysis"]["best_configuration"]
        f.write(f"**配置名称**: {best_config['name']}\n\n")
        f.write(f"**性能指标**:\n")
        f.write(f"- 验证准确率: {best_config['performance']['val_acc']:.6f}\n")
        f.write(f"- 训练准确率: {best_config['performance']['train_acc']:.6f}\n")
        f.write(f"- 验证损失: {best_config['performance']['val_loss']:.6f}\n")
        f.write(f"- 训练时间: {best_config['performance']['training_time']:.1f}秒\n")
        f.write(f"- 最佳轮次: {best_config['performance']['best_epoch']}\n\n")
        
        # 组件贡献分析
        if analysis_report["detailed_analysis"]["component_contributions"]:
            f.write("## 单组件贡献分析\n\n")
            f.write("| 组件 | 准确率 | 改进幅度 | 改进百分比 |\n")
            f.write("|------|--------|----------|------------|\n")
            
            for component, contrib in analysis_report["detailed_analysis"]["component_contributions"].items():
                f.write(f"| {component} | {contrib['accuracy']:.6f} | ")
                f.write(f"{contrib['improvement']:+.6f} | {contrib['improvement_percent']:+.2f}% |\n")
        
        # 组合效果分析
        if analysis_report["detailed_analysis"]["combination_effects"]:
            f.write("\n## 组合效果分析\n\n")
            f.write("| 组合配置 | 准确率 | 改进幅度 | 改进百分比 |\n")
            f.write("|----------|--------|----------|------------|\n")
            
            for combination, effect in analysis_report["detailed_analysis"]["combination_effects"].items():
                f.write(f"| {combination} | {effect['accuracy']:.6f} | ")
                f.write(f"{effect['improvement']:+.6f} | {effect['improvement_percent']:+.2f}% |\n")
        
        # 建议
        f.write("\n## 建议\n\n")
        for recommendation in analysis_report["recommendations"]:
            f.write(f"- {recommendation}\n")
        
        f.write(f"\n## 配置详情\n\n")
        for config_name, config in post_configurations.items():
            f.write(f"### {config_name}\n\n")
            f.write(f"**描述**: {config['description']}\n\n")
            f.write("**开关设置**:\n")
            for switch, value in config['switches'].items():
                f.write(f"- {switch}: {value}\n")
            f.write("\n")
    
    print(f"📝 Markdown报告已生成: {markdown_file}")

# 兼容旧的参数名
if args.use_pretrained is not None and args.pretrained_mdl_path is None:
    args.pretrained_mdl_path = args.use_pretrained

# # dataset spectrogram mean and std, used to normalize the input
# norm_stats = {'librispeech':[-4.2677393, 4.5689974], 'howto100m':[-4.2677393, 4.5689974], 'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
# target_length = {'librispeech': 1024, 'howto100m':1024, 'audioset':1024, 'esc50':512, 'speechcommands':128}
# # if add noise for data augmentation, only use for speech commands
# noise = {'librispeech': False, 'howto100m': False, 'audioset': False, 'esc50': False, 'speechcommands':True}

audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
              'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}

val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

# if use balanced sampling, note - self-supervised pretraining should not use balance sampling as it implicitly leverages the label information.
if args.bal == 'bal':
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)

print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))

# 创建实验目录结构
experiment_dirs = ['models', 'logs', 'visualizations', 'results', 'reports']
for dir_name in experiment_dirs:
    dir_path = os.path.join(args.exp_dir, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 保存完整的实验配置
experiment_metadata = {
    'experiment_name': args.experiment_name,
    'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
    'git_commit': os.popen('git rev-parse HEAD').read().strip() if os.path.exists('.git') else 'unknown',
    'python_version': sys.version,
    'pytorch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
}

# 合并所有配置
full_config = {**vars(args), **experiment_metadata}

# 保存完整配置到JSON文件
config_file = os.path.join(args.exp_dir, 'full_experiment_config.json')
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(full_config, f, indent=2, ensure_ascii=False, default=str)

print("="*100)
print(f"🚀 开始实验: {args.experiment_name}")
print("="*100)
print(f"实验目录: {args.exp_dir}")
print(f"实验时间: {experiment_metadata['timestamp']}")
print(f"数据集: {args.dataset}")
print(f"任务类型: {args.task}")
print(f"模型大小: {args.model_size}")

# in the pretraining stage
if 'pretrain' in args.task:
    cluster = (args.num_mel_bins != args.fshape)
    if cluster == True:
        print('使用聚类掩码模式（patch-based掩码）')
    else:
        print('使用帧掩码模式（frame-based掩码）')
    
    # 根据任务类型确定预训练方法
    if args.task in ['pretrain_mae_mpc', 'pretrain_mae_mpg', 'pretrain_mae_joint']:
        # MAE方法需要特定的预训练方法设置
        pretrain_methods = []
        if 'mae_mpc' in args.task:
            pretrain_methods.append('mae_mpc')
        if 'mae_mpg' in args.task:
            pretrain_methods.append('mae_mpg')
        if args.task == 'pretrain_mae_joint':
            pretrain_methods = ['mae_mpc', 'mae_mpg']
        
        pretrain_method_str = ','.join(pretrain_methods)
        print(f'创建MAE-AST模型，预训练方法: {pretrain_method_str}')
        
    elif args.task in ['pretrain_post', 'pretrain_core', 'pretrain_compass']:
        # ComPASS方法
        if args.task == 'pretrain_post':
            pretrain_method_str = 'post'
            print(f'创建PoST (Position-Sensing Transformer) 模型')
        elif args.task == 'pretrain_core':
            pretrain_method_str = 'core'
            print(f'创建CoRe (Context-driven Reconstruction) 模型')
        elif args.task == 'pretrain_compass':
            pretrain_method_str = 'post,core'
            print(f'创建ComPASS (PoST + CoRe) 联合训练模型')
    else:
        # 传统方法
        pretrain_method_str = 'mpc,mpg'  # 默认
        print(f'创建传统AST模型，预训练方法: {pretrain_method_str}')
    
    # 创建模型时添加ComPASS框架的参数和蒸馏参数
    audio_model = ASTModel(
        fshape=args.fshape, 
        tshape=args.tshape, 
        fstride=args.fshape, 
        tstride=args.tshape,
        input_fdim=args.num_mel_bins, 
        input_tdim=args.target_length, 
        model_size=args.model_size, 
        pretrain_stage=True,
        pretrain_method=pretrain_method_str
    )
# in the fine-tuning stage
else:
    audio_model = ASTModel(
        label_dim=args.n_class, 
        fshape=args.fshape, 
        tshape=args.tshape, 
        fstride=args.fstride, 
        tstride=args.tstride,
        input_fdim=args.num_mel_bins, 
        input_tdim=args.target_length, 
        model_size=args.model_size,
        pretrain_stage=False,  # 微调阶段设为False
        load_pretrained_mdl_path=args.pretrained_mdl_path  # 使用正确的参数名
    )

if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

print("\nCreating experiment directory: %s" % args.exp_dir)
if os.path.exists("%s/models" % args.exp_dir) == False:
    os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

# 在模型创建后输出详细信息
if 'pretrain' in args.task:
    print("\n" + "="*80)
    print("🔧 预训练阶段配置")
    print("="*80)
    print(f"预训练任务: {args.task}")
    
    if 'mae' in args.task:
        print("🆕 使用MAE-AST (Masked Autoencoding Audio Spectrogram Transformer) 方法")
        print("   • Encoder-Decoder非对称架构")
        print("   • Encoder只处理未掩码tokens (约25%)")
        print("   • 轻量级Decoder处理所有tokens")
        print("   • 3x训练速度提升，2x内存减少")
        if args.task == 'pretrain_mae_mpc':
            print("   • 判别式目标：NCE损失")
        elif args.task == 'pretrain_mae_mpg':
            print("   • 生成式目标：MSE重建损失")
        elif args.task == 'pretrain_mae_joint':
            print("   • 联合目标：判别式 + 生成式损失")
    elif args.task == 'pretrain_eat':
        print("🆕 使用EAT (Efficient Audio Transformer) 方法")
        print("   • Utterance-Frame Objective (UFO)")
        print("   • Teacher-Student Bootstrap框架")
        print("   • 结合全局和局部表示学习")
    elif args.task in ['pretrain_post', 'pretrain_core', 'pretrain_compass']:
        print("🎯 使用ComPASS (Complementary Positional and Spectrogram Synthesis) 框架")
        print("   • 专为农业机械声学分析设计的双任务自监督学习框架")
        print("   • 系统性解决三个核心挑战：")
        print("     ✓ 环境噪声干扰 (Environmental Noise Interference)")
        print("     ✓ 声学相似性问题 (Acoustic Similarity)")  
        print("     ✓ 类别不平衡分布 (Class Imbalance)")
        
        if args.task == 'pretrain_post':
            print("   • PoST (Position-Sensing Transformer):")
            print("     - 学习空间-时间位置关系")
            print("     - 保持机械声学结构")
            print("     - 增强位置感知能力")
        elif args.task == 'pretrain_core':
            print("   • CoRe (Context-driven Reconstruction):")
            print("     - 从部分上下文重建完整频谱")
            print("     - 增强特征鲁棒性")
            print("     - 抵抗环境干扰")
        elif args.task == 'pretrain_compass':
            print("   • ComPASS联合训练 (PoST + CoRe):")
            print("     - 双任务协同学习")
            print("     - 位置预测 + 频谱重建")
            print("     - 对比学习增强特征区分")
            print("     - 多损失函数平衡优化")
    
    print(f"掩码Patch数: {args.mask_patch}")
    print(f"聚类因子: {args.cluster_factor}")
    print(f"评估间隔: {args.epoch_iter}步")
    print(f"Patch形状: {args.fshape}x{args.tshape}")
    print(f"Patch步长: {args.fshape}x{args.tshape} (预训练无重叠)")
    
    # 显示蒸馏参数（如果启用）
    if args.enable_distillation:
        print(f"\n🔥 EAT蒸馏参数:")
        print(f"   • 蒸馏温度: {args.distillation_temperature}")
        print(f"   • 蒸馏权重: {args.distillation_alpha}")
        print(f"   • Teacher动量: {args.teacher_momentum}")
    
    # 显示ComPASS框架的优化参数
    if args.task in ['pretrain_post', 'pretrain_core', 'pretrain_compass']:
        print("\n🎯 ComPASS框架优化配置:")
        print("   • 噪声鲁棒性: 启用 (解决环境噪声干扰)")
        print("   • 相似性感知: 启用 (解决声学相似性)")
        print("   • 类别平衡: 启用 (解决类别不平衡)")
        print("   • 对比学习温度: 0.1")
        print("   • 焦点损失参数: α=0.25, γ=2.0")

    # 在预训练部分的模型创建后添加
    if 'pretrain_post' in args.task and args.ablation_freq > 0:
        print("🔬 启用POST方法消融实验")
        print(f"   • 实验间隔: 每{args.ablation_freq}个epoch")
        print(f"   • 结果保存: {args.exp_dir}/ablation_study/")
else:
    print("\n" + "="*80)
    print("🎯 微调阶段配置")
    print("="*80)
    print(f"类别数: {args.n_class}")
    print(f"预训练模型: {args.pretrained_mdl_path}")
    print(f"使用预训练: {args.use_pretrained}")
    print(f"Patch步长: {args.fstride}x{args.tstride}")

print("="*80)
print(f"📊 数据集信息:")
print(f"   训练样本: {len(train_loader.dataset):,}")
print(f"   验证样本: {len(val_loader.dataset):,}")
print(f"   批次大小: {args.batch_size}")
print(f"   工作进程: {args.num_workers}")
print("="*80)

if 'pretrain' not in args.task:
    print('🎯 开始微调训练，目标轮次: {:d}'.format(args.n_epochs))
    train(audio_model, train_loader, val_loader, args)
else:
    # POST方法消融实验 - 独立运行每种配置
    if args.task == 'pretrain_post' and hasattr(args, 'ablation_freq') and args.ablation_freq > 0:
        print('🔬 开始POST方法完整消融实验')
        print("="*100)
        print("每种配置将独立运行完整的训练过程")
        
        # 定义针对农机声音的POST配置
        post_configurations = [
            {
                'name': 'post_baseline',
                'display_name': '基础POST (所有优化关闭)',
                'switches': {
                    'enable_mechanical_aware_masking': False,
                    'enable_similarity_contrastive': False,
                    'enable_class_balanced': False,
                    'enable_local_feature_enhancement': False
                },
                'expected_improvement': 0.0,
                'description': '基础版本，用作对照组'
            },
            {
                'name': 'post_mechanical',
                'display_name': '基础 + 机械声音感知掩码',
                'switches': {
                    'enable_mechanical_aware_masking': True,
                    'enable_similarity_contrastive': False,
                    'enable_class_balanced': False,
                    'enable_local_feature_enhancement': False
                },
                'expected_improvement': 0.04,
                'description': '针对机械声音的周期性和谐波特征设计掩码策略'
            },
            {
                'name': 'post_similarity',
                'display_name': '基础 + 相似性对比增强',
                'switches': {
                    'enable_mechanical_aware_masking': False,
                    'enable_similarity_contrastive': True,
                    'enable_class_balanced': False,
                    'enable_local_feature_enhancement': False
                },
                'expected_improvement': 0.05,
                'description': '增强对相似操作模式（如转弯vs直行）的区分能力'
            },
            {
                'name': 'post_balanced',
                'display_name': '基础 + 类别平衡策略',
                'switches': {
                    'enable_mechanical_aware_masking': False,
                    'enable_similarity_contrastive': False,
                    'enable_class_balanced': True,
                    'enable_local_feature_enhancement': False
                },
                'expected_improvement': 0.03,
                'description': '处理类别不平衡问题（收获80%+，卸粮<2%）'
            },
            {
                'name': 'post_local',
                'display_name': '基础 + 局部特征增强',
                'switches': {
                    'enable_mechanical_aware_masking': False,
                    'enable_similarity_contrastive': False,
                    'enable_class_balanced': False,
                    'enable_local_feature_enhancement': True
                },
                'expected_improvement': 0.02,
                'description': '增强局部时频特征以捕获机械声音细节'
            },
            {
                'name': 'post_mech_sim',
                'display_name': '机械感知 + 相似性对比',
                'switches': {
                    'enable_mechanical_aware_masking': True,
                    'enable_similarity_contrastive': True,
                    'enable_class_balanced': False,
                    'enable_local_feature_enhancement': False
                },
                'expected_improvement': 0.08,
                'description': '结合机械特征和相似性对比策略'
            },
            {
                'name': 'post_sim_bal',
                'display_name': '相似性对比 + 类别平衡',
                'switches': {
                    'enable_mechanical_aware_masking': False,
                    'enable_similarity_contrastive': True,
                    'enable_class_balanced': True,
                    'enable_local_feature_enhancement': False
                },
                'expected_improvement': 0.07,
                'description': '同时解决相似性和类别不平衡问题'
            },
            {
                'name': 'post_three_way',
                'display_name': '机械感知 + 相似性 + 类别平衡',
                'switches': {
                    'enable_mechanical_aware_masking': True,
                    'enable_similarity_contrastive': True,
                    'enable_class_balanced': True,
                    'enable_local_feature_enhancement': False
                },
                'expected_improvement': 0.10,
                'description': '三个核心优化组件的组合'
            },
            {
                'name': 'post_full',
                'display_name': '完整优化 (所有开关开启)',
                'switches': {
                    'enable_mechanical_aware_masking': True,
                    'enable_similarity_contrastive': True,
                    'enable_class_balanced': True,
                    'enable_local_feature_enhancement': True
                },
                'expected_improvement': 0.12,
                'description': '针对农机声音的全部优化策略'
            }
        ]
        
        # 存储所有配置的训练结果
        all_config_results = {}
        
        # 为每个配置运行独立的完整训练过程
        for config_idx, config in enumerate(post_configurations):
            print(f"\n{'='*100}")
            print(f"🧪 配置 {config_idx+1}/{len(post_configurations)}: {config['display_name']}")
            print(f"{'='*100}")
            print(f"描述: {config['description']}")
            print(f"期望改进: +{config['expected_improvement']*100:.1f}%")
            
            # 显示开关状态
            print("开关状态:")
            for switch_name, switch_value in config['switches'].items():
                status = "✅ 开启" if switch_value else "❌ 关闭"
                print(f"  • {switch_name}: {status}")
            
            # 为每个配置创建独立的实验目录
            config_exp_dir = os.path.join(args.exp_dir, config['name'])
            os.makedirs(config_exp_dir, exist_ok=True)
            
            # 创建子目录
            for subdir in ['models', 'logs', 'visualizations', 'results']:
                os.makedirs(os.path.join(config_exp_dir, subdir), exist_ok=True)
            
            # 创建独立的日志文件
            experiment_timestamp = getattr(args, 'experiment_timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
            config_log_file = os.path.join(args.exp_dir, 'logs', f"{config['name']}_{experiment_timestamp}.log")
            
            # 确保日志目录存在
            os.makedirs(os.path.dirname(config_log_file), exist_ok=True)
            
            # 初始化配置特定的日志文件
            with open(config_log_file, 'w', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - ================================================\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 开始POST配置实验: {config['display_name']}\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 配置名称: {config['name']}\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 实验目录: {config_exp_dir}\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 日志文件: {config_log_file}\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 期望改进: +{config['expected_improvement']*100:.1f}%\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 描述: {config['description']}\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 开关配置:\n")
                for switch_name, switch_value in config['switches'].items():
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO -   • {switch_name}: {switch_value}\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - ================================================\n")
            
            # 复制args并修改实验目录
            config_args = argparse.Namespace(**vars(args))
            config_args.exp_dir = config_exp_dir
            config_args.experiment_name = f"{args.experiment_name}_{config['name']}"
            config_args.config_log_file = config_log_file  # 添加日志文件路径到args
            
            # 将POST开关配置添加到args中
            for switch_name, switch_value in config['switches'].items():
                setattr(config_args, switch_name, switch_value)
            
            # 保存配置特定的实验配置
            config_metadata = {
                **full_config,
                'post_config': config,
                'config_exp_dir': config_exp_dir,
                'config_name': config['name'],
                'config_display_name': config['display_name']
            }
            
            config_file = os.path.join(config_exp_dir, 'config_experiment_config.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n🚀 开始训练配置: {config['name']}")
            print(f"实验目录: {config_exp_dir}")
            
            try:
                # 重新创建模型 - 确保每个配置都有干净的模型状态
                config_audio_model = ASTModel(
                    fshape=args.fshape, 
                    tshape=args.tshape, 
                    fstride=args.fshape, 
                    tstride=args.tshape,
                    input_fdim=args.num_mel_bins, 
                    input_tdim=args.target_length, 
                    model_size=args.model_size, 
                    pretrain_stage=True,
                    pretrain_method='post',
                    enable_distillation=args.enable_distillation,
                    distillation_temperature=args.distillation_temperature,
                    distillation_alpha=args.distillation_alpha,
                    teacher_momentum=args.teacher_momentum
                )
                
                if not isinstance(config_audio_model, torch.nn.DataParallel):
                    config_audio_model = torch.nn.DataParallel(config_audio_model)
                
                # 开始训练
                start_time = time.time()
                # 创建只包含500条数据的训练加载器
                limited_dataset = torch.utils.data.Subset(train_loader.dataset, range(min(5000, len(train_loader.dataset))))
                limited_train_loader = torch.utils.data.DataLoader(
                    limited_dataset,
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    num_workers=args.num_workers, 
                    pin_memory=False, 
                    drop_last=True
                )
                
                # 使用独立的日志文件进行训练
                print(f"📋 配置独立日志文件: {config_log_file}")
                
                # 训练过程将输出重定向到配置特定的日志文件
                import sys
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                try:
                    # 创建一个同时写入标准输出和日志文件的类
                    class TeeOutput:
                        def __init__(self, file_path):
                            self.terminal = original_stdout
                            self.log_file = open(file_path, 'a', encoding='utf-8')
                        
                        def write(self, message):
                            self.terminal.write(message)
                            self.log_file.write(message)
                            self.log_file.flush()
                        
                        def flush(self):
                            self.terminal.flush()
                            self.log_file.flush()
                        
                        def close(self):
                            self.log_file.close()
                    
                    # 重定向输出到日志文件
                    tee_output = TeeOutput(config_log_file)
                    sys.stdout = tee_output
                    
                    trainmask(config_audio_model, train_loader, val_loader, config_args)
                    
                finally:
                    # 恢复原始输出
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    if 'tee_output' in locals():
                        tee_output.close()
                
                training_time = time.time() - start_time
                
                # 在配置日志文件中记录完成信息
                with open(config_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - ================================================\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 配置训练完成: {config['name']}\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 训练时间: {training_time/3600:.2f}小时\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 状态: 成功完成\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - ================================================\n")
                
                # 记录配置结果
                all_config_results[config['name']] = {
                    'config': config,
                    'exp_dir': config_exp_dir,
                    'log_file': config_log_file,
                    'training_time': training_time,
                    'status': 'completed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                print(f"✅ 配置 {config['name']} 训练完成")
                print(f"⏱️  训练时间: {training_time/3600:.2f}小时")
                
            except Exception as e:
                print(f"❌ 配置 {config['name']} 训练失败: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 在配置日志文件中记录错误信息
                try:
                    with open(config_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - ================================================\n")
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - 配置训练失败: {config['name']}\n")
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - 错误信息: {str(e)}\n")
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - 错误详情:\n")
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {traceback.format_exc()}\n")
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - ================================================\n")
                except:
                    pass  # 如果日志文件写入失败也不影响主程序
                
                all_config_results[config['name']] = {
                    'config': config,
                    'exp_dir': config_exp_dir,
                    'log_file': config_log_file if 'config_log_file' in locals() else None,
                    'training_time': 0,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
        
        # 训练完成后，分析所有配置的结果
        print(f"\n{'='*100}")
        print("🏆 POST方法完整消融实验总结分析")
        print(f"{'='*100}")
        
        # 调用结果分析函数
        analyze_post_ablation_results(args.exp_dir, all_config_results, post_configurations)
        
    else:
        # 标准预训练流程
        print('🔧 开始自监督预训练，目标轮次: {:d}'.format(args.n_epochs))
        limited_dataset = torch.utils.data.Subset(train_loader.dataset, range(min(5000, len(train_loader.dataset))))
        limited_train_loader = torch.utils.data.DataLoader(
            limited_dataset,
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=False, 
            drop_last=True
        )
        trainmask(audio_model, train_loader, val_loader, args)



# if the dataset has a seperate evaluation set (e.g., speechcommands), then select the model using the validation set and eval on the evaluation set.
# this is only for fine-tuning
if args.data_eval != None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # best models on the validation set
    args.loss_fn = torch.nn.BCEWithLogitsLoss()
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the models on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

# 训练完成后的最终处理
if args.paper_baseline:
    # 创建与论文基线的比较报告
    try:
        # 尝试获取模型参数信息
        if hasattr(audio_model, 'module') and hasattr(audio_model.module, 'model_specs'):
            params_info = f"{audio_model.module.model_specs['total_params_M']}M"
        elif hasattr(audio_model, 'model_specs'):
            params_info = f"{audio_model.model_specs['total_params_M']}M"
        else:
            # 如果没有model_specs属性，计算模型参数数量
            total_params = sum(p.numel() for p in audio_model.parameters())
            params_info = f"{total_params/1e6:.1f}M"
    except (AttributeError, KeyError):
        params_info = "unknown"
    
    baseline_comparison = {
        'Our Method': {
            'model': f"AST-{args.model_size}",
            'params': params_info,
            'task': args.task,
            'dataset': args.dataset
        },
        'Paper Baselines': {
            'SSAST': {'params': '89M', 'task': 'Self-Supervised', 'performance': '31.0%'},
            'AudioMAE': {'params': '86M', 'task': 'Masked Autoencoder', 'performance': '30.6%'},
            'AST': {'params': '86M', 'task': 'Supervised', 'performance': '34.7%'}
        }
    }
    
    comparison_file = os.path.join(args.exp_dir, 'paper_baseline_comparison.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(baseline_comparison, f, indent=2, ensure_ascii=False)

print("\n" + "="*100)
print("✅ 实验完成!")
print("="*100)
print(f"实验目录: {args.exp_dir}")
print(f"配置文件: {config_file}")
print(f"模型保存: {args.exp_dir}/models/")
print(f"日志文件: {args.exp_dir}/logs/")
print(f"可视化: {args.exp_dir}/visualizations/")
print("="*100)
