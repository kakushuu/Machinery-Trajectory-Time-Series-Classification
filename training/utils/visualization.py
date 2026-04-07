#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免Tkinter相关错误
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import librosa
import librosa.display
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error
import seaborn as sns

class SSASTVisualizer:
    """SSAST模型可视化工具，用于生成预训练过程中的各种视觉输出"""
    
    def __init__(self, save_dir='./visualizations'):
        """
        初始化SSAST可视化工具
        
        Args:
            save_dir: 可视化结果保存路径
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体支持 (如果有需要)
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        
        # 设置可视化样式
        plt.style.use('seaborn-v0_8-whitegrid')
        self.cmap = plt.cm.viridis
        self.mask_cmap = ListedColormap(['white', 'red'])
    
    def visualize_spectrogram(self, spec, title="Mel Spectrogram", filename="spectrogram.png"):
        """
        可视化原始梅尔频谱图
        
        Args:
            spec: 梅尔频谱图 [频率, 时间]
            title: 图表标题
            filename: 保存的文件名
        """
        plt.figure(figsize=(10, 6))
        
        if torch.is_tensor(spec):
            spec = spec.cpu().numpy()
        
        librosa.display.specshow(
            spec, 
            sr=16000, 
            x_axis='time', 
            y_axis='mel', 
            cmap=self.cmap
        )
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        print(f"Spectrogram saved to: {os.path.join(self.save_dir, filename)}")
    
    
    def visualize_real_pred_result_np(self, original, masked, predicted, mask, title="Prediction Results", 
                              filename="pred_result_npresults.png"):
        """
        可视化预测结果
        
        Args:
            original: 原始梅尔频谱图
            masked: 掩码后的梅尔频谱图
            predicted: 预测结果
            mask: 掩码矩阵
            title: 图表标题
            filename: 保存的文件名
        """
        plt.figure(figsize=(15, 10))
        
        if torch.is_tensor(original):
            original = original.cpu().numpy()
        if torch.is_tensor(masked):
            masked = masked.cpu().numpy()
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # 原始频谱图
        plt.subplot(221)
        librosa.display.specshow(
            original, 
            sr=16000, 
            x_axis='time', 
            y_axis='mel', 
            cmap=self.cmap
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title("Original Spectrogram")
        
        # 掩码后的频谱图
        plt.subplot(222)
        librosa.display.specshow(
            masked, 
            sr=16000, 
            x_axis='time', 
            y_axis='mel', 
            cmap=self.cmap
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title("Masked Spectrogram")
        
        # 预测结果
        plt.subplot(223)
        librosa.display.specshow(
            predicted, 
            sr=16000, 
            x_axis='time', 
            y_axis='mel', 
            cmap=self.cmap
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title("Prediction Results")
        
        # 掩码区域
        plt.subplot(224)
        plt.imshow(mask, aspect='auto', cmap=self.mask_cmap, alpha=0.7)
        plt.title("Masked Regions (Red)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        # 计算掩码区域的均方误差
        masked_mse = mean_squared_error(
            original[mask == 1], 
            predicted[mask == 1]
        )
        print(f"Prediction results saved to: {os.path.join(self.save_dir, filename)}")
        print(f"Masked Region MSE: {masked_mse:.4f}")
        
        return masked_mse
    
    def visualize_pred_result_nperror(self, original, predicted, mask, title="Prediction Error", 
                                   filename="pred_result_nperror.png"):
        """
        可视化预测误差
        
        Args:
            original: 原始梅尔频谱图
            predicted: 预测结果
            mask: 掩码矩阵
            title: 图表标题
            filename: 保存的文件名
        """
        plt.figure(figsize=(12, 8))
        
        if torch.is_tensor(original):
            original = original.cpu().numpy()
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # 计算误差
        error = np.abs(original - predicted)
        
        # 原始频谱图
        plt.subplot(221)
        librosa.display.specshow(
            original, 
            sr=16000, 
            x_axis='time', 
            y_axis='mel', 
            cmap=self.cmap
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title("Original Spectrogram")
        
        # 预测结果
        plt.subplot(222)
        librosa.display.specshow(
            predicted, 
            sr=16000, 
            x_axis='time', 
            y_axis='mel', 
            cmap=self.cmap
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title("Prediction Results")
        
        # 误差图
        plt.subplot(223)
        im = plt.imshow(error, aspect='auto', cmap='hot')
        plt.colorbar(im, format='%.2f')
        plt.title("Prediction Error (Absolute)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        
        # 掩码区域误差
        plt.subplot(224)
        masked_error = np.zeros_like(error)
        masked_error[mask == 1] = error[mask == 1]
        im = plt.imshow(masked_error, aspect='auto', cmap='hot')
        plt.colorbar(im, format='%.2f')
        plt.title("Masked Region Error")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        print(f"Prediction error saved to: {os.path.join(self.save_dir, filename)}")
    
    def visualize_grid_path(self, spec, patch_size, stride, mask=None, title="Patch Grid Path", 
                             filename="grid_path.png"):
        """
        可视化完整路径网格与掩码区域
        
        Args:
            spec: 梅尔频谱图 [频率, 时间]
            patch_size: 元组，表示patch的尺寸 (频率, 时间)
            stride: 元组，表示步长 (频率, 时间)
            mask: 可选，掩码矩阵
            title: 图表标题
            filename: 保存的文件名
        """
        # 设置Seaborn样式
        sns.set_style("dark") # 改为dark样式去掉白色网格线
        sns.set_context("notebook", font_scale=1.2)
        
        plt.figure(figsize=(12, 8))
        
        if torch.is_tensor(spec):
            spec = spec.cpu().numpy()
        if mask is not None and torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # 获取频谱图尺寸
        freq_dim, time_dim = spec.shape
        
        # 创建主要的绘图区域
        plt.imshow(spec, aspect='auto', origin='lower', cmap=sns.color_palette("viridis", as_cmap=True))
        
        # 创建网格线
        fshape, tshape = patch_size
        fstride, tstride = stride
        
        # 绘制水平线 (频率轴)
        for i in range(0, freq_dim, fstride):
            plt.axhline(y=i, color="white", linestyle='--', alpha=0.5, linewidth=2.0)
        
        # 绘制垂直线 (时间轴)
        for i in range(0, time_dim, tstride):
            plt.axvline(x=i, color="white", linestyle='--', alpha=0.5, linewidth=2.0)
        
        # 如果提供了掩码，标记掩码区域
        if mask is not None:
            # 找到掩码为1的位置
            mask_indices = np.where(mask == 1)
            
            # 找到patch边界
            freq_patches = freq_dim // fstride
            time_patches = time_dim // tstride
            
            for i in range(freq_patches):
                for j in range(time_patches):
                    # 检查当前patch是否包含掩码区域
                    f_start = i * fstride
                    f_end = min(f_start + fshape, freq_dim)
                    t_start = j * tstride
                    t_end = min(t_start + tshape, time_dim)
                    
                    patch_mask = mask[f_start:f_end, t_start:t_end]
                    
                    # 如果patch包含掩码区域，用红色标记
                    if np.any(patch_mask == 1):
                        rect = patches.Rectangle(
                            (t_start, f_start), 
                            t_end - t_start, 
                            f_end - f_start, 
                            linewidth=2, 
                            edgecolor="red",
                            facecolor='yellow',
                            alpha=0.8
                        )
                        plt.gca().add_patch(rect)
        
        plt.title(title, pad=20)
        plt.xlabel("Time", labelpad=10)
        plt.ylabel("Frequency", labelpad=10)
        
        # 调整布局和边距
        plt.tight_layout()
        
        # 保存图像时使用深色背景
        with sns.axes_style("dark"):
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grid path visualization saved to: {os.path.join(self.save_dir, filename)}")
    
    def visualize_training_progress(self, loss_history, val_loss_history=None, steps_history=None, 
                                    title="Training Progress", filename="training_progress.png"):
        """Visualize training progress"""
        plt.figure(figsize=(10, 6))
        
        # Check if data exists
        if not loss_history or len(loss_history) == 0:
            plt.text(0.5, 0.5, "No training data available yet", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
            plt.close()
            print(f"Training progress visualization saved (empty): {os.path.join(self.save_dir, filename)}")
            return
        
        # Ensure all data are scalars and handle NaN values
        loss_history = [float(l) if not np.isnan(float(l)) else None for l in loss_history]
        loss_history = [l for l in loss_history if l is not None]
        
        # Ensure data is not empty
        if not loss_history:
            plt.text(0.5, 0.5, "Training loss data invalid or all NaN", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
            plt.close()
            print(f"Loss visualization saved (empty): {os.path.join(self.save_dir, filename)}")
            return
        
        # Process validation loss
        valid_val_indices = []
        filtered_val_loss = []
        
        if val_loss_history is not None:
            for i, val in enumerate(val_loss_history):
                if val is not None and not np.isnan(float(val)):
                    valid_val_indices.append(i)
                    filtered_val_loss.append(float(val))
        
        # Use steps_history as x-axis if provided
        if steps_history is not None and len(steps_history) >= len(loss_history):
            steps = steps_history[:len(loss_history)]
            # Plot training loss
            plt.plot(steps, loss_history, 'o-', label='Training Loss', color='blue')
            
            # Plot validation loss if available
            if filtered_val_loss and valid_val_indices:
                valid_steps = [steps_history[i] for i in valid_val_indices if i < len(steps_history)]
                if valid_steps and len(valid_steps) == len(filtered_val_loss):
                    plt.plot(valid_steps, filtered_val_loss, 's-', label='Validation Loss', color='red')
            
            plt.xlabel("Training Steps")
        else:
            # Plot training loss
            plt.plot(loss_history, 'o-', label='Training Loss', color='blue')
            
            # Plot validation loss if available
            if filtered_val_loss and valid_val_indices:
                plt.plot(valid_val_indices, filtered_val_loss, 's-', label='Validation Loss', color='red')
            
            plt.xlabel("Iterations")
        
        plt.title(title)
        plt.ylabel("Loss Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        print(f"Training progress visualization saved: {os.path.join(self.save_dir, filename)}")

    def visualize_precision_progress(self, train_precision_history, val_precision_history=None, steps_history=None,
                                    title="Precision Progress", filename="precision_progress.png"):
        """
        可视化精度进度
        
        Args:
            train_precision_history: 训练精度历史记录
            val_precision_history: 可选，验证精度历史记录
            steps_history: 可选，步骤历史记录，用于x轴
            title: 图表标题
            filename: 保存的文件名
        """
        plt.figure(figsize=(10, 6))
        
        # 检查是否有数据
        if not train_precision_history:
            plt.text(0.5, 0.5, "No training data available yet", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
            plt.close()
            print(f"Precision progress visualization saved (empty) to: {os.path.join(self.save_dir, filename)}")
            return
        
        # 处理可能包含None的验证精度
        valid_val_indices = []
        filtered_val_precision = []
        
        if val_precision_history is not None:
            for i, val in enumerate(val_precision_history):
                if val is not None:
                    valid_val_indices.append(i)
                    filtered_val_precision.append(val)
        
        # 如果提供了步骤历史，则使用它作为x轴
        if steps_history is not None and len(steps_history) == len(train_precision_history):
            # 绘制训练精度
            plt.plot(steps_history, train_precision_history, label='Training Precision', color='blue')
            
            # 如果有验证精度，也绘制
            if filtered_val_precision and valid_val_indices:
                valid_steps = [steps_history[i] for i in valid_val_indices]
                plt.plot(valid_steps, filtered_val_precision, label='Validation Precision', color='red')
            
            plt.xlabel("Training Steps")
        else:
            # 绘制训练精度
            plt.plot(train_precision_history, label='Training Precision', color='blue')
            
            # 如果有验证精度，也绘制
            if filtered_val_precision and valid_val_indices:
                plt.plot(valid_val_indices, filtered_val_precision, label='Validation Precision', color='red')
            
            plt.xlabel("Iterations")
        
        plt.title(title)
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        print(f"Precision progress visualization saved to: {os.path.join(self.save_dir, filename)}")

    def visualize_attention_map(self, attention_weights, title="Attention Map", filename="attention_map.png"):
        """Visualize attention weights"""
        if attention_weights is None:
            print("No attention weight data available for visualization")
            return
        
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.cpu().numpy()
        
        # Validate data
        if not isinstance(attention_weights, np.ndarray) or attention_weights.size == 0:
            print("Attention weight data invalid or empty")
            return
        
        # Handle dimension issues - may need reshape
        if len(attention_weights.shape) < 3:
            print(f"Warning: Attention weight shape not expected 3D [heads, seq_len, seq_len], current: {attention_weights.shape}")
            if len(attention_weights.shape) == 2:
                # Assume single attention head
                attention_weights = attention_weights.reshape(1, *attention_weights.shape)
        
        num_heads = attention_weights.shape[0]
        
        plt.figure(figsize=(15, num_heads * 4))
        
        for i in range(num_heads):
            plt.subplot(num_heads, 1, i+1)
            plt.imshow(attention_weights[i], cmap='viridis')
            plt.colorbar(format='%.2f', label='Attention Intensity')
            plt.title(f"Attention Head #{i+1} (Bright colors show high attention areas)")
            plt.xlabel("Target Position Index")
            plt.ylabel("Query Position Index")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        print(f"Attention map visualization saved: {os.path.join(self.save_dir, filename)}")
        
    def visualize_attention_progress(self, attention_history, steps_history=None, head_indices=None,
                                     title="Attention Progress", filename="attention_progress.png"):
        """
        可视化注意力权重随时间的变化
        
        Args:
            attention_history: 注意力权重历史记录列表，每个元素是一个注意力权重矩阵 [头数, 序列长度, 序列长度]
            steps_history: 可选，步骤历史记录，用于x轴
            head_indices: 可选，要可视化的头索引列表，默认显示所有头
            title: 图表标题
            filename: 保存的文件名
        """
        if not attention_history:
            print("Attention history is empty, cannot generate visualization")
            return
            
        # 确保所有注意力权重都是numpy数组
        processed_history = []
        for attn in attention_history:
            if torch.is_tensor(attn):
                processed_history.append(attn.cpu().numpy())
            else:
                processed_history.append(attn)
                
        # 获取头数和序列长度
        num_heads = processed_history[0].shape[0]
        
        # 如果未指定头索引，则显示所有头
        if head_indices is None:
            head_indices = list(range(num_heads))
        else:
            head_indices = [i for i in head_indices if i < num_heads]
            
        # 计算每个头的平均注意力分数随时间的变化
        head_avg_scores = []
        for head_idx in head_indices:
            scores = []
            for attn in processed_history:
                # 计算该头的平均注意力分数
                scores.append(np.mean(attn[head_idx]))
            head_avg_scores.append(scores)
            
        plt.figure(figsize=(12, 8))
        
        # 如果提供了步骤历史，则使用它作为x轴
        x_values = steps_history if steps_history is not None else list(range(len(processed_history)))
        
        # 绘制每个头的平均注意力分数变化
        for i, scores in enumerate(head_avg_scores):
            plt.plot(x_values[:len(scores)], scores, label=f"Attention Head #{head_indices[i]+1}")
            
        plt.title(title)
        plt.xlabel("Training Steps")
        plt.ylabel("Average Attention Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        print(f"Attention progress visualization saved to: {os.path.join(self.save_dir, filename)}")
        
    def visualize_attention_distribution(self, attention_weights, steps=None, title="Attention Distribution", 
                                        filename="attention_distribution.png"):
        """
        可视化注意力分布
        
        Args:
            attention_weights: 注意力权重矩阵 [头数, 序列长度, 序列长度] 或历史记录列表
            steps: 可选，如果提供了多个注意力权重，指定要可视化的步骤索引
            title: 图表标题
            filename: 保存的文件名
        """
        # 确定是单个注意力矩阵还是历史记录
        is_history = isinstance(attention_weights, list)
        
        # 如果是历史记录，但未指定步骤，则使用最后一步
        if is_history and steps is None:
            attention_weights = attention_weights[-1]
            is_history = False
            
        # 如果是历史记录，且指定了步骤，则选择对应步骤的注意力矩阵
        elif is_history and steps is not None:
            if isinstance(steps, int):
                attention_weights = attention_weights[min(steps, len(attention_weights)-1)]
                is_history = False
            else:
                # 保持历史记录，仅选择指定步骤
                attention_weights = [attention_weights[min(s, len(attention_weights)-1)] for s in steps]
                
        # 如果现在是单个注意力矩阵
        if not is_history:
            if torch.is_tensor(attention_weights):
                attention_weights = attention_weights.cpu().numpy()
                
            num_heads = attention_weights.shape[0]
            
            plt.figure(figsize=(15, 10))
            
            # 绘制每个头的注意力分布直方图
            for i in range(min(num_heads, 9)):  # 最多显示9个头
                plt.subplot(3, 3, i+1)
                plt.hist(attention_weights[i].flatten(), bins=50, alpha=0.7)
                plt.title(f"Attention Head #{i+1}")
                plt.xlabel("Attention Score")
                plt.ylabel("Frequency")
                
            plt.tight_layout()
            plt.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.9)
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
            plt.close()
            
            print(f"Attention distribution visualization saved to: {os.path.join(self.save_dir, filename)}")
            
        # 如果是多个步骤的历史记录
        else:
            # 确保所有注意力权重都是numpy数组
            processed_attn = []
            for attn in attention_weights:
                if torch.is_tensor(attn):
                    processed_attn.append(attn.cpu().numpy())
                else:
                    processed_attn.append(attn)
                    
            num_steps = len(processed_attn)
            num_heads = processed_attn[0].shape[0]
            head_idx = 0  # 默认显示第一个头
            
            plt.figure(figsize=(15, 10))
            
            # 为每个步骤绘制注意力分布直方图
            for i in range(min(num_steps, 9)):  # 最多显示9个步骤
                plt.subplot(3, 3, i+1)
                plt.hist(processed_attn[i][head_idx].flatten(), bins=50, alpha=0.7)
                plt.title(f"Step {steps[i]}, Head #{head_idx+1}")
                plt.xlabel("Attention Score")
                plt.ylabel("Frequency")
                
            plt.tight_layout()
            plt.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.9)
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
            plt.close()
            
            print(f"Multiple steps attention distribution visualization saved to: {os.path.join(self.save_dir, filename)}")

    def visualize_discriminative_mask_phase(self, original_features, mask,
                                      real_pred_result_np, pred_result_np,
                                   step, task_name, filename="discriminative_mask_visualization.png", fshape=16, tshape=16):
        """
        Visualize pretrain mask phase comprehensive results
        """
        import datetime
        
        # 设置Seaborn样式，与visualize_grid_path保持一致
        sns.set_style("dark")  # 改为dark样式去掉白色网格线
        sns.set_context("notebook", font_scale=1.2)
        
        # Ensure all inputs are numpy arrays
        if torch.is_tensor(original_features):
            original_features = original_features.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if torch.is_tensor(real_pred_result_np):
            real_pred_result_np = real_pred_result_np.cpu().numpy()
        if torch.is_tensor(pred_result_np):
            pred_result_np = pred_result_np.cpu().numpy()
            
        # Ensure input dimensions are correct
        original_features = np.squeeze(original_features)
        mask = np.squeeze(mask)
        real_pred_result_np = np.squeeze(real_pred_result_np)
        pred_result_np = np.squeeze(pred_result_np)
        # Ensure dimension consistency
        if original_features.shape != real_pred_result_np.shape:
            print(f"Warning: Shape mismatch - original: {original_features.shape}, real_pred_result_np: {real_pred_result_np.shape}")
            if original_features.shape[::-1] == real_pred_result_np.shape:
                real_pred_result_np = real_pred_result_np.transpose()
                mask = mask.transpose()
                pred_result_np = pred_result_np.transpose()
        # 获取掩码尺寸
        freq_dim, time_dim = mask.shape
        
        # 直接将mask二值化，掩码值大于0的位置认为是掩码区域
        binary_mask = (mask > 0).astype(np.float32)
        
        # 创建patch级别的掩码状态数组
        freq_patches = freq_dim // fshape
        time_patches = time_dim // tshape
        patch_mask_status = np.zeros((freq_patches, time_patches))
        
        # 确定每个patch的掩码状态
        for i in range(freq_patches):
            for j in range(time_patches):
                f_start = i * fshape
                f_end = min(f_start + fshape, freq_dim)
                t_start = j * tshape
                t_end = min(t_start + tshape, time_dim)
                
                # 检查当前patch中掩码像素的百分比
                patch_area = (f_end - f_start) * (t_end - t_start)
                mask_pixels = np.sum(binary_mask[f_start:f_end, t_start:t_end])
                mask_ratio = mask_pixels / patch_area
                
                # 如果超过50%的像素是掩码，将此patch标记为掩码区域
                if mask_ratio > 0.5:
                    patch_mask_status[i, j] = 1
        
        # 计算掩码覆盖率
        mask_count = int(np.sum(binary_mask))
        mask_percentage = (mask_count / (freq_dim * time_dim)) * 100
        
        # 计算预测误差
        error = np.abs(original_features - real_pred_result_np)
        error_ = np.abs(original_features - pred_result_np)
        masked_error = error * binary_mask  # 只关注掩码区域
        
        # 计算评估指标
        masked_indices = binary_mask > 0
        if np.any(masked_indices):  # 防止除零错误
            mse = np.mean(np.square(original_features[masked_indices] - real_pred_result_np[masked_indices]))
            mse_ = np.mean(np.square(original_features[masked_indices] - pred_result_np[masked_indices]))
            mae = np.mean(np.abs(original_features[masked_indices] - real_pred_result_np[masked_indices]))
            mae_ = np.mean(np.abs(original_features[masked_indices] - pred_result_np[masked_indices]))
            max_error = np.max(np.abs(original_features[masked_indices] - real_pred_result_np[masked_indices]))
            max_error_ = np.max(np.abs(original_features[masked_indices] - pred_result_np[masked_indices]))
            percentile_90 = np.percentile(np.abs(original_features[masked_indices] - real_pred_result_np[masked_indices]), 90)
            percentile_90_ = np.percentile(np.abs(original_features[masked_indices] - pred_result_np[masked_indices]), 90)
        else:
            mse, mae, max_error, percentile_90 = 0, 0, 0, 0
        
        # 创建patch网格可视化
        patch_grid = np.zeros_like(binary_mask, dtype=float)
        patch_grid[binary_mask > 0] = 1.0  # 掩码区域设为1.0（洋红色）
        
        # 增大图形尺寸以容纳更多内容
        plt.figure(figsize=(14, 10))
        
        # 辅助函数：在子图上添加patch网格线，使用与visualize_grid_path一致的白色虚线
        def add_patch_grid(ax):
            # 绘制垂直线（时间轴）
            for t in range(0, time_dim, tshape):
                ax.axvline(x=t, color="white", linestyle='--', alpha=0.5, linewidth=2.0)
            # 绘制水平线（频率轴）
            for f in range(0, freq_dim, fshape):
                ax.axhline(y=f, color="white", linestyle='--', alpha=0.5, linewidth=2.0)
        
        # 1. Top-left: Original audio features
        ax1 = plt.subplot(3, 2, 1)
        plt.imshow(original_features, aspect='auto', origin='lower', cmap=sns.color_palette("viridis", as_cmap=True))
        plt.title("Original Audio Features", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        add_patch_grid(ax1)  # 添加patch网格线
        
        # 2. Top-right: Masked regions
        ax2 = plt.subplot(3, 2, 2)
        plt.imshow(original_features, aspect='auto', origin='lower', cmap=sns.color_palette("viridis", as_cmap=True))
        
        # 添加patch网格线
        add_patch_grid(ax2)
        
        # 找到patch边界并标记掩码区域，与visualize_grid_path相同的实现方式
        marked_patches = 0
        
        for i in range(freq_patches):
            for j in range(time_patches):
                # 检查当前patch是否是掩码区域
                if patch_mask_status[i, j] > 0:
                    marked_patches += 1
                    f_start = i * fshape
                    f_end = min(f_start + fshape, freq_dim)
                    t_start = j * tshape
                    t_end = min(t_start + tshape, time_dim)
                    
                    rect = patches.Rectangle(
                        (t_start, f_start), 
                        t_end - t_start, 
                        f_end - f_start, 
                        linewidth=2, 
                        edgecolor="red",
                        facecolor='yellow',
                        alpha=0.8
                    )
                    ax2.add_patch(rect)
        
        plt.title(f"Marked Masked Patches ({marked_patches})", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        
        # 3. Mid-left: Complete real_pred_result_np results - 参考visualize_audiomae_listen实现
        ax3 = plt.subplot(3, 2, 3)
        plt.imshow(real_pred_result_np, aspect='auto', origin='lower', cmap=sns.color_palette("viridis", as_cmap=True))
        plt.title("Model Reconstructed Features", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        add_patch_grid(ax3)  # 添加patch网格线
        
        # 标记掩码区域的patch边界（需要预测的区域）
        for i in range(freq_patches):
            for j in range(time_patches):
                # 如果这个patch是掩码区域(需要预测)
                if patch_mask_status[i, j] > 0:
                    f_start = i * fshape
                    f_end = min(f_start + fshape, freq_dim)
                    t_start = j * tshape
                    t_end = min(t_start + tshape, time_dim)
                    
                    rect = patches.Rectangle(
                        (t_start, f_start),
                        t_end - t_start,
                        f_end - f_start,
                        linewidth=2,
                        edgecolor="red",
                        facecolor='none',
                        alpha=0.6
                    )
                    ax3.add_patch(rect)
        
        # 4. Mid-right: Prediction errors
        ax4 = plt.subplot(3, 2, 4)
        plt.imshow(masked_error, aspect='auto', origin='lower', cmap=sns.color_palette("plasma", as_cmap=True))
        plt.title("Prediction Error Distribution", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        add_patch_grid(ax4)  # 添加patch网格线
        
        # 添加colorbar
        plt.colorbar(label='Error Magnitude')
        
        # 5. Bottom-left: 叠加原始特征和预测特征的对比图
        ax5 = plt.subplot(3, 2, 5)
        
        # 将掩码区域的原始特征和预测特征转换为散点图数据
        mask_indices = np.where(binary_mask > 0)
        if len(mask_indices[0]) > 0:  # 确保有掩码区域
            x_values = original_features[mask_indices]  # 原始特征值作为x轴
            y_values = real_pred_result_np[mask_indices]         # 预测特征值作为y轴
            
            # 为了控制点的数量，可以随机抽样
            if len(x_values) > 5000:
                idx = np.random.choice(len(x_values), 5000, replace=False)
                x_sample = x_values[idx]
                y_sample = y_values[idx]
            else:
                x_sample = x_values
                y_sample = y_values
            
            # 计算相关系数
            correlation = np.corrcoef(x_values, y_values)[0, 1] if len(x_values) > 1 else 0
            
            # 散点图：x轴为原始值，y轴为预测值
            plt.scatter(x_sample, y_sample, c='blue', alpha=0.5, s=10)
            
            # 添加对角线（理想预测线）
            min_val = min(np.min(x_values), np.min(y_values))
            max_val = max(np.max(x_values), np.max(y_values))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
            
            # 添加回归线以显示预测趋势
            if len(x_values) > 1:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                line_x = np.linspace(min_val, max_val, 100)
                line_y = slope * line_x + intercept
                plt.plot(line_x, line_y, 'g-', label=f'Trend Line (r={r_value:.2f})')
            
            plt.xlabel("Original Feature Values")
            plt.ylabel("Predicted Feature Values")
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.title(f"Prediction Correlation Plot (r={correlation:.3f})", fontsize=11, pad=10)
        else:
            plt.text(0.5, 0.5, "No masked area data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax5.transAxes)
            plt.title("Prediction Correlation Plot", fontsize=11, pad=10)
        
        # 6. Bottom-right: 掩码区域预测准确度热图 - 参考visualize_audiomae_listen实现
        ax6 = plt.subplot(3, 2, 6)
        
        # 准备预测准确度热图 - 只显示掩码区域的准确度
        pred_accuracy = np.zeros_like(binary_mask, dtype=float)
        
        # 计算掩码区域的相对误差 (归一化到0-1之间，0表示无误差，1表示最大误差)
        max_error_value = np.max(error[binary_mask > 0]) if np.any(binary_mask > 0) else 1.0
        
        # 对掩码区域，计算并填充归一化的误差值
        if np.any(binary_mask > 0):
            # 将误差转换为准确度分数: 100表示完全匹配，0表示最大误差
            for i, j in zip(*mask_indices):
                norm_error = error[i, j] / max_error_value
                pred_accuracy[i, j] = 100 * (1.0 - norm_error)
        
        # 创建掩码版本的准确度热图，使非掩码区域变为NaN值(不显示)
        masked_pred_accuracy = pred_accuracy.copy()
        for i in range(freq_dim):
            for j in range(time_dim):
                if binary_mask[i, j] == 0:  # 非掩码区域
                    masked_pred_accuracy[i, j] = np.nan  # 设为NaN，使其透明
                        
        # 设置底图背景为白色
        ax6.set_facecolor('white')
        
        # 创建红到绿的颜色映射，红色表示低准确度(高误差)，绿色表示高准确度(低误差)
        green_to_red = matplotlib.colors.LinearSegmentedColormap.from_list(
            'GreenToRed', [(0, 1, 0), (1, 1, 0), (1, 0, 0)])
            
        # 绘制相对误差热图，使用掩码数据
        im = plt.imshow(masked_pred_accuracy, aspect='auto', origin='lower', cmap=green_to_red, vmin=0, vmax=100)
        plt.colorbar(im, label='Accuracy Score (0-100)')
        
        # 标记掩码区域的边界，使用白色虚线
        for i in range(freq_patches):
            for j in range(time_patches):
                # 如果这个patch是掩码区域
                if patch_mask_status[i, j] > 0:
                    f_start = i * fshape
                    f_end = min(f_start + fshape, freq_dim)
                    t_start = j * tshape
                    t_end = min(t_start + tshape, time_dim)
                    
                    rect = patches.Rectangle(
                        (t_start, f_start), 
                        t_end - t_start, 
                        f_end - f_start, 
                        linewidth=1, 
                        edgecolor="white",
                        facecolor='none',
                        linestyle='--'
                    )
                    ax6.add_patch(rect)
        
        plt.title("Prediction Accuracy Heatmap", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        
        # 添加掩码信息到图表底部而不是底部外侧
        mask_info = f"Patch: {fshape}×{tshape} | Mask: {mask_percentage:.2f}%"
        plt.figtext(0.75, 0.01, mask_info, fontsize=8, ha='center')
        
        
        # 整体标题 - 放在顶部中央，并有足够边距
        plt.suptitle(f"Train Phase Mask Visualization (Step: {step}, Task: {task_name})", 
                    fontsize=13, y=0.98)
        
        # 调整子图布局和间距
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)  # 增加顶部空间和子图间距
        
        # 保存图像时使用深色背景，与visualize_grid_path保持一致
        with sns.axes_style("dark"):
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mask real_pred_result_np phase visualization saved: {os.path.join(self.save_dir, filename)}")
        
        return {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'percentile_90': percentile_90,
            'mask_count': mask_count,
            'mask_percentage': mask_percentage
        }    
    
    def visualize_generative_mask_phase(self, original_features, mask, real_pred_result_np, pred_result_np,
                       step, task_name, filename="CoRe_visualization.png", fshape=16, tshape=16):
        """
        可视化CoRe模型的重建结果

        Args:
            original_features: 原始音频特征
            mask: 掩码矩阵，1表示掩码区域（需要预测），0表示非掩码区域（已知内容）
            real_pred_result_np: 模型重建的完整特征
            pred_result_np: 预测准确度分数
            step: 当前训练步数
            task_name: 任务名称
            filename: 保存的文件名
            fshape: patch的频率维度大小
            tshape: patch的时间维度大小

        Returns:
            包含评估指标的字典
        """
        # 设置Seaborn样式
        sns.set_style("dark")
        sns.set_context("notebook", font_scale=1.2)

        # 确保所有输入都是numpy数组并降维
        if torch.is_tensor(original_features):
            original_features = original_features.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if torch.is_tensor(real_pred_result_np):
            real_pred_result_np = real_pred_result_np.cpu().numpy()
        if torch.is_tensor(pred_result_np):
            pred_result_np = pred_result_np.cpu().numpy()

        # 确保输入维度正确
        original_features = np.squeeze(original_features)
        mask = np.squeeze(mask)
        real_pred_result_np = np.squeeze(real_pred_result_np)
        pred_result_np = np.squeeze(pred_result_np)

        # 确保维度一致性
        if original_features.shape != real_pred_result_np.shape:
            print(f"Warning: Shape mismatch - original: {original_features.shape}, prediction: {real_pred_result_np.shape}")
            if original_features.shape[::-1] == real_pred_result_np.shape:
                real_pred_result_np = real_pred_result_np.transpose()
                mask = mask.transpose()
                pred_result_np = pred_result_np.transpose()

        # 获取频谱图尺寸
        freq_dim, time_dim = original_features.shape

        # 计算patch数量
        freq_patches = int(np.ceil(freq_dim / fshape))
        time_patches = int(np.ceil(time_dim / tshape))

        # 注意：在新的实现中，掩码区域（需要预测的区域）是 mask=1，已知区域（输入给编码器的区域）是 mask=0
        masked_area = mask.copy()  # 1表示掩码区域（需要预测）
        known_area = 1 - masked_area  # 1表示已知区域（编码器输入）
        
        # 获取掩码区域的索引位置
        masked_indices = np.where(masked_area > 0)

        # 计算每个patch的状态（掩码/非掩码）
        patch_masked_status = np.zeros((freq_patches, time_patches))
        for i in range(freq_patches):
            for j in range(time_patches):
                f_start = i * fshape
                f_end = min(f_start + fshape, freq_dim)
                t_start = j * tshape
                t_end = min(t_start + tshape, time_dim)
                
                # 如果patch内50%以上像素是掩码的，则将整个patch标记为掩码
                patch_area = masked_area[f_start:f_end, t_start:t_end]
                if np.mean(patch_area) > 0.5:
                    patch_masked_status[i, j] = 1

        # 计算掩码patch和已知patch的数量
        masked_patch_count = int(np.sum(patch_masked_status))
        total_patch_count = freq_patches * time_patches
        known_patch_count = total_patch_count - masked_patch_count

        # 进行统计，确保数据正确
        total_cells = freq_dim * time_dim
        masked_count = int(np.sum(masked_area))
        known_count = total_cells - masked_count

        if total_cells > 0:
            masked_percentage = (masked_count / total_cells) * 100
            known_percentage = 100 - masked_percentage
        else:
            masked_percentage = 0
            known_percentage = 0

        # 打印统计信息用于调试
        print(f"Mask statistics: Masked areas: {masked_percentage:.2f}% ({masked_count} cells), Known areas: {known_percentage:.2f}% ({known_count} cells)")

        # 计算预测误差 - 只关注掩码区域(需要预测的部分)
        error = np.abs(original_features - real_pred_result_np)
        masked_error = error * masked_area  # 掩码区域的误差

        # 计算掩码区域的原始特征和重建特征，用于分布对比
        masked_original_values = original_features[masked_indices]
        masked_reconstructed_values = real_pred_result_np[masked_indices]

        # 计算原始特征和重建特征的范围，用于保持一致的颜色映射
        vmin = min(np.min(original_features), np.min(real_pred_result_np))
        vmax = max(np.max(original_features), np.max(real_pred_result_np))

        # 创建图形 - 使用3x3布局以包含特征分布图
        plt.figure(figsize=(18, 12))

        # 辅助函数：在子图上添加patch网格线
        def add_patch_grid(ax):
            # 绘制垂直线（时间轴）
            for t in range(0, time_dim, tshape):
                ax.axvline(x=t, color="white", linestyle='--', alpha=0.5, linewidth=1.5)
            # 绘制水平线（频率轴）
            for f in range(0, freq_dim, fshape):
                ax.axhline(y=f, color="white", linestyle='--', alpha=0.5, linewidth=1.5)

        # 1. Top-left: 原始音频特征
        ax1 = plt.subplot(3, 3, 1)
        plt.imshow(original_features, aspect='auto', origin='lower', 
                  cmap=sns.color_palette("viridis", as_cmap=True), 
                  vmin=vmin, vmax=vmax)  # 使用相同的颜色范围
        plt.title("Original Audio Features", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        add_patch_grid(ax1)

        # 2. Top-middle: 编码器输入(非掩码区域，即已知区域)
        ax2 = plt.subplot(3, 3, 2)
        # 创建一个与原始特征相同形状的数组，初始化为原始特征的最小值，作为背景色（通常为黑色）
        visible_features = np.ones_like(original_features) * np.min(original_features)  # 黑色背景
        # 找出所有已知区域的索引位置（即known_area值大于0的位置）
        known_indices = np.where(known_area > 0)
        # 只在已知区域显示原始特征值，掩码区域保持为黑色背景
        visible_features[known_indices] = original_features[known_indices]
        # 使用viridis颜色映射绘制图像，aspect='auto'自动调整宽高比，origin='lower'使频率轴从下到上增加
        plt.imshow(visible_features, aspect='auto', origin='lower', 
                  cmap=sns.color_palette("viridis", as_cmap=True),
                  vmin=vmin, vmax=vmax)  # 使用相同的颜色范围
        # 添加patch网格线，显示特征块的边界
        add_patch_grid(ax2)
        
        plt.title(f"Encoder Input (Known Patches: {known_patch_count})", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)

        # 3. Top-right: 特征分布对比图 (新增)
        ax3 = plt.subplot(3, 3, 3)
        # 使用KDE图显示掩码区域原始特征和重建特征的分布
        sns.kdeplot(masked_original_values, ax=ax3, color="blue", label="Original Features", shade=True)
        sns.kdeplot(masked_reconstructed_values, ax=ax3, color="red", label="Reconstructed Features", shade=True)
        plt.title("Feature Distribution Comparison (Masked Region)", fontsize=11, pad=10)
        plt.xlabel("Feature Value", fontsize=9, labelpad=5)
        plt.ylabel("Density", fontsize=9, labelpad=5)
        plt.legend(fontsize=8)
        
        # 4. Middle-left: 模型重建的完整频谱图
        ax4 = plt.subplot(3, 3, 4)
        plt.imshow(real_pred_result_np, aspect='auto', origin='lower', 
                  cmap=sns.color_palette("viridis", as_cmap=True),
                  vmin=vmin, vmax=vmax)  # 使用相同的颜色范围
        plt.title("Model Reconstructed Spectrogram", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        add_patch_grid(ax4)

        # 5. Middle-middle: 掩码区域的预测误差
        ax5 = plt.subplot(3, 3, 5)
        plt.imshow(masked_error, aspect='auto', origin='lower', cmap=sns.color_palette("plasma", as_cmap=True))
        plt.title("Prediction Error Distribution (Masked Regions)", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        add_patch_grid(ax5)
        plt.colorbar(label='Error Magnitude')

        # 6. Middle-right: 原始和重建特征的散点图对比 (新增)
        ax6 = plt.subplot(3, 3, 6)
        plt.scatter(masked_original_values, masked_reconstructed_values, alpha=0.5, s=1, c='blue')
        # 添加对角线参考线 (y=x)
        min_val = min(np.min(masked_original_values), np.min(masked_reconstructed_values))
        max_val = max(np.max(masked_original_values), np.max(masked_reconstructed_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        plt.title("Feature Reconstruction Scatter Plot", fontsize=11, pad=10)
        plt.xlabel("Original Values", fontsize=9, labelpad=5)
        plt.ylabel("Reconstructed Values", fontsize=9, labelpad=5)
        
        # 添加统计信息到图表中
        stats_text = f"MSE: {np.mean(np.square(masked_original_values - masked_reconstructed_values)):.6f}\n"
        stats_text += f"Corr: {np.corrcoef(masked_original_values, masked_reconstructed_values)[0,1]:.4f}"
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # 7. Bottom-left: 掩码区域原始特征与预测特征的对比
        ax7 = plt.subplot(3, 3, 7)
        # 创建一个仅显示掩码区域原始特征的图
        masked_original = np.ones_like(original_features) * np.min(original_features)  # 背景色
        # 只在掩码区域显示原始特征
        masked_original[masked_indices] = original_features[masked_indices]
        
        plt.imshow(masked_original, aspect='auto', origin='lower', 
                  cmap=sns.color_palette("viridis", as_cmap=True),
                  vmin=vmin, vmax=vmax)  # 使用相同的颜色范围
        plt.title("Masked Region Original Features", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        add_patch_grid(ax7)
        
        # 标记掩码区域的patch边界 - 只使用轮廓不填充
        for i in range(freq_patches):
            for j in range(time_patches):
                if patch_masked_status[i, j] == 1:  # 掩码的patch
                    f_start = i * fshape
                    f_end = min(f_start + fshape, freq_dim)
                    t_start = j * tshape
                    t_end = min(t_start + tshape, time_dim)
                    rect = patches.Rectangle(
                        (t_start, f_start),
                        t_end - t_start,
                        f_end - f_start,
                        linewidth=2,
                        edgecolor="red",
                        facecolor='none',  # 不使用填充色
                        alpha=0.8
                    )
                    ax7.add_patch(rect)

        # 8. Bottom-middle: 掩码区域预测特征图
        ax8 = plt.subplot(3, 3, 8)
        # 创建一个仅显示掩码区域预测特征的图
        masked_prediction = np.ones_like(real_pred_result_np) * np.min(real_pred_result_np)  # 背景色
        # 只在掩码区域显示预测特征
        masked_prediction[masked_indices] = real_pred_result_np[masked_indices]
        
        plt.imshow(masked_prediction, aspect='auto', origin='lower', 
                  cmap=sns.color_palette("viridis", as_cmap=True),
                  vmin=vmin, vmax=vmax)  # 使用相同的颜色范围
        plt.title("Masked Region Reconstructed Features", fontsize=11, pad=10)
        plt.xlabel("Time Frames", fontsize=9, labelpad=5)
        plt.ylabel("Frequency", fontsize=9, labelpad=5)
        add_patch_grid(ax8)
        
        # 标记掩码区域的patch边界 - 只使用轮廓不填充
        for i in range(freq_patches):
            for j in range(time_patches):
                if patch_masked_status[i, j] == 1:  # 掩码的patch
                    f_start = i * fshape
                    f_end = min(f_start + fshape, freq_dim)
                    t_start = j * tshape
                    t_end = min(t_start + tshape, time_dim)
                    rect = patches.Rectangle(
                        (t_start, f_start),
                        t_end - t_start,
                        f_end - f_start,
                        linewidth=2,
                        edgecolor="red",
                        facecolor='none',  # 不使用填充色
                        alpha=0.8
                    )
                    ax8.add_patch(rect)

        # 9. Bottom-right: 原始特征和重建特征的直方图对比 (新增)
        ax9 = plt.subplot(3, 3, 9)
        plt.hist(masked_original_values, bins=50, alpha=0.5, label='Original', color='blue')
        plt.hist(masked_reconstructed_values, bins=50, alpha=0.5, label='Reconstructed', color='red')
        plt.title("Feature Value Histograms", fontsize=11, pad=10)
        plt.xlabel("Feature Value", fontsize=9, labelpad=5)
        plt.ylabel("Count", fontsize=9, labelpad=5)
        plt.legend(fontsize=8)
        
        # 计算并显示统计信息
        orig_mean = np.mean(masked_original_values)
        recon_mean = np.mean(masked_reconstructed_values)
        orig_std = np.std(masked_original_values)
        recon_std = np.std(masked_reconstructed_values)
        
        stats_text = f"Orig μ: {orig_mean:.4f}, σ: {orig_std:.4f}\n"
        stats_text += f"Recon μ: {recon_mean:.4f}, σ: {recon_std:.4f}"
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # 添加掩码统计信息到图表底部
        mask_info = f"Patch: {fshape}×{tshape} | Masked: {masked_percentage:.2f}% | Known: {known_percentage:.2f}%"
        plt.figtext(0.75, 0.01, mask_info, fontsize=8, ha='center')

        # 整体标题
        plt.suptitle(f"Visualization Results (Step: {step}, Task: {task_name})",
                     fontsize=13, y=0.98)

        # 调整子图布局和间距
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)

        # 保存图像
        with sns.axes_style("dark"):
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"visualization saved: {save_path}")

        # 计算掩码区域的评估指标
        eval_metrics = {
            'mask_count': masked_count,  # 与traintest_mask.py兼容
            'mask_percentage': masked_percentage,  # 与traintest_mask.py兼容
            'masked_count': masked_count,
            'masked_percentage': masked_percentage,
            'known_count': known_count,
            'known_percentage': known_percentage
        }

        if masked_count > 0:
            # 均方误差(MSE)：计算掩码区域原始特征与预测特征之间差值的平方的平均值，值越小表示预测越准确
            eval_metrics['mse'] = np.mean(np.square(original_features[masked_indices] - real_pred_result_np[masked_indices]))
            
            # 平均绝对误差(MAE)：计算掩码区域原始特征与预测特征之间绝对差值的平均值，对异常值不敏感
            eval_metrics['mae'] = np.mean(np.abs(original_features[masked_indices] - real_pred_result_np[masked_indices]))
            
            # 最大误差：掩码区域中预测误差的最大值，反映最差情况下的预测偏差
            eval_metrics['max_error'] = np.max(np.abs(original_features[masked_indices] - real_pred_result_np[masked_indices]))
            
            # 90百分位误差：表示90%的预测误差都小于此值，比最大误差更稳健，不受极端异常值影响
            eval_metrics['percentile_90'] = np.percentile(np.abs(original_features[masked_indices] - real_pred_result_np[masked_indices]), 90)
            
            # 添加新指标 - 特征分布指标
            eval_metrics['orig_mean'] = orig_mean
            eval_metrics['recon_mean'] = recon_mean
            eval_metrics['orig_std'] = orig_std
            eval_metrics['recon_std'] = recon_std
            eval_metrics['correlation'] = np.corrcoef(masked_original_values, masked_reconstructed_values)[0,1]
        else:
            eval_metrics['mse'] = 0
            eval_metrics['mae'] = 0
            eval_metrics['max_error'] = 0
            eval_metrics['percentile_90'] = 0
            eval_metrics['orig_mean'] = 0
            eval_metrics['recon_mean'] = 0
            eval_metrics['orig_std'] = 0
            eval_metrics['recon_std'] = 0
            eval_metrics['correlation'] = 0

        return eval_metrics


    def add_patch_grid(self, ax, shape, fshape, tshape):
        """Add patch grid to visualization"""
        # ... existing code ...
        height, width = shape
        
        # Draw vertical lines (time axis)
        for t in range(0, width, tshape):
            ax.axvline(x=t-0.5, color='white', linewidth=0.5, alpha=0.7)
        
        # Draw horizontal lines (frequency axis)  
        for f in range(0, height, fshape):
            ax.axhline(y=f-0.5, color='white', linewidth=0.5, alpha=0.7)


    def visualize_mae_ast(self, original_features, mask_vis_data,
                      real_pred_result_np, pred_result_np,
                      step, task_name, filename="mae_ast_visualization.png", fshape=16, tshape=16):
        """
        Visualize MAE-AST pretraining results with comprehensive analysis
        参考CoRe和PoST的样式，突出显示patch块结构
        """
        print(f"[DEBUG] MAE-AST可视化开始")
        print(f"[DEBUG] 输入数据形状: original_features={original_features.shape}, mask_vis_data={mask_vis_data.shape}")
        print(f"[DEBUG] real_pred_result_np={real_pred_result_np.shape}, pred_result_np={pred_result_np.shape}")
        
        # Convert to numpy if needed
        if hasattr(original_features, 'cpu'):
            original_features = original_features.cpu().numpy()
        if hasattr(mask_vis_data, 'cpu'):
            mask_vis_data = mask_vis_data.cpu().numpy()
        if hasattr(real_pred_result_np, 'cpu'):
            real_pred_result_np = real_pred_result_np.cpu().numpy()
        if hasattr(pred_result_np, 'cpu'):
            pred_result_np = pred_result_np.cpu().numpy()
        
        # Remove batch dimension if present
        if original_features.ndim == 3:
            original_features = original_features[0]
        if mask_vis_data.ndim == 3:
            mask_vis_data = mask_vis_data[0]
        if real_pred_result_np.ndim == 3:
            real_pred_result_np = real_pred_result_np[0]
        if pred_result_np.ndim == 3:
            pred_result_np = pred_result_np[0]
        
        print(f"[DEBUG] 处理后数据形状: original_features={original_features.shape}")
        print(f"[DEBUG] mask_vis_data范围: {mask_vis_data.min():.2f} to {mask_vis_data.max():.2f}")
        print(f"[DEBUG] mask_vis_data中99.0的数量: {np.sum(mask_vis_data > 90)}")
        
        # 关键修复：如果mask_vis_data中没有99标记，说明数据传递有问题
        if np.sum(mask_vis_data > 90) == 0:
            print(f"[WARNING] mask_vis_data中没有发现99标记，可能数据传递有问题")
            print(f"[WARNING] mask_vis_data的唯一值: {np.unique(mask_vis_data)[:10]}...")
            
            # 尝试从数据中推断掩码区域（作为备选方案）
            diff = np.abs(real_pred_result_np - original_features)
            threshold = np.percentile(diff, 75)
            binary_mask = (diff > threshold).astype(np.float32)
            print(f"[WARNING] 使用差异阈值推断掩码区域，阈值: {threshold:.3f}")
        else:
            # 正常情况：使用99标记识别掩码区域
            binary_mask = (mask_vis_data > 90).astype(np.float32)
        
        print(f"[DEBUG] binary_mask sum: {binary_mask.sum()}, 总像素: {binary_mask.size}")
        print(f"[DEBUG] 掩码覆盖率: {binary_mask.sum() / binary_mask.size * 100:.2f}%")
        
        # Calculate mask statistics
        total_cells = binary_mask.size
        masked_cells = int(binary_mask.sum())
        mask_percentage = (masked_cells / total_cells) * 100
        
        print(f"[DEBUG] 掩码统计: 总像素={total_cells}, 掩码像素={masked_cells}, 百分比={mask_percentage:.2f}%")
        
        # 参考CoRe和PoST的布局：2x3网格布局，突出显示patch结构
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'MAE-AST Pretraining Visualization (Step: {step})\n'
                    f'Masked: {mask_percentage:.2f}% | Patch: {fshape}x{tshape}', 
                    fontsize=16, fontweight='bold')
        
        # 定义patch网格绘制函数，参考CoRe和PoST的样式
        def add_patch_grid(ax, shape, fshape, tshape):
            """Add patch grid to visualization with enhanced visibility"""
            height, width = shape
            
            # Draw vertical lines (time axis) - 更明显的网格线
            for t in range(0, width, tshape):
                ax.axvline(x=t-0.5, color='white', linewidth=1.0, alpha=0.8)
            
            # Draw horizontal lines (frequency axis)  
            for f in range(0, height, fshape):
                ax.axhline(y=f-0.5, color='white', linewidth=1.0, alpha=0.8)
            
            # 添加patch边界高亮
            for f in range(0, height, fshape):
                for t in range(0, width, tshape):
                    rect = plt.Rectangle((t-0.5, f-0.5), tshape, fshape, 
                                       fill=False, edgecolor='white', linewidth=0.5, alpha=0.6)
                    ax.add_patch(rect)
        
        # 1. Original Spectrogram (左上)
        ax1 = axes[0, 0]
        im1 = ax1.imshow(original_features, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('Original Spectrogram', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Frames')
        ax1.set_ylabel('Frequency')
        add_patch_grid(ax1, original_features.shape, fshape, tshape)
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 2. Masked Regions (中上) - 参考CoRe样式，突出显示掩码patch
        ax2 = axes[0, 1]
        if masked_cells > 0:
            # 创建掩码可视化：使用红色高亮掩码patch
            masked_display = original_features.copy()
            
            # 计算patch级别的掩码
            height, width = original_features.shape
            patch_mask = np.zeros((height // fshape, width // tshape))
            
            for f in range(0, height, fshape):
                for t in range(0, width, tshape):
                    f_end = min(f + fshape, height)
                    t_end = min(t + tshape, width)
                    patch_region = binary_mask[f:f_end, t:t_end]
                    if patch_region.sum() > (fshape * tshape * 0.5):  # 如果patch中超过50%被掩码
                        patch_f_idx = f // fshape
                        patch_t_idx = t // tshape
                        if patch_f_idx < patch_mask.shape[0] and patch_t_idx < patch_mask.shape[1]:
                            patch_mask[patch_f_idx, patch_t_idx] = 1
                            # 将整个patch标记为掩码区域
                            masked_display[f:f_end, t:t_end] = np.nan
            
            im2 = ax2.imshow(masked_display, aspect='auto', origin='lower', cmap='viridis')
            
            # 在掩码patch上叠加红色矩形，参考CoRe的样式
            for f_idx in range(patch_mask.shape[0]):
                for t_idx in range(patch_mask.shape[1]):
                    if patch_mask[f_idx, t_idx] == 1:
                        rect = plt.Rectangle((t_idx * tshape - 0.5, f_idx * fshape - 0.5), 
                                           tshape, fshape, 
                                           fill=True, facecolor='red', alpha=0.6, 
                                           edgecolor='darkred', linewidth=2)
                        ax2.add_patch(rect)
            
            masked_patch_count = int(patch_mask.sum())
        else:
            im2 = ax2.imshow(original_features, aspect='auto', origin='lower', cmap='viridis')
            masked_patch_count = 0
        
        ax2.set_title(f'Masked Regions (Masked patches: {masked_patch_count})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Frames')
        ax2.set_ylabel('Frequency')
        add_patch_grid(ax2, original_features.shape, fshape, tshape)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # 3. Reconstructed Spectrogram (右上)
        ax3 = axes[0, 2]
        im3 = ax3.imshow(real_pred_result_np, aspect='auto', origin='lower', cmap='viridis')
        ax3.set_title('MAE-AST Reconstructed Spectrogram', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Frames')
        ax3.set_ylabel('Frequency')
        add_patch_grid(ax3, real_pred_result_np.shape, fshape, tshape)
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # 4. Reconstruction Error (左下) - 参考PoST样式
        ax4 = axes[1, 0]
        if masked_cells > 0:
            error_map = np.abs(real_pred_result_np - original_features)
            # 只显示掩码区域的误差
            error_display = np.full_like(error_map, np.nan)
            error_display[binary_mask > 0] = error_map[binary_mask > 0]
            
            im4 = ax4.imshow(error_display, aspect='auto', origin='lower', cmap='hot')
            ax4.set_title('Reconstruction Error (Masked Regions Only)', fontsize=12, fontweight='bold')
            plt.colorbar(im4, ax=ax4, shrink=0.8)
        else:
            ax4.text(0.5, 0.5, 'No masked region data available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Reconstruction Error', fontsize=12, fontweight='bold')
        
        ax4.set_xlabel('Time Frames')
        ax4.set_ylabel('Frequency')
        add_patch_grid(ax4, original_features.shape, fshape, tshape)
        
        # 5. Patch-wise Reconstruction Quality (中下) - 新增patch级别分析
        ax5 = axes[1, 1]
        if masked_cells > 0:
            height, width = original_features.shape
            patch_quality = np.zeros((height // fshape, width // tshape))
            
            for f_idx in range(patch_quality.shape[0]):
                for t_idx in range(patch_quality.shape[1]):
                    f_start = f_idx * fshape
                    f_end = min(f_start + fshape, height)
                    t_start = t_idx * tshape
                    t_end = min(t_start + tshape, width)
                    
                    patch_mask_region = binary_mask[f_start:f_end, t_start:t_end]
                    if patch_mask_region.sum() > 0:  # 如果patch包含掩码区域
                        patch_orig = original_features[f_start:f_end, t_start:t_end]
                        patch_recon = real_pred_result_np[f_start:f_end, t_start:t_end]
                        
                        # 计算patch的重建质量（使用相关系数）
                        if patch_orig.std() > 0 and patch_recon.std() > 0:
                            correlation = np.corrcoef(patch_orig.flatten(), patch_recon.flatten())[0, 1]
                            patch_quality[f_idx, t_idx] = max(0, correlation)  # 确保非负
                        else:
                            patch_quality[f_idx, t_idx] = 0
                    else:
                        patch_quality[f_idx, t_idx] = np.nan  # 非掩码区域
            
            im5 = ax5.imshow(patch_quality, aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
            ax5.set_title('Patch-wise Reconstruction Quality', fontsize=12, fontweight='bold')
            plt.colorbar(im5, ax=ax5, shrink=0.8, label='Correlation')
            
            # 添加patch网格
            for f_idx in range(patch_quality.shape[0] + 1):
                ax5.axhline(y=f_idx-0.5, color='black', linewidth=0.5, alpha=0.7)
            for t_idx in range(patch_quality.shape[1] + 1):
                ax5.axvline(x=t_idx-0.5, color='black', linewidth=0.5, alpha=0.7)
        else:
            ax5.text(0.5, 0.5, 'No masked region data available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Patch-wise Reconstruction Quality', fontsize=12, fontweight='bold')
        
        ax5.set_xlabel('Time Patches')
        ax5.set_ylabel('Frequency Patches')
        
        # 6. Feature Distribution Comparison (右下)
        ax6 = axes[1, 2]
        if masked_cells > 0:
            original_masked = original_features[binary_mask > 0]
            pred_masked = real_pred_result_np[binary_mask > 0]
            
            # 绘制分布对比
            ax6.hist(original_masked.flatten(), bins=50, alpha=0.7, label='Original (Masked)', 
                    color='blue', density=True, histtype='step', linewidth=2)
            ax6.hist(pred_masked.flatten(), bins=50, alpha=0.7, label='Reconstructed (Masked)', 
                    color='red', density=True, histtype='step', linewidth=2)
            
            # 添加统计信息
            mse = np.mean((original_masked - pred_masked) ** 2)
            mae = np.mean(np.abs(original_masked - pred_masked))
            correlation = np.corrcoef(original_masked, pred_masked)[0, 1]
            
            ax6.text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nCorr: {correlation:.3f}', 
                    transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax6.legend()
            ax6.set_xlabel('Feature Value')
            ax6.set_ylabel('Density')
        else:
            ax6.text(0.5, 0.5, 'No masked region data available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax6.transAxes, fontsize=12)
        
        ax6.set_title('Feature Distribution Comparison', fontsize=12, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # Save the figure
        full_path = os.path.join(self.save_dir, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[DEBUG] MAE-AST可视化完成，保存到: {full_path}")
        print(f"[DEBUG] 最终统计: 掩码区域={mask_percentage:.2f}%, 掩码像素数={masked_cells}")
        
        # 返回统计信息
        stats = {
            'mask_count': masked_cells,
            'mask_percentage': mask_percentage,
            'mse': 0.0,
            'mae': 0.0,
            'max_error': 0.0,
            'percentile_90': 0.0
        }
        
        if masked_cells > 0:
            original_masked = original_features[binary_mask > 0]
            pred_masked = real_pred_result_np[binary_mask > 0]
            
            stats['mse'] = np.mean((original_masked - pred_masked) ** 2)
            stats['mae'] = np.mean(np.abs(original_masked - pred_masked))
            stats['max_error'] = np.max(np.abs(original_masked - pred_masked))
            stats['percentile_90'] = np.percentile(np.abs(original_masked - pred_masked), 90)
        
        return stats


    def visualize_behavior_comparison(self, wav_paths, labels=None, sr=16000, n_mels=128,
                                      n_fft=1024, hop_length=160, win_length=400,
                                      title="Behavior Comparison (10 Classes)",
                                      filename="behavior_comparison.png"):
        """
        读取多段音频并绘制2x5梅尔频谱对比图（统一颜色尺度）

        Args:
            wav_paths: 音频文件绝对路径列表（期望长度为10）
            labels: 可选，自定义标题标签列表；若为None，则使用上级目录名
            sr: 采样率
            n_mels, n_fft, hop_length, win_length: 梅尔谱参数
            title: 总标题
            filename: 输出图片文件名
        """
        # 读取与转换
        specs_db = []
        auto_labels = []
        for p in wav_paths:
            try:
                y, _sr = librosa.load(p, sr=sr, mono=True)
                S = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                    win_length=win_length, n_mels=n_mels, fmin=20, fmax=sr/2
                )
                S_db = librosa.power_to_db(S, ref=np.max)
                specs_db.append(S_db)
                auto_labels.append(os.path.basename(os.path.dirname(p)))
            except Exception as e:
                print(f"[WARN] 读取失败: {p} -> {e}")
                # 放入一个空谱，保证布局不破坏
                specs_db.append(np.full((n_mels, 10), -80.0, dtype=np.float32))
                auto_labels.append(os.path.basename(os.path.dirname(p)))

        # 统一颜色范围（鲁棒分位数）
        all_vals = np.concatenate([s.flatten() for s in specs_db]) if specs_db else np.array([-80.0])
        vmin = float(np.percentile(all_vals, 2))
        vmax = float(np.percentile(all_vals, 98))
        if vmin >= vmax:
            vmin, vmax = np.min(all_vals), np.max(all_vals)

        # 绘图
        rows, cols = 2, 5
        fig, axes = plt.subplots(rows, cols, figsize=(22, 8))
        axes = axes.ravel().tolist()

        panel_tags = list("ABCDEFGHIJ")

        for idx, ax in enumerate(axes):
            if idx < len(specs_db):
                im = librosa.display.specshow(
                    specs_db[idx], sr=sr, hop_length=hop_length,
                    x_axis='time', y_axis='mel', cmap=self.cmap,
                    vmin=vmin, vmax=vmax, ax=ax
                )
                cap = labels[idx] if labels and idx < len(labels) else auto_labels[idx]
                # 面板标签 + 类别标题
                ax.set_title(f"({panel_tags[idx]}) {cap}", fontsize=11, pad=6)

                # 只在底行显示时间刻度
                if idx < 5:
                    ax.set_xlabel("Time (s)")
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.set_xlabel("Time (s)")
                    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
                    ax.tick_params(axis='x', labelsize=9)

                # 只在左列显示频率刻度
                if idx % 5 == 0:
                    # ax.set_ylabel("Mel Frequency (Hz)")
                    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
                    ax.tick_params(axis='y', labelsize=9)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelleft=False)

                # 网格提升可读性
                ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
            else:
                ax.axis('off')

        # fig.suptitle(title, fontsize=16)
        # 共享色条
        # cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
        # cbar.set_label('dB')

        # 全局坐标轴标签（再次强化）
        # try:
        #     fig.supxlabel("Time (s)", fontsize=12)
        #     fig.supylabel("Mel Frequency (Hz)", fontsize=12)
        # except Exception:
        #     pass

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(self.save_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Behavior comparison saved to: {out_path}")

if __name__ == "__main__":
    # 简单的使用示例
    visualizer = SSASTVisualizer(save_dir='./test_visualizations')

    # 创建一个测试频谱图
    x = np.random.randn(128, 100) * 10

    # 创建一个测试掩码
    mask = np.zeros((128, 100))
    mask[30:60, 40:70] = 1  # 中间区域掩码

    # 创建一个测试预测结果
    pred = x.copy()
    pred[mask == 1] = pred[mask == 1] + np.random.randn(np.sum(mask)) * 5

    # 可视化频谱图
    visualizer.visualize_spectrogram(x, filename="test_spectrogram.png")

    # 可视化掩码区域
    visualizer.visualize_mask(x, mask, filename="test_mask.png")

    # 可视化预测结果
    masked_x = x.copy()
    masked_x[mask == 1] = 0
    visualizer.visualize_real_pred_result_np(x, masked_x, pred, mask, filename="test_real_pred_result_np.png")

    # 可视化预测误差
    visualizer.visualize_pred_result_nperror(x, pred, mask, filename="test_error.png")

    # 可视化网格路径
    visualizer.visualize_grid_path(x, (16, 16), (16, 16), mask, filename="test_grid.png")

    print("测试可视化完成！")