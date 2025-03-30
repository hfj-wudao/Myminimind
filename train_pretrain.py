#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# @Time    : 2025/3/30 03:13
# @Author  : huangfujue
# @File    : train_pretrain.py
# @Date    : 2025/3/30
"""
MiniMindLM 预训练模块

本模块实现了 MiniMindLM 语言模型的预训练流程，提供了完整的训练、监控和恢复机制。
主要功能包括：

1. 训练流程管理
   - 支持从头开始训练和断点续训
   - 自动检测和配置GPU/CPU训练环境
   - 实现动态学习率调整（余弦衰减调度）
   - 支持梯度裁剪和权重衰减等优化技术

2. 训练状态监控与可视化
   - 集成 TensorBoard 可视化支持，记录关键训练指标
   - 跟踪并记录损失、学习率、训练速度等指标
   - 提供详细的训练日志和进度报告
   - 监控GPU内存使用情况

3. 模型和检查点管理
   - 定期保存训练检查点，防止数据丢失
   - 保存每个轮次的模型状态
   - 单独保存最佳模型和最终模型
   - 自动管理训练输出目录结构

4. 训练恢复机制
   - 支持从任意检查点恢复训练
   - 保存并恢复完整训练状态，包括：
     * 模型参数
     * 优化器状态
     * 训练进度（步数和轮次）
     * 训练配置
     * 最佳损失记录

5. 性能优化
   - 支持批处理和多线程数据加载
   - 使用 pin_memory 加速数据传输到GPU
   - 跟踪并报告训练吞吐量（每秒处理词元数）

主要类:
- TrainingStats: 训练状态跟踪器，管理指标记录和TensorBoard可视化

主要函数:
- train(config): 执行训练流程，管理模型训练的完整生命周期
- main(): 主函数，设置训练配置并启动训练过程

使用方法:
1. 配置训练参数 (在main函数的config字典中)
2. 直接运行脚本开始新训练
3. 通过设置resume_from_checkpoint参数从检查点继续训练
4. 使用TensorBoard查看训练进度和指标

依赖:
- PyTorch: 核心深度学习框架
- Transformers: 提供分词器功能
- TensorBoard: 训练可视化
- tqdm: 进度显示
- 项目自定义模块: LMConfig, PretrainDataset, MiniMindLM

作者: huangfujue
日期: 2025/3/30
"""
# ----------------------------------------------------------------------------------------------------------------------
import math
import os
import time
import logging
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import datetime
import json
# 添加 TensorBoard 相关导入
from torch.utils.tensorboard import SummaryWriter

from LMConfig import LMConfig
from dataset import PretrainDataset
from model import MiniMindLM

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingStats:
    """训练状态跟踪器，记录详细的训练指标"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 训练指标记录
        self.stats = {
            'steps': [],
            'loss': [],
            'lr': [],
            'epoch': [],
            'tokens_per_second': [],
            'timestamp': []
        }

        # 按轮次保存统计数据
        self.epoch_stats = {}

        # 初始化日志文件
        self.log_file = os.path.join(log_dir, "training_log.txt")
        with open(self.log_file, 'w') as f:
            f.write("步骤,轮次,损失,学习率,每秒处理词元数,时间戳\n")

        # 添加 TensorBoard 支持
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

        # 为每个轮次创建单独的日志文件
        self.epoch_log_dir = os.path.join(log_dir, "epoch_logs")
        os.makedirs(self.epoch_log_dir, exist_ok=True)

    def update(self, step, epoch, loss, lr, tokens_per_second):
        """更新训练统计信息"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 更新内存中的统计信息
        self.stats['steps'].append(step)
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['lr'].append(lr)
        self.stats['tokens_per_second'].append(tokens_per_second)
        self.stats['timestamp'].append(timestamp)

        # 按轮次组织统计数据
        if epoch not in self.epoch_stats:
            self.epoch_stats[epoch] = {
                'steps': [],
                'loss': [],
                'lr': [],
                'tokens_per_second': [],
                'timestamp': []
            }

        self.epoch_stats[epoch]['steps'].append(step)
        self.epoch_stats[epoch]['loss'].append(loss)
        self.epoch_stats[epoch]['lr'].append(lr)
        self.epoch_stats[epoch]['tokens_per_second'].append(tokens_per_second)
        self.epoch_stats[epoch]['timestamp'].append(timestamp)

        # 写入主日志文件
        with open(self.log_file, 'a') as f:
            f.write(f"{step},{epoch},{loss:.6f},{lr:.8f},{tokens_per_second:.2f},{timestamp}\n")

        # 写入轮次特定的日志文件
        epoch_log_file = os.path.join(self.epoch_log_dir, f"epoch_{epoch}.txt")
        epoch_log_exists = os.path.exists(epoch_log_file)

        with open(epoch_log_file, 'a') as f:
            if not epoch_log_exists:
                f.write("步骤,损失,学习率,每秒处理词元数,时间戳\n")
            f.write(f"{step},{loss:.6f},{lr:.8f},{tokens_per_second:.2f},{timestamp}\n")

        # 记录到 TensorBoard - 按通用和轮次特定分类
        # 通用记录
        self.writer.add_scalar('训练/损失', loss, step)
        self.writer.add_scalar('训练/学习率', lr, step)
        self.writer.add_scalar('性能/每秒词元数', tokens_per_second, step)

        # 轮次特定记录 - 便于比较不同轮次的性能
        self.writer.add_scalar(f'轮次_{epoch}/损失', loss, step)
        self.writer.add_scalar(f'轮次_{epoch}/学习率', lr, step)
        self.writer.add_scalar(f'轮次_{epoch}/每秒词元数', tokens_per_second, step)

    def finish_epoch(self, epoch, avg_loss, epoch_time, tokens_per_second):
        """记录轮次结束统计信息"""
        # 记录轮次摘要到TensorBoard
        self.writer.add_scalar('轮次摘要/平均损失', avg_loss, epoch)
        self.writer.add_scalar('轮次摘要/耗时(分钟)', epoch_time / 60, epoch)
        self.writer.add_scalar('轮次摘要/每秒词元数', tokens_per_second, epoch)

        # 保存轮次统计摘要
        epoch_summary = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'duration_seconds': epoch_time,
            'tokens_per_second': tokens_per_second,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(self.epoch_log_dir, f"epoch_{epoch}_summary.json"), 'w') as f:
            json.dump(epoch_summary, f, indent=2)

    def save_stats(self):
        """保存完整的统计信息到文件"""
        # 保存总体统计
        with open(os.path.join(self.log_dir, "training_stats.json"), 'w') as f:
            json.dump(self.stats, f, indent=2)

        # 保存按轮次的统计
        with open(os.path.join(self.log_dir, "epoch_stats.json"), 'w') as f:
            json.dump(self.epoch_stats, f, indent=2)

        # 关闭 TensorBoard writer
        self.writer.close()


def train(config):
    """训练主函数"""
    # 检查是否是断点续训
    resume_training = config.get('resume_from_checkpoint', None)

    # 如果是断点续训，使用原来的输出目录
    if resume_training:
        # 从检查点路径推断输出目录结构
        output_dir = os.path.dirname(os.path.dirname(resume_training))
        model_dir = os.path.join(output_dir, "checkpoints")
        log_dir = os.path.join(output_dir, "logs")
        logger.info(f"断点续训: 从检查点 {resume_training} 继续训练")
        logger.info(f"使用原输出目录: {output_dir}")
    else:
        # 创建新的输出目录结构
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config['output_dir'], f"run_{timestamp}")
        model_dir = os.path.join(output_dir, "checkpoints")
        log_dir = os.path.join(output_dir, "logs")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 保存训练配置
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    # 初始化训练状态跟踪器
    stats = TrainingStats(log_dir)

    # 初始化模型配置
    lm_config = LMConfig(
        dim=config['dim'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len']
    )

    # 设置设备 - 增强GPU检测和日志
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # 设置当前设备为第一个GPU
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
        logger.info("未检测到GPU，使用CPU进行训练")

    # 初始化模型和tokenizer
    logger.info(f"初始化tokenizer: {config['tokenizer_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    logger.info(f"初始化MiniMind模型 (dim={config['dim']}, layers={config['n_layers']})")
    model = MiniMindLM(lm_config).to(device)  # 确保模型到GPU

    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)
    )

    # 初始化训练状态变量
    global_step = 0
    start_epoch = 0
    best_loss = float('inf')

    # 如果是断点续训，加载之前的模型和优化器状态
    if resume_training:
        logger.info(f"加载检查点: {resume_training}")
        checkpoint = torch.load(resume_training, map_location=device)

        # 读取保存的训练状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['step']
        start_epoch = checkpoint['epoch']

        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']

        logger.info(f"断点续训: 从步骤 {global_step}, 轮次 {start_epoch} 继续")

        # 读取之前的统计数据
        try:
            with open(os.path.join(log_dir, "training_stats.json"), 'r') as f:
                prev_stats = json.load(f)
                # 将之前的统计数据导入当前统计对象
                for key in stats.stats:
                    if key in prev_stats:
                        stats.stats[key] = prev_stats[key]

            # 尝试加载轮次统计数据
            try:
                with open(os.path.join(log_dir, "epoch_stats.json"), 'r') as f:
                    stats.epoch_stats = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning("无法加载之前的轮次统计数据")

            logger.info("已加载之前的训练统计数据")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("无法加载之前的训练统计数据，将从头开始记录")

    # 详细的模型信息和GPU位置验证
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    logger.info(f"模型是否在GPU上: {next(model.parameters()).is_cuda}")

    # 初始化数据集和数据加载器
    logger.info(f"加载训练数据: {config['data_path']}")
    train_dataset = PretrainDataset(config['data_path'], tokenizer, max_length=lm_config.max_seq_len)
    logger.info(f"数据集大小: {len(train_dataset)} 样本")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False  # 使用pin_memory加速数据传输到GPU
    )

    # 损失函数
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # 训练参数计算
    steps_per_epoch = len(train_loader)
    total_steps = config['epochs'] * steps_per_epoch
    tokens_per_batch = config['batch_size'] * config['max_seq_len']

    logger.info(f"开始训练:")
    logger.info(f"  每轮步数: {steps_per_epoch}")
    logger.info(f"  总训练步数: {total_steps}")
    logger.info(f"  批次大小: {config['batch_size']}")
    logger.info(f"  序列长度: {config['max_seq_len']}")
    logger.info(f"  每批次词元数: {tokens_per_batch}")

    # 训练循环
    start_time = time.time()

    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_token_count = 0

        # 添加明显的轮次分隔符到日志
        logger.info("=" * 80)
        logger.info(f"开始轮次 {epoch + 1}/{config['epochs']}")
        logger.info("=" * 80)

        model.train()
        progress_bar = tqdm(train_loader, desc=f"轮次 {epoch + 1}/{config['epochs']}")

        for step, (X, Y, loss_mask) in enumerate(progress_bar):
            step_start = time.time()

            # 确保数据正确移动到GPU
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            # 每100步记录一次内存使用情况(如果使用GPU)
            if step % 100 == 0 and torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                gpu_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU内存使用: {gpu_mem:.2f} GB / {gpu_reserved:.2f} GB")
                # 记录到 TensorBoard
                stats.writer.add_scalar('系统/GPU内存使用', gpu_mem, global_step)

            # 更新学习率（余弦调度）
            lr = config['learning_rate'] * 0.5 * (1 + math.cos(math.pi * global_step / total_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播
            outputs = model(X)
            logits = outputs.logits

            # 计算损失
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            # 添加MoE辅助损失（如果有的话）
            if hasattr(outputs, 'aux_loss'):
                aux_loss = outputs.aux_loss
                loss += aux_loss
                # 记录辅助损失
                stats.writer.add_scalar('训练/辅助损失', aux_loss.item(), global_step)
                stats.writer.add_scalar(f'轮次_{epoch + 1}/辅助损失', aux_loss.item(), global_step)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            # 记录梯度范数
            stats.writer.add_scalar('训练/梯度范数', grad_norm, global_step)
            stats.writer.add_scalar(f'轮次_{epoch + 1}/梯度范数', grad_norm, global_step)

            # 更新参数
            optimizer.step()

            # 计算性能指标
            step_time = time.time() - step_start
            tokens_per_second = tokens_per_batch / step_time

            # 更新累计损失和词元计数
            epoch_loss += loss.item()
            epoch_token_count += tokens_per_batch
            global_step += 1

            # 记录详细的训练状态
            stats.update(
                step=global_step,
                epoch=epoch + 1,
                loss=loss.item(),
                lr=lr,
                tokens_per_second=tokens_per_second
            )

            # 更新进度条信息
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{lr:.6f}",
                'grad_norm': f"{grad_norm:.2f}",
                'tok/s': f"{tokens_per_second:.1f}"
            })

            # 定期保存检查点
            if global_step % config['save_steps'] == 0:
                checkpoint_path = os.path.join(model_dir, f"step_{global_step}.pt")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'best_loss': best_loss,
                    'config': config,  # 保存配置以便恢复训练
                }, checkpoint_path)
                logger.info(f"保存检查点: {checkpoint_path}")

                # 每次保存时也更新最新的模型
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'best_loss': best_loss,
                    'config': config,
                }, os.path.join(model_dir, "latest.pt"))

                # 保存统计数据
                stats.save_stats()

        # 每个epoch结束后的统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / steps_per_epoch
        epoch_tokens_per_second = epoch_token_count / epoch_time

        # 添加明显的轮次结束分隔符
        logger.info("-" * 80)
        logger.info(f"轮次 {epoch + 1} 完成:")
        logger.info(f"  平均损失: {avg_loss:.4f}")
        logger.info(f"  耗时: {epoch_time / 60:.2f}分钟")
        logger.info(f"  每秒处理词元: {epoch_tokens_per_second:.2f}")
        logger.info("-" * 80)

        # 记录轮次统计信息
        stats.finish_epoch(epoch + 1, avg_loss, epoch_time, epoch_tokens_per_second)

        # 保存每个epoch的模型
        epoch_save_path = os.path.join(model_dir, f"epoch_{epoch + 1}.pt")
        torch.save({
            'step': global_step,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'config': config,
        }, epoch_save_path)
        logger.info(f"保存轮次模型: {epoch_save_path}")

        # 如果是最佳模型，单独保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save({
                'step': global_step,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'config': config,
            }, best_model_path)
            logger.info(f"发现更好的模型! 已保存至: {best_model_path}")

    # 训练结束统计
    total_training_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"训练完成!")
    logger.info(f"  总耗时: {total_training_time / 3600:.2f}小时")
    logger.info(f"  总步数: {global_step}")
    logger.info(f"  最佳损失: {best_loss:.4f}")
    logger.info("=" * 80)

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model.pt")
    torch.save({
        'step': global_step,
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'best_loss': best_loss,
        'config': config,
    }, final_model_path)
    logger.info(f"最终模型已保存至: {final_model_path}")

    # 保存完整统计信息
    stats.save_stats()

    return {
        'final_loss': epoch_loss / steps_per_epoch,
        'best_loss': best_loss,
        'training_time': total_training_time,
        'model_dir': model_dir,
        'total_steps': global_step
    }


def main():
    """主函数，设置训练配置并启动训练"""
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        logger.info("CUDA信息:")
        logger.info(f"  可用: 是")
        logger.info(f"  设备数量: {torch.cuda.device_count()}")
        logger.info(f"  当前设备: {torch.cuda.current_device()}")
        logger.info(f"  设备名称: {torch.cuda.get_device_name(0)}")
        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一张GPU
    else:
        logger.info("CUDA不可用，将使用CPU进行训练")

    # 所有训练参数在此处设置，无需命令行参数
    config = {
        # 模型参数
        'dim': 512,  # 模型隐藏层维度
        'n_layers': 8,  # Transformer层数
        'max_seq_len': 512,  # 最大序列长度

        # 训练参数
        'batch_size': 10,  # 训练批次大小
        'learning_rate': 1e-3,  # 初始学习率
        'weight_decay': 0.15,  # 权重衰减
        'epochs': 4,  # 训练轮数
        'grad_clip': 1.0,  # 梯度裁剪阈值

        # 系统参数
        'num_workers': 4,  # 数据加载线程数
        'save_steps': 500,  # 每多少步保存一次检查点

        # 路径参数
        'data_path': "D:/pycharm/minimind/dataset/pretrain_hq.bacckup.jsonl",  # 训练数据路径
        'tokenizer_path': "D:/pycharm/minimind/model/minimind_tokenizer",  # 分词器路径
        'output_dir': "./outputs",  # 输出目录根路径

        # 断点续训参数 - 如果需要从检查点继续训练，取消下行注释并指定检查点路径
        # 'resume_from_checkpoint': "./outputs/run_20250330_123456/checkpoints/latest.pt",
    }

    # 打印配置信息
    logger.info("训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # 启动训练
    train_results = train(config)

    # 打印训练结果摘要
    logger.info("训练完成，结果摘要:")
    logger.info(f"  最终损失: {train_results['final_loss']:.4f}")
    logger.info(f"  最佳损失: {train_results['best_loss']:.4f}")
    logger.info(f"  训练时间: {train_results['training_time'] / 3600:.2f}小时")
    logger.info(f"  总训练步数: {train_results['total_steps']}")
    logger.info(f"  模型保存路径: {train_results['model_dir']}")


if __name__ == "__main__":
    main()