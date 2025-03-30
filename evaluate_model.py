#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# @Time    : 2025/3/30
# @Author  : huangfujue
# @File    : evaluate_model.py
# @Date    : 2025/3/30
"""
模型评估脚本：支持自动测试和交互式测试，评估模型效果和性能
"""
# ----------------------------------------------------------------------------------------------------------------------
import os
import time
import logging
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from LMConfig import LMConfig
from model import MiniMindLM

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationDataset(Dataset):
    """用于评估的数据集"""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                if isinstance(sample, dict) and 'text' in sample:
                    self.samples.append(sample['text'])
                elif isinstance(sample, str):
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        # 编码文本
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze()

        # 分割输入和目标
        X = input_ids[:-1].clone()
        Y = input_ids[1:].clone()

        # 创建损失掩码
        loss_mask = torch.ones_like(Y)

        return X, Y, loss_mask, text


class ModelEvaluator:
    """模型评估器：支持自动和交互式评估"""

    def __init__(self, model_path, tokenizer_path, config_path=None, device=None):
        """
        初始化评估器

        Args:
            model_path: 模型权重路径
            tokenizer_path: 分词器路径
            config_path: 模型配置路径(可选)
            device: 运行设备
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logger.info(f"使用设备: {self.device}")

        # 加载分词器
        logger.info(f"加载分词器: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 加载配置
        if config_path and os.path.exists(config_path):
            logger.info(f"从文件加载配置: {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            self.lm_config = LMConfig(
                dim=config_dict.get('dim', 512),
                n_layers=config_dict.get('n_layers', 8),
                max_seq_len=config_dict.get('max_seq_len', 512)
            )
        else:
            logger.info("使用默认配置")
            self.lm_config = LMConfig(
                dim=512,
                n_layers=8,
                max_seq_len=512
            )

        # 加载模型
        logger.info(f"加载模型: {model_path}")
        self.model = MiniMindLM(self.lm_config).to(self.device)

        # 如果模型路径是目录，尝试找到最佳模型或最新模型
        if os.path.isdir(model_path):
            best_model_path = os.path.join(model_path, "best_model.pt")
            final_model_path = os.path.join(model_path, "final_model.pt")
            latest_model_path = os.path.join(model_path, "latest.pt")

            if os.path.exists(best_model_path):
                logger.info(f"加载最佳模型: {best_model_path}")
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            elif os.path.exists(final_model_path):
                logger.info(f"加载最终模型: {final_model_path}")
                self.model.load_state_dict(torch.load(final_model_path, map_location=self.device))
            elif os.path.exists(latest_model_path):
                logger.info(f"加载最新模型: {latest_model_path}")
                self.model.load_state_dict(torch.load(latest_model_path, map_location=self.device))
            else:
                # 查找最新的epoch模型
                epoch_models = [f for f in os.listdir(model_path) if f.startswith("epoch_") and f.endswith(".pt")]
                if epoch_models:
                    latest_epoch = max(epoch_models, key=lambda x: int(x.split("_")[1].split(".")[0]))
                    latest_path = os.path.join(model_path, latest_epoch)
                    logger.info(f"加载最新轮次模型: {latest_path}")
                    checkpoint = torch.load(latest_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    raise FileNotFoundError(f"在目录 {model_path} 中未找到有效的模型文件")
        else:
            # 直接加载指定的模型文件
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.eval()  # 设置为评估模式

    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
        """
        根据提示生成文本

        Args:
            prompt: 提示文本
            max_length: 最大生成长度
            temperature: 温度参数，控制生成的随机性
            top_k: top-k采样参数
            top_p: 核采样参数

        Returns:
            生成的文本
        """
        # 编码提示文本
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # 记录原始输入长度
        input_len = input_ids.size(1)

        # 开始时间
        start_time = time.time()

        # 自回归生成
        with torch.no_grad():
            for _ in range(max_length):
                # 对于过长的序列，只保留最后512个token
                if input_ids.size(1) > self.lm_config.max_seq_len:
                    input_ids = input_ids[:, -self.lm_config.max_seq_len:]

                # 前向传播
                outputs = self.model(input_ids)
                logits = outputs.logits

                # 获取最后一个token的预测
                next_token_logits = logits[:, -1, :] / temperature

                # Top-K采样
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Top-p (nucleus) 采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

                    # 移除概率累积超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 将第一个token保持为False，确保至少有一个token可以被选择
                    sorted_indices_to_remove[..., 0] = 0

                    # 创建掩码
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # 计算概率分布
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)

                # 拼接到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # 如果生成了结束标记，提前结束
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # 计算生成时间和速度
        generation_time = time.time() - start_time
        tokens_generated = input_ids.size(1) - input_len
        generation_speed = tokens_generated / generation_time if generation_time > 0 else 0

        # 解码生成的文本
        generated_text = self.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time": generation_time,
            "tokens_per_second": generation_speed
        }

    def calculate_perplexity(self, input_ids, target_ids, loss_mask=None):
        """
        计算困惑度

        Args:
            input_ids: 输入ID
            target_ids: 目标ID
            loss_mask: 损失掩码

        Returns:
            困惑度
        """
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            # 计算交叉熵损失
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1)).view(target_ids.size())

            # 应用掩码（如果有）
            if loss_mask is not None:
                loss = (loss * loss_mask).sum() / loss_mask.sum()
            else:
                loss = loss.mean()

            # 计算困惑度
            perplexity = torch.exp(loss)

            return perplexity.item()

    def evaluate_file(self, data_path, batch_size=8, sample_limit=None):
        """
        评估模型在文件上的性能

        Args:
            data_path: 数据文件路径
            batch_size: 批量大小
            sample_limit: 样本限制数量

        Returns:
            评估结果
        """
        logger.info(f"使用数据文件进行评估: {data_path}")

        # 创建数据集和数据加载器
        eval_dataset = EvaluationDataset(data_path, self.tokenizer, max_length=self.lm_config.max_seq_len)

        # 如果指定了样本限制，截取数据集
        if sample_limit and sample_limit < len(eval_dataset):
            indices = torch.randperm(len(eval_dataset))[:sample_limit].tolist()
            subset_samples = [eval_dataset.samples[i] for i in indices]
            eval_dataset.samples = subset_samples
            logger.info(f"限制评估样本数量为 {sample_limit}")

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        # 评估指标
        total_loss = 0
        total_perplexity = 0
        total_tokens = 0
        total_time = 0

        # 生成样本结果
        generation_samples = []

        logger.info(f"开始评估 ({len(eval_dataset)} 个样本)")
        self.model.eval()

        progress_bar = tqdm(eval_loader, desc="评估中")

        with torch.no_grad():
            for batch_idx, (X, Y, loss_mask, texts) in enumerate(progress_bar):
                batch_start = time.time()

                # 计算困惑度
                perplexity = self.calculate_perplexity(X, Y, loss_mask)
                total_perplexity += perplexity

                # 计算时间和速度
                batch_time = time.time() - batch_start
                total_time += batch_time

                # 更新进度条
                progress_bar.set_postfix({
                    "perplexity": f"{perplexity:.2f}"
                })

                # 生成一些样本（每10个批次选择1个样本进行生成测试）
                if batch_idx % 10 == 0:
                    sample_idx = 0  # 选择批次中的第一个样本
                    sample_text = texts[sample_idx]
                    prompt = sample_text[:100]  # 使用前100个字符作为提示

                    generation_result = self.generate(
                        prompt=prompt,
                        max_length=50,
                        temperature=0.7
                    )

                    generation_samples.append(generation_result)

        # 计算平均指标
        avg_perplexity = total_perplexity / len(eval_loader)
        avg_generation_speed = np.mean([s["tokens_per_second"] for s in generation_samples])

        # 保存评估结果
        eval_results = {
            "dataset_size": len(eval_dataset),
            "avg_perplexity": avg_perplexity,
            "avg_generation_speed": avg_generation_speed,
            "generation_samples": generation_samples[:5],  # 只保存前5个样本
            "evaluation_time": total_time
        }

        logger.info(f"评估完成:")
        logger.info(f"  平均困惑度: {avg_perplexity:.4f}")
        logger.info(f"  平均生成速度: {avg_generation_speed:.2f} 词元/秒")
        logger.info(f"  评估总时间: {total_time:.2f} 秒")

        return eval_results

    def interactive_evaluation(self):
        """交互式评估模式"""
        logger.info("进入交互式评估模式 (输入 'exit' 退出)")

        results = []

        while True:
            try:
                prompt = input("\n请输入提示文本 > ")

                if prompt.lower() in ['exit', 'quit', '退出']:
                    break

                print("\n生成中...")

                # 默认参数
                max_length = 100
                temperature = 0.7
                top_k = 50
                top_p = 0.9

                # 自定义参数（可选）
                param_input = input("自定义参数? (y/n, 默认n) > ")
                if param_input.lower() == 'y':
                    try:
                        max_length = int(input("最大生成长度 (默认100) > ") or 100)
                        temperature = float(input("温度 (默认0.7) > ") or 0.7)
                        top_k = int(input("Top-K (默认50, 0表示禁用) > ") or 50)
                        top_p = float(input("Top-P (默认0.9, 1.0表示禁用) > ") or 0.9)
                    except ValueError:
                        print("参数无效，使用默认值")

                # 生成文本
                result = self.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                # 输出结果
                print("\n生成结果:")
                print("-" * 50)
                print(result["generated_text"])
                print("-" * 50)
                print(f"生成 {result['tokens_generated']} 个词元，用时 {result['generation_time']:.2f} 秒")
                print(f"生成速度: {result['tokens_per_second']:.2f} 词元/秒")

                # 保存结果
                results.append(result)

            except KeyboardInterrupt:
                print("\n已中断")
                break
            except Exception as e:
                print(f"错误: {e}")

        logger.info("交互式评估结束")

        # 返回评估结果
        return {
            "interactive_samples": results,
            "avg_generation_speed": np.mean([r["tokens_per_second"] for r in results]) if results else 0
        }

    def save_evaluation_report(self, results, output_dir):
        """
        保存评估报告

        Args:
            results: 评估结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 保存评估结果为JSON
        with open(os.path.join(output_dir, "evaluation_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 创建HTML报告
        html_report = os.path.join(output_dir, "evaluation_report.html")

        with open(html_report, 'w', encoding='utf-8') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>模型评估报告</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2, h3 { color: #333; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    pre { background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }
                    .sample { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                    .prompt { font-weight: bold; margin-bottom: 10px; }
                    .metrics { color: #666; font-style: italic; }
                </style>
            </head>
            <body>
                <h1>模型评估报告</h1>
            """)

            # 写入评估概要
            f.write("<h2>评估概要</h2>")
            f.write("<table>")
            f.write("<tr><th>指标</th><th>值</th></tr>")

            if "avg_perplexity" in results:
                f.write(f"<tr><td>平均困惑度</td><td>{results['avg_perplexity']:.4f}</td></tr>")

            f.write(f"<tr><td>平均生成速度</td><td>{results['avg_generation_speed']:.2f} 词元/秒</td></tr>")

            if "dataset_size" in results:
                f.write(f"<tr><td>数据集大小</td><td>{results['dataset_size']} 样本</td></tr>")

            if "evaluation_time" in results:
                f.write(f"<tr><td>评估总时间</td><td>{results['evaluation_time']:.2f} 秒</td></tr>")

            f.write("</table>")

            # 写入生成样本
            if "generation_samples" in results:
                f.write("<h2>生成样本</h2>")

                for i, sample in enumerate(results["generation_samples"]):
                    f.write(f"<div class='sample'>")
                    f.write(f"<h3>样本 {i + 1}</h3>")
                    f.write(f"<div class='prompt'>提示: {sample['prompt']}</div>")
                    f.write(f"<pre>{sample['generated_text']}</pre>")
                    f.write(f"<div class='metrics'>")
                    f.write(f"生成 {sample['tokens_generated']} 个词元，用时 {sample['generation_time']:.2f} 秒，")
                    f.write(f"速度: {sample['tokens_per_second']:.2f} 词元/秒")
                    f.write(f"</div>")
                    f.write(f"</div>")

            # 写入交互式样本
            if "interactive_samples" in results and results["interactive_samples"]:
                f.write("<h2>交互式评估样本</h2>")

                for i, sample in enumerate(results["interactive_samples"]):
                    f.write(f"<div class='sample'>")
                    f.write(f"<h3>交互样本 {i + 1}</h3>")
                    f.write(f"<div class='prompt'>提示: {sample['prompt']}</div>")
                    f.write(f"<pre>{sample['generated_text']}</pre>")
                    f.write(f"<div class='metrics'>")
                    f.write(f"生成 {sample['tokens_generated']} 个词元，用时 {sample['generation_time']:.2f} 秒，")
                    f.write(f"速度: {sample['tokens_per_second']:.2f} 词元/秒")
                    f.write(f"</div>")
                    f.write(f"</div>")

            f.write("""
            </body>
            </html>
            """)

        logger.info(f"评估报告已保存至: {html_report}")

        # 生成困惑度柱状图(如果有自动评估结果)
        if "avg_perplexity" in results:
            plt.figure(figsize=(8, 6))
            plt.bar(["平均困惑度"], [results["avg_perplexity"]])
            plt.title("模型困惑度")
            plt.ylabel("困惑度")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "perplexity.png"))

        # 生成速度柱状图
        plt.figure(figsize=(8, 6))
        plt.bar(["平均生成速度"], [results["avg_generation_speed"]])
        plt.title("生成速度")
        plt.ylabel("词元/秒")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "generation_speed.png"))


def main():
    """主函数，设置训练配置并启动评估"""

    # 硬编码的评估参数
    config = {
        # 模型路径
        "model_path": "./outputs/run_20250330_032218/checkpoints",

        # 分词器路径
        "tokenizer_path": "D:/pycharm/minimind/model/minimind_tokenizer",

        # 配置文件路径（可选）
        "config_path": "./outputs/run_20250330_120000/config.json",

        # 输出目录
        "output_dir": "./evaluation_results",

        # 评估模式: "auto", "interactive", 或 "both"
        "mode": "both",

        # 自动评估的数据文件路径
        "eval_file": "D:/pycharm/minimind/dataset/test_data.jsonl",

        # 评估批次大小
        "batch_size": 8,

        # 评估样本数量限制（可选，若设为None则使用全部样本）
        "sample_limit": 100,

        # 生成参数
        "generation": {
            "max_length": 100,  # 最大生成长度
            "temperature": 0.7,  # 温度参数
            "top_k": 50,  # top-k采样参数
            "top_p": 0.9  # 核采样参数
        }
    }

    # 创建时间戳目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"eval_{timestamp}")

    # 输出评估配置
    logger.info("评估配置:")
    for key, value in config.items():
        if key != "generation":
            logger.info(f"  {key}: {value}")

    # 初始化评估器
    evaluator = ModelEvaluator(
        model_path=config["model_path"],
        tokenizer_path=config["tokenizer_path"],
        config_path=config["config_path"]
    )

    results = {}

    # 自动评估
    if config["mode"] in ["auto", "both"]:
        if config["eval_file"]:
            logger.info("开始自动评估...")
            auto_results = evaluator.evaluate_file(
                data_path=config["eval_file"],
                batch_size=config["batch_size"],
                sample_limit=config["sample_limit"]
            )
            results.update(auto_results)
        else:
            logger.warning("未指定评估文件，跳过自动评估")

    # 交互式评估
    if config["mode"] in ["interactive", "both"]:
        logger.info("开始交互式评估...")
        interactive_results = evaluator.interactive_evaluation()
        results.update(interactive_results)

    # 保存评估报告
    evaluator.save_evaluation_report(results, output_dir)

    logger.info(f"评估完成，结果已保存至: {output_dir}")
    logger.info(f"HTML报告: {os.path.join(output_dir, 'evaluation_report.html')}")


if __name__ == "__main__":
    main()