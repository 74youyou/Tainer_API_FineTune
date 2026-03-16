import os

from dill import settings

import wandb
from dotenv import load_dotenv
import sys

from sympy import evaluate


def load_env():
    try:
        load_dotenv()
        print("✅ 从 .env 文件加载环境变量成功")
    except Exception as e:
        print(f"❌ 从 .env 文件加载环境变量失败: {e}")
        sys.exit()


from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModel, AutoModelForPreTraining, \
    AutoModelForSequenceClassification, Trainer


def load_tokenizer(check_point):
    try:
        tokenizer = AutoTokenizer.from_pretrained(check_point, truncation=True, local_files_only=True)
        print(f"✅ 从本地缓存加载 Tokenizer 成功")
        return tokenizer
    except(OSError, FileNotFoundError):
        print(f"⚠️ 本地无缓存，正在从网络下载 {check_point}...")
        tokenizer = AutoTokenizer.from_pretrained(
            check_point,
            truncation=True,
            local_files_only=False  # 允许联网下载

        )
        print(f"✅ 下载并缓存 Tokenizer 成功")
        return tokenizer


def tokenize_function(example):  # 给训练样本map函数用的映射函数
    return tokenizer(example["sentence1"], example['sentence2'], truncation=True)


def load_TrainingArgs_return_TrainingArgs():
    """
     从.env环境中加载超参数信息，并实例化TrainingArguments对象

     Returns:
         TrainingArguments: 包含超参数信息的TrainingArguments对象
     """
    try:
        training_args = TrainingArguments(
            # 训练的超参数
            output_dir=os.getenv("OUTPUT_DIR"),
            num_train_epochs=int(os.getenv("NUM_TRAIN_EPOCHS")),
            weight_decay=float(os.getenv("WEIGHT_DECAY")),
            learning_rate=float(os.getenv("LEARNING_RATE")),
            per_device_train_batch_size=int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE")),
            # 评估的超参数
            eval_strategy=os.getenv("EVAL_STRATEGY"),
            eval_steps=None if os.getenv("EVAL_STEPS") == "None" else int(os.getenv("EVAL_STEPS")),
            per_device_eval_batch_size=int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE")),
            # 保存的超参数
            save_strategy=os.getenv("SAVE_STRATEGY"),
            save_steps=None if os.getenv("SAVE_STEPS") == "None" else int(os.getenv("SAVE_STEPS")),
            report_to=os.getenv("REPORT_TO"),
        )
        print(f"✅ 取自.env文件信息的TrainingArguments实例化成功")
    except Exception as e:
        print(f"❌ 取自.env文件信息的TrainingArguments实例化失败: {e}")
    return training_args


import evaluate
import numpy as np


def compute_metric(eval_preds: tuple[np.array, np.array]) -> dict[str, float]:
    """
     计算评估指标

     Args:
         eval_preds: 包含模型预测结果和真实标签的元组
                    第一个元素是模型输出的logits
                    第二个元素是真实标签
     """
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    pred_labels = np.argmax(logits, axis=-1)  # 该函数返回最大值logits数组中，最大值的索引
    return metric.compute(predictions=pred_labels, references=labels)

def load_Dataset_operate():
    try:
        raw_datasets = load_from_disk("./dataset")#从磁盘加载数据集的函数
        print(f"✅ 从本地缓存加载数据集成功")
        return raw_datasets
    except Exception as e:
        print(f"⚠️ 本地无缓存，正在从网络下载数据集...")
        raw_datasets = load_dataset("glue", "mrpc")  # 从网络加载数据集的函数
        raw_datasets.save_to_disk("./dataset")#保存数据集的函数
        print(f"✅ 下载并缓存数据集成功")
        return raw_datasets

load_env()  # 加载环境
# ----数据process部分
raw_datasets=load_Dataset_operate()

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

wandb.init(
    entity=os.getenv("WANDB_ENTITY"),
    project=os.getenv("WANDB_PROJECT"),
    name="bert-mrpc-analysis",
)  # 加载wandb项目，用于可视化

check_point = os.getenv("CHECKPOINT")  # 读取配置信息的检查点（即：CheckPoint）
tokenizer = load_tokenizer(check_point)  # 从检查点中，获取分词器
collator = DataCollatorWithPadding(tokenizer)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # 将映射函数作用于训练样本
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])

# ----获取Train需要实例的部分
training_args = load_TrainingArgs_return_TrainingArgs()  # 获取超参数实例
model = AutoModelForSequenceClassification.from_pretrained(check_point, num_labels=2)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    compute_metrics=compute_metric,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
)
trainer.train()
