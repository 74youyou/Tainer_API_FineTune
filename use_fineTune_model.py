from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

check_point=".results/checkpoint-690"
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

check_point = os.getenv("USER_FINETUNE_MODEL_CHECKPOINT")


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(check_point)

# 加载训练好的模型
model = AutoModelForSequenceClassification.from_pretrained(check_point)
model.eval()  # 设置为评估模式


def predict_paraphrase(sentence1, sentence2):
    """
    判断两个句子是否是同义句

    Args:
        sentence1: 第一个句子
        sentence2: 第二个句子

    Returns:
        dict: 包含预测结果和置信度的字典
    """
    # 分词
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True)#tokenized化
    collator = DataCollatorWithPadding(tokenizer)
    inputs_data=collator(inputs)#collated化，进行动态填充
    # 预测
    with torch.no_grad():#关闭梯度计算，提高推理速度
        outputs = model(**inputs_data)#解析输入，传给模型
        logits = outputs.logits#获取模型的logits值
        probabilities = torch.softmax(logits, dim=1)#在dim=1上进行softmax，得到每个类别的概率
        prediction = torch.argmax(logits, dim=1).item()#获取logits中最大值的索引，即预测的类别

    # 结果解析
    result = "同义句" if prediction == 1 else "非同义句"
    confidence = probabilities[0][prediction].item()#获取第一个样本的预测类别（即：prediction）的概率值，并转换为Python浮点数（即：item）

    return {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "prediction": result,
        "confidence": round(confidence, 4),
        "label": prediction
    }#封装模型的推理输出


# 示例使用
if __name__ == "__main__":
    print("使用训练好的模型判断同义句")
    print("=" * 50)

    # 测试示例1：同义句
    test1 = predict_paraphrase(
        "The cat sat on the mat.",
        "A cat was sitting on the mat."
    )
    print(f"示例1: {test1['sentence1']}")
    print(f"      {test1['sentence2']}")
    print(f"预测结果: {test1['prediction']} (置信度: {test1['confidence']})")
    print()

    # 测试示例2：非同义句
    test2 = predict_paraphrase(
        "The cat sat on the mat.",
        "The dog chased the ball."
    )
    print(f"示例2: {test2['sentence1']}")
    print(f"      {test2['sentence2']}")
    print(f"预测结果: {test2['prediction']} (置信度: {test2['confidence']})")
    print()

    # 测试示例3：MRPC数据集中的真实示例
    test3 = predict_paraphrase(
        "Amrozi accused his brother, whom he called 'the witness', of deliberately distorting his evidence.",
        "Referring to him as only 'the witness', Amrozi accused his brother of deliberately distorting his evidence."
    )
    print(f"示例3: {test3['sentence1']}")
    print(f"      {test3['sentence2']}")
    print(f"预测结果: {test3['prediction']} (置信度: {test3['confidence']})")
    print("=" * 50)
