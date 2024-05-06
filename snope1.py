import pandas as pd
from tqdm import tqdm
import csv
from collections import Counter
from api_chat import generate_response
from datetime import datetime
from pathlib import Path
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 构建基础路径
base_path = Path("/home/jiajun/project/data/data/data/ukp_snopes_corpus/results/") / timestamp

# 创建文件夹
base_path.mkdir(parents=True, exist_ok=True)

input_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/prepared_test_data2.csv"

# 正确地构建 output_path 和 details_output_path
output_path = base_path / "final_test_result.csv"
details_output_path = base_path / "details_test_result.csv"


df = pd.read_csv(input_path, usecols=['id', 'claim', 'evidence', 'label'], sep='\t')
# 只取label为SUPPORTED, REFUTED的数据
df = df[df['label'].isin(['supported', 'refuted'])]


def clean_pre(pre):
    pre = pre.lower().strip()
    # 使用正则表达式匹配refuted的各种可能表达
    if re.search(r"refut(ed)?|reft|refted|reuted|refiuted", pre):
        pre = 'refuted'
    elif 'supported' in pre:
        pre = 'supported'
    else:
        pre = 'refuted'
    # # 更严格的NEI判断条件，使用正则表达式匹配nei的表达
    # elif re.search(r"\bnei\b", pre):
    #     pre = 'NEI'
    # else:
    #     print(f"Unknown prediction: {pre}")
    #     pre = 'NEI'
    return pre

def generate_and_clean_predictions():
    predictions = []
    # 这里替换为实际的预测代码
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        claim = row['claim']
        evidence = row['evidence']
        message = "claim:" + claim + "/n " + "evidence:" + evidence
        # 假设这行调用你的API或其他生成预测的方式
        pre = generate_response(message) 
        cleaned_pre = clean_pre(pre)
        predictions.append(cleaned_pre)
    return predictions

# 初始化一个字典，用于保存每个id的所有预测结果
all_predictions = {row['id']: [] for index, row in df.iterrows()}

# 重复预测过程10次，并收集每轮的预测结果
for _ in range(1):
    predictions = generate_and_clean_predictions()
    for index, prediction in enumerate(predictions):
        claim_id = df.iloc[index]['id']
        all_predictions[claim_id].append(prediction)
    
final_predictions = []
detailed_counts = {}
for claim_id, preds in all_predictions.items():
    count = Counter(preds)
    most_common_pred = count.most_common(1)[0][0]
    final_predictions.append((claim_id, most_common_pred))
    detailed_counts[claim_id] = count

# 将最终结果保存到CSV文件中
with open(output_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["id", "final_prediction"])
    for claim_id, prediction in final_predictions:
        csv_writer.writerow([claim_id, prediction])

# 将每个claim的详细统计结果保存到另一个CSV文件中
with open(details_output_path, 'w', newline='') as details_file:
    csv_writer = csv.writer(details_file)
    # 写入表头，包括所有可能的分类
    csv_writer.writerow(["id", "supported", "NEI", "refuted"])
    for claim_id, counts in detailed_counts.items():
        csv_writer.writerow([claim_id, counts.get('supported', 0), counts.get('NEI', 0), counts.get('refuted', 0)])

# evaluation
y_pred = [prediction for _, prediction in final_predictions]

# 假设df['label']包含了真实的标签
y_true = df['label'].tolist()

# 计算性能指标
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

# 打印指标
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

metrics_path = base_path / "classification_metrics.txt"
with open(metrics_path, 'w') as metrics_file:
    metrics_file.write(f"Precision: {precision}\n")
    metrics_file.write(f"Recall: {recall}\n")
    metrics_file.write(f"F1 Score: {f1}\n")
    metrics_file.write(f"Accuracy: {accuracy}\n")
    
report = classification_report(y_true, y_pred,digits=4)
print(report)