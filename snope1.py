import pandas as pd
from tqdm import tqdm
import csv
from collections import Counter
from api_chat import generate_response

input_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/prepared_test_data1.csv"
output_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/final_test_result.csv"
details_output_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/details_test_result.csv"

df = pd.read_csv(input_path, usecols=['id', 'claim', 'evidence', 'label'], sep='\t')

def clean_pre(pre):
    pre = pre.lower()
    if 'supported' in pre:
        pre = 'supported'
    elif 'nei' in pre:
        pre = 'NEI'
    elif 'refuted' in pre:
        pre = 'refuted'
    return pre

def generate_and_clean_predictions():
    predictions = []
    # 这里替换为实际的预测代码
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        claim = row['claim']
        evidence = row['evidence']
        message = "claim:" + claim + "/n" + " evidence:" + evidence
        # 假设这行调用你的API或其他生成预测的方式
        pre = generate_response(message) 
        cleaned_pre = clean_pre(pre)
        predictions.append(cleaned_pre)
    return predictions

# 初始化一个字典，用于保存每个id的所有预测结果
all_predictions = {row['id']: [] for index, row in df.iterrows()}

# 重复预测过程10次，并收集每轮的预测结果
for _ in range(10):
    predictions = generate_and_clean_predictions()
    for index, prediction in enumerate(predictions):
        claim_id = df.iloc[index]['id']
        all_predictions[claim_id].append(prediction)

# 分析每个claim的预测结果，选择最常见的类别作为最终预测，并计算每个类别的出现次数
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
