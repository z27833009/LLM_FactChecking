import pandas as pd
from tqdm import tqdm
import csv
# from api_generate import generate_response
from api_chat import generate_response
from collections import Counter

input_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/prepared_test_data1.csv"
output_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/final_test_result.csv"
df = pd.read_csv(input_path, usecols=['id','claim','evidence','label'], sep='\t')

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
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        claim = row['claim']
        evidence = row['evidence']
        message = "claim:" + claim + "/n" + " evidence:" + evidence
        # 这里假设 generate_response 是你调用的API或者模型生成预测的函数
        pre = generate_response(message) # 假设这行调用你的API或其他生成预测的方式
        cleaned_pre = clean_pre(pre)
        predictions.append(cleaned_pre)
    return predictions

# 初始化一个字典，用于保存每个id的所有预测结果
all_predictions = {row['id']: [] for index, row in df.iterrows()}

# 重复预测过程10次
for _ in range(10):
    predictions = generate_and_clean_predictions()
    for index, prediction in enumerate(predictions):
        claim_id = df.iloc[index]['id']
        all_predictions[claim_id].append(prediction)

# 分析每个claim的预测结果，选择最常见的类别作为最终预测
final_predictions = []
for claim_id, preds in all_predictions.items():
    most_common_pred = Counter(preds).most_common(1)[0][0]
    final_predictions.append((claim_id, most_common_pred))

# 将最终结果保存到CSV文件中
with open(output_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["id", "final_prediction"])
    for claim_id, prediction in final_predictions:
        csv_writer.writerow([claim_id, prediction])
