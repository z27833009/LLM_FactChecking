import pandas as pd
from tqdm import tqdm
import csv
# from api_generate import generate_response
from api_chat import generate_response
from sklearn.metrics import precision_score, recall_score, f1_score

input_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/prepared_test_data1.csv"
output_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/test_result.csv"
df = pd.read_csv(
    input_path,
    usecols=['id','claim','evidence','label'],
    sep='\t'
    )

# df = df.iloc[0:5]


def clean_pre(pre):
    pre = pre.lower()
    if 'supported' in pre:
        pre = 'supported'
    if 'nei' in pre:
        pre = 'NEI'
    if 'refuted' in pre:
        pre = 'refuted'
    return pre

def eva(labels ,predictions):
        
    precision = precision_score(labels,
                                predictions,
                                average='weighted')
    recall = recall_score(labels,
                        predictions,
                        average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    print(labels.tolist())
    print(predictions)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

#to save prediction
predictions = []
clean_predictions = []
# for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#     claim = row['claim']
#     evidence = row['evidence']
#     message = "claim:" + claim + "/n" + " evidence:" + evidence
#     pre = clean_pre(generate_response(message))
#     predictions.append(pre)

with open(output_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["id", "claim", "evidence", "prediction","cleaned_pre"])

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        claim_id = row['id']
        claim = row['claim']
        evidence = row['evidence']
        message = "claim:" + claim + "/n" + " evidence:" + evidence
        pre = generate_response(message)
        cleaned_pre = clean_pre(pre)
        predictions.append(pre)
        clean_predictions.append(cleaned_pre)
        # 将每次迭代的预测结果立即写入文件
        csv_writer.writerow([claim_id, claim, evidence, pre, cleaned_pre])


# print(df["label"].tolist())
# print(predictions)
print(clean_predictions[1:10])
print(p for p in clean_predictions if p not in ["supported","refuted","NEI"])



