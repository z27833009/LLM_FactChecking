import re
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from tqdm.auto import tqdm  
from api_chat_llava import generate_response
# from api_chat_llama_fc import generate_response
import logging
from datetime import datetime

log_directory = "log/"
log_filename = datetime.now().strftime(log_directory + "refuted_wrong_pre_log_%Y-%m-%d_%H-%M-%S.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

file_path = "/mnt/d/Mocheg-main/Mocheg-main/data/test/no_duplicated.csv"
df = pd.read_csv(
    file_path,
    usecols=['claim_id', 'Claim', 'Evidence', 'cleaned_truthfulness','img_evidence_id'],
)
# indexes_to_drop = df[df['claim_id'].isin([1399, 13855, 6206, 13739, 9720, 6415, 2577, 6236, 10220, 6291, 14873, 1433, 5493, 7943, 9514, 14565, 1090, 5183, 9560, 13777, 9918, 11258, 12879, 10203, 10203, 1971, 13598, 9102, 169, 8413, 14905, 259, 900, 419, 1190, 1216, 12320,8616,8519,3539,2209,4286,11060,4286,194,12433,3539,7679,1126,3450, 7418, 1194,13011,13827, 7655, 2491,7408])].index
# df = df.drop(indexes_to_drop)


# input_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/prepared_test_data2.csv"
# df = pd.read_csv(input_path, usecols=['id', 'claim', 'evidence', 'label'], sep='\t')
# df.columns = ['claim_id', 'Claim', 'Evidence', 'cleaned_truthfulness']

def clean_pre(pre):
    pre = pre.lower().strip()
    # 使用正则表达式匹配refuted的各种可能表达
    if re.search(r"refut(ed)?|reft|refted|reuted|refiuted", pre):
        return 'refuted'
    elif 'supported' in pre:
        return 'supported'
    # 更严格的NEI判断条件，使用正则表达式匹配nei的表达
    elif re.search(r"\bnei\b", pre):
        return 'NEI'
    else:
        print(f"Unknown prediction: {pre}")
        return 'NEI'

def evaluate_model(df, n_splits=5, n_samples=50):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=n_samples, random_state=42)
    reports = []
    accuracies = []
    
    for i, (train_index, test_index) in enumerate(tqdm(list(sss.split(df, df['cleaned_truthfulness'])), desc="Evaluating"), start=1):
        logging.info(f"Starting evaluation round {i}")
        sampled_df = df.iloc[test_index]
        predictions = []
        
        for index, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0], leave=False, desc="Predicting"):
            claim = row['Claim']
            evidence = row['Evidence']
            message = f"claim: {claim} evidence: {evidence}"
            #  "claim:" + claim + "\n " + "evidence:" + evidence
            # pre = clean_pre(generate_response(message))
            # logging.info(f"Processing  claim_id: {row['claim_id']}, pre: {pre}")
            # logging.info(f"Processing  claim_id: {row['claim_id']}, img_evidence_id: {row['img_evidence_id']}")
            pre = clean_pre(generate_response(message, row['img_evidence_id']))
            # logging.info(f"Processing  claim_id: {row['claim_id']}, pre: {pre}, label:{row['cleaned_truthfulness']}")
            # if pre != 'refuted' and row['cleaned_truthfulness'] == 'refuted':
            #     logging.info(f"Processing  claim_id: {row['claim_id']}, pre: {pre}, label:{row['cleaned_truthfulness']} , img_evidence_id: {row['img_evidence_id']}")
            logging.info(f"Processing  claim_id: {row['claim_id']}, pre: {pre}, label:{row['cleaned_truthfulness']} , img_evidence_id: {row['img_evidence_id']}")
            predictions.append(pre)
        
        labels = sampled_df['cleaned_truthfulness'].tolist()
        accuracy = accuracy_score(labels, predictions)  # 计算当前轮次的准确度
        accuracies.append(accuracy)  # 将当前轮次的准确度添加到列表中
        logging.info(f"Accuracy for round {i}: {accuracy:.4f}")  # 日志记录当前轮次的准确度
        
        report = classification_report(labels, predictions, output_dict=True, zero_division=1)
        report['accuracy'] = accuracy  # 将准确度添加到报告字典中
        reports.append(report)
        
        logging.info(f"Finished evaluation round {i}. Detailed classification report for this round:")
        # 将本轮的评估报告写入日志
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # 如果metrics是字典，迭代它的项
                logging.info(f"Label: {label}")
                for metric, value in metrics.items():
                    logging.info(f"{metric}: {value:.4f}")
            else:
                # 如果metrics不是字典（例如，accuracy这样的单一数值），直接记录这个值
                logging.info(f"{label}: {metrics:.4f}")
    
    avg_accuracy = np.mean(accuracies)
    print(f"\nOverall average accuracy: {avg_accuracy:.4f}")
    logging.info(f"Overall average accuracy: {avg_accuracy:.4f}") # 将平均准确度写入日志
                
    # 计算所有重复实验的平均性能
    avg_report = {}
    for label in reports[0].keys():
        if isinstance(reports[0][label], dict):  # 确保只处理字典类型的条目
            dict_sum = {}
            for metric in reports[0][label].keys():
                dict_sum[metric] = np.mean([report[label][metric] for report in reports if label in report])
            avg_report[label] = dict_sum
    
    print("Detailed classification report for each label:")
    for label, metrics in avg_report.items():
        print(f"\nLabel: {label}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

# 执行评估
evaluate_model(df)