import json
import csv
import pandas as pd

def read_jsonl(file_path):
    """Read a JSON Lines file and return the data."""
    # Open the JSON Lines file and read each line
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def read_json(file_path):
    """Read a JSON file and return the data."""
    # Open the JSON file and load the data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def prepare_data1(train_data, evidence_data):
    """Prepare the data for model input."""
    records = []
    for item in train_data:
        evidence_texts = []
        for evidence_group in item['evidence']:
            for source, index in evidence_group:
                # get corresponding evidence sentence
                line = evidence_data[source]['lines'][index] 
                evidence_texts.append(f"{index}: {line}")
        # change label
        label = item['label'].lower()
        if label == 'not enough info':
            label = 'NEI'
        # add record
        records.append({
            'id': item['id'],
            'claim': item['claim'],
            'evidence': ' '.join(evidence_texts),
            'label': label
        })
    return records

def prepare_data(train_data, evidence_data):
    """Prepare the data for model input, skipping entries without evidence."""
    records = []
    for item in train_data:
        evidence_texts = []
        i = 0
        for evidence_group in item['evidence']:
            for source, index in evidence_group:
                # 获取对应的证据句子，如果存在
                i += 1
                if source in evidence_data and index < len(evidence_data[source]['lines']):
                    line = evidence_data[source]['lines'][index]
                    evidence_texts.append(f"{i}: {line}")
        # 只有当evidence_texts不为空时，才添加记录
        if evidence_texts:  # 确保evidence_texts不为空
            label = item['label'].lower()
            if label == 'not enough info':
                label = 'NEI'
            records.append({
                'id': item['id'],
                'claim': item['claim'],
                'evidence': ' '.join(evidence_texts),
                'label': label
            })
    return records

dir = '/mnt/d/Mocheg-main/Mocheg-main/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/'
# Read the trai/dev/test data from the JSON Lines file
train_data = read_jsonl(dir + 'snopes.train.jsonl')
dev_data = read_jsonl(dir + 'snopes.dev.jsonl')
test_data = read_jsonl(dir + 'snopes.test.jsonl')
# Read the evidence data from the JSON file
evidence_data = read_json(dir + 'snopes.page.json')

# Prepare the final dataset for the model input
prepared_train_data = prepare_data(train_data, evidence_data)
prepared_dev_data = prepare_data(dev_data, evidence_data)
prepared_test_data = prepare_data(test_data, evidence_data)

# Create a DataFrame from the prepared data
df = pd.DataFrame(prepared_train_data)
df1 = pd.DataFrame(prepared_dev_data)
df2 = pd.DataFrame(prepared_test_data)

df['label'] = df['label'].replace('refutes', 'refuted')
df1['label'] = df1['label'].replace('refutes', 'refuted')
df2['label'] = df2['label'].replace('refutes', 'refuted')
# supported
df['label'] = df['label'].replace('supports', 'supported')
df1['label'] = df1['label'].replace('supports', 'supported')
df2['label'] = df2['label'].replace('supports', 'supported')

df['evidence'].apply(lambda x: x.replace('\t', ' '))
df1['evidence'].apply(lambda x: x.replace('\t', ' '))
df2['evidence'].apply(lambda x: x.replace('\t', ' '))
# Save the DataFrame to CSV files
df.to_csv(dir + 'prepared_train_data2.csv', sep='\t', quoting=csv.QUOTE_ALL, index=False, encoding='utf-8')
df1.to_csv(dir + 'prepared_dev_data2.csv', sep='\t', quoting=csv.QUOTE_ALL, index=False, encoding='utf-8')
df2.to_csv(dir + 'prepared_test_data2.csv', sep='\t', quoting=csv.QUOTE_ALL, index=False, encoding='utf-8')



print('train_data: ',df.shape[0],'行\n',df['label'].value_counts())
print('dev_data: ',df1.shape[0],'行\n',df1['label'].value_counts())
print('test_data: ',df2.shape[0],'行\n',df2['label'].value_counts())
# print((df2 == '').sum())

# Print the first few rows of the DataFrame to verify its contents
# print(df.iloc[0])
