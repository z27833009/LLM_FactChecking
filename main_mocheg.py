import pandas as pd
from tqdm import tqdm
# from api_generate import generate_response
from api_chat import generate_response
from sklearn.metrics import precision_score, recall_score, f1_score

file_path = "/mnt/d/Mocheg-main/Mocheg-main/data/test/no_duplicated.csv"
df = pd.read_csv(
    file_path,
    usecols=['claim_id', 'Claim', 'Evidence', 'cleaned_truthfulness'],

    )

df=df.iloc[30:40]

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

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    claim = row['Claim']
    evidence = row['Evidence']
    message = "claim:" + claim + "\n" + " evidence:" + evidence
    pre = clean_pre(generate_response(message))
    predictions.append(pre)

print(df["cleaned_truthfulness"].tolist())
print(predictions)
df["predictions"] = predictions
df.to_csv("/mnt/d/Mocheg-main/Mocheg-main/data/test/predictions_text.csv", index=False)

