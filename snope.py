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
        # It is assumed that generate_response is the API you call or the function the model uses to generate predictions.
        pre = generate_response(message) # Suppose this line calls your API or other way of generating predictions
        cleaned_pre = clean_pre(pre)
        predictions.append(cleaned_pre)
    return predictions

# Initialise a dictionary to hold all predictions for each id
all_predictions = {row['id']: [] for index, row in df.iterrows()}

# Repeat the prediction process 10 times
for _ in range(10):
    predictions = generate_and_clean_predictions()
    for index, prediction in enumerate(predictions):
        claim_id = df.iloc[index]['id']
        all_predictions[claim_id].append(prediction)

# Analyse the predictions for each claim and select the most common category as the final prediction
final_predictions = []
for claim_id, preds in all_predictions.items():
    most_common_pred = Counter(preds).most_common(1)[0][0]
    final_predictions.append((claim_id, most_common_pred))

# Save the final result to a CSV file
with open(output_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["id", "final_prediction"])
    for claim_id, prediction in final_predictions:
        csv_writer.writerow([claim_id, prediction])
