import pandas as pd
from tqdm import tqdm
from api_chat_llava import generate_response
from result_evaluation import evaluation

file_path = "/mnt/d/Mocheg-main/Mocheg-main/data/test/no_duplicated.csv"
image_folder = "/mnt/d/Mocheg-main/Mocheg-main/data/test/images/"
"""
load the data of file_path as a dataframe
column names: claim_id, Claim, Evidence, cleaned_truthfulness, img_evidence_id
img_evidence_id is image file names in image_path
write api_chat.generate_response(message) to get the prediction
"""
df = pd.read_csv(
    file_path,
    usecols=['claim_id', 'Claim', 'Evidence', 'cleaned_truthfulness', 'img_evidence_id']
    )
# select 5th row of df
# print(df.iloc[5])


def clean_pre(pre):
    pre = pre.lower()
    if 'supported' in pre:
        pre = 'supported'
    if 'nei' in pre:
        pre = 'NEI'
    if 'refuted' in pre:
        pre = 'refuted'
    return pre

predictions = []


for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    
    claim = row['Claim']
    evidence = row['Evidence']
    message = "claim:" + claim + "\n" + " evidence:" + evidence 
    # + "\n" + "Tell me if this claim should be categorized as support, rebuttal, or NEI "
    image_path = image_folder + row['img_evidence_id']
    
    # pre = generate_response(message, image_path)
    pre = clean_pre(generate_response(message, image_path))
    predictions.append(pre)

print(df["cleaned_truthfulness"].tolist())
print(predictions)

# save the claim_id, Claim, Evidence, cleaned_truthfulness, and predictions to a new csv file
df["predictions"] = predictions
df.to_csv("/mnt/d/Mocheg-main/Mocheg-main/data/test/predictions_image.csv", index=False)

# evaluation(df["cleaned_truthfulness"].tolist(), predictions)

# print(df["cleaned_truthfulness"].tolist())
# print(predictions)