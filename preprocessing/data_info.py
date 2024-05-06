# The direct approach to read the file caused an exception likely due to environment constraints.
# Let's try another method to read the JSON lines (jsonl) file by loading it into a Pandas DataFrame for analysis.

import pandas as pd

# # Attempt to read the file as a DataFrame
# data = pd.read_json('/mnt/d/Mocheg-main/Mocheg-main/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/snopes.test.jsonl', lines=True)

# # Display the first few rows of the DataFrame to understand its structure
# print(data.head())


input_path = "/home/jiajun/project/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/prepared_test_data2.csv"
df = pd.read_csv(input_path, sep='\t')
df = df[df['label'] == "NEI"]
print(df.head())
print(df.loc[df['id'] == 2930, 'claim'].values[0])
print(df.loc[df['id'] == 2930, 'evidence'].values[0])

# print(df['label'].value_counts())
# print(df[df['label']=='NEI'].iloc[5:10])

# print(df.loc[df['id'] == 4489, 'claim'].values[0])
# print(df.loc[df['id'] == 4489, 'evidence'].values[0])
# print(df[df['id']==5514].head())

# input_path1 = "/home/jiajun/project/data/data/data/ukp_snopes_corpus/results/final_test_result.csv"
# input_path2 = "/home/jiajun/project/data/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/final_test_result.csv"
# ground_true_path = "/home/jiajun/project/data/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/prepared_test_data1.csv"
# df = pd.read_csv(ground_true_path)
# df1 = pd.read_csv(input_path2)
# df2 = pd.read_csv(input_path1)
# print(df['final_prediction'].value_counts())
# df['final_prediction'] = df['final_prediction'].replace(
#     {"reft": "refuted", "refted": "refuted", "reuted": "refuted", "refiuted": "refuted"})
# print(df['final_prediction'].value_counts())
# df.to_csv(input_path2, index=False)

# ground_input_path = "/home/jiajun/project/data/data/data/ukp_snopes_corpus/ukp_snopes_corpus/datasets/prepared_test_data1.csv"
# df1 = pd.read_csv(ground_input_path, sep='\t')

# print(df['label'].value_counts())


# print(df['claim'][1])
# print(df['evidence'][1])

# mocheg_input_path = "data/data/test/Corpus2_for_verification.csv"
# df_mocheg = pd.read_csv(mocheg_input_path, usecols=['claim_id', 'Claim', 'Evidence', 'cleaned_truthfulness'])
# print(df_mocheg.head())
# print(df_mocheg['cleaned_truthfulness'].value_counts())