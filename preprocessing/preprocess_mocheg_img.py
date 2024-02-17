import pandas as pd

file_path = "/mnt/d/Mocheg-main/Mocheg-main/data/test/"

df_ver = pd.read_csv(
    file_path +"Corpus2_for_verification.csv",
    usecols=['claim_id', 'Claim', 'Evidence', 'cleaned_truthfulness']
)

df_img = pd.read_csv(
    file_path + "img_evidence_qrels.csv",
    usecols= ["TOPIC","RELEVANCY","evidence_id"],
    
)
df_img = df_img[df_img["RELEVANCY"]==1]
df_merged = pd.merge(df_ver, df_img, how='left', left_on='claim_id', right_on='TOPIC')

df_merged.drop(columns=['TOPIC','RELEVANCY'], inplace=True)

# Rename the evidence_id column to img_evidence_id
df_merged.rename(columns={'evidence_id': 'img_evidence_id'}, inplace=True)
# print("join后的数据大小：{} 行".format(df_merged.shape[0]) )

# print(df_merged.head(5))
df_filtered = df_merged[df_merged['img_evidence_id'].notna()]
# print("剔除img_evidence_id: {} 行".format(df_filtered.shape[0]))
# output_file_path = file_path + "with_duplicated.csv"
# df_filtered.to_csv(output_file_path,index=False)
print(df_filtered.head(5))
print(df_filtered.info())
print(df_filtered['cleaned_truthfulness'].value_counts())

# duplicate_claim_ids = df_merged[df_merged['claim_id'].duplicated(keep='first')]
# print("重复的claim_id数量:", len(duplicate_claim_ids))

# 根据claim_id列去重
df_unique = df_filtered.drop_duplicates(subset=['claim_id'])
print(df_unique.head(5))
print(df_unique.info())
print(df_unique['cleaned_truthfulness'].value_counts())
# # 打印去重后的DataFrame
# print(df_unique.shape[0])
# print(df_unique.head(10))
# output_file_path = file_path + "no_duplicated.csv"
# df_unique.to_csv(output_file_path,index=False)