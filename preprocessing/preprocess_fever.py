import json

def read_first_n_lines(file_path, n=5):
    data = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= n:  # 只读取前n行
                break
            data.append(json.loads(line))
    return data

# 使用示例
file_path = 'train.jsonl'  # 你的文件路径
first_n_entries = read_first_n_lines("/mnt/d/Mocheg-main/Mocheg-main/data/"+ file_path, 10)
# for entry in first_n_entries:
#     print(entry)
print(first_n_entries[9])
