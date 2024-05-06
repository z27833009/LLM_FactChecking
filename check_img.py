import pandas as pd
import os

# Excel文件路径和要检查的列名
excel_file_path = '/mnt/d/Mocheg-main/Mocheg-main/data/test/no_duplicated.csv'
column_name = 'img_evidence_id'  # Excel中包含文件名的列名
directory_path = '/mnt/d/Mocheg-main/Mocheg-main/data/test/images/'  # 图片文件所在的文件夹路径

def check_files_existence(excel_path, column, directory):
    # 读取Excel文件
    df = pd.read_csv(excel_path)
    
    # 存储不存在的文件名
    not_found_files = []
    
    # 遍历指定列的每个文件名
    for filename in df[column]:
        # 构建文件的完整路径
        file_path = os.path.join(directory, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            not_found_files.append(filename)
            
    return not_found_files

if __name__ == '__main__':
    missing_files = check_files_existence(excel_file_path, column_name, directory_path)
    if missing_files:
        print("以下文件不存在：")
        for file in missing_files:
            print(file)
    else:
        print("所有文件都存在。")
