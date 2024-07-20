'''
Author: Guo Wenyu
Date: 2023-11-02 16:00:37
LastEditTime: 2023-11-02 16:10:48
LastEditors: GUOdeMacBook-Air.local
Description: In User Settings Edit
FilePath: /pythonProject/vscode/photocurrent/dataNormalize.py
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os   


# 读取Excel文件
current_dir = os.getcwd()
filePath = os.path.join(current_dir, 'data.xlsx')


# load data
data = pd.read_excel(filePath)

scaler = MinMaxScaler(feature_range=(0, 1))

devices = [(f"LCP_{i}", f"RCP_{i}") for i in range(1, 4)]

# 对每个设备的LCP和RCP进行归一化
for lcp_col, rcp_col in devices:
    # 使用相同的归一化参数对LCP和RCP进行归一化
    combined_data = pd.concat([data[lcp_col], data[rcp_col]], axis=1)
    normalized_data = scaler.fit_transform(combined_data)

    data[f"{lcp_col}_Normalized"] = normalized_data[:, 0]
    data[f"{rcp_col}_Normalized"] = normalized_data[:, 1]

# 保存归一化后的数据到新的Excel文件
output_file_path = 'normalized_data.xlsx'
data.to_excel(output_file_path, index=False)
