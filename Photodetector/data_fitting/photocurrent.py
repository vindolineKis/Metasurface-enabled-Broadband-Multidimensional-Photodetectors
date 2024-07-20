'''
Author: Guo Wenyu
Date: 2023-10-19 16:34:41
LastEditTime: 2023-11-02 16:32:25
LastEditors: GUOdeMacBook-Air.local
Description: In User Settings Edit
FilePath: /pythonProject/.vscode/photocurrent.py
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler



current_dir = os.getcwd()
filePath = os.path.join(current_dir, 'data.xlsx')

# data
data = pd.read_excel(filePath)

# wavelength as x-axis
wavelength = data['Wavelength (µm)']

lcp_scaler = MinMaxScaler(feature_range=(0, 6000))
rcp_scaler = MinMaxScaler(feature_range=(-6000, 0))
num_columns = 6

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#  for 3 devices
for device_num in range(1, 4):  
    
    # select LCP and RCP data
    lcp_data = data.iloc[:, 2*device_num - 1]
    rcp_data = data.iloc[:, 2*device_num]

    # normalize data
    lcp_data_normalized = lcp_scaler.fit_transform(lcp_data.values.reshape(-1, 1))
    rcp_data_normalized = rcp_scaler.fit_transform(rcp_data.values.reshape(-1, 1))

    # model 
    lcp_model = LinearRegression()
    lcp_model.fit(wavelength.values.reshape(-1, 1), lcp_data_normalized)
    rcp_model = LinearRegression()
    rcp_model.fit(wavelength.values.reshape(-1, 1), rcp_data_normalized)
    # plot
    ax = axs[device_num - 1]  # 0,1,2
    
    

    ax.scatter(wavelength, lcp_data_normalized, color='blue', label='Normalized LCP Data')
    ax.plot(wavelength, lcp_model.predict(wavelength.values.reshape(-1, 1)), color='blue', label='Fitted LCP Line')

    ax.scatter(wavelength, rcp_data_normalized, color='red', label='Normalized RCP Data')
    ax.plot(wavelength, rcp_model.predict(wavelength.values.reshape(-1, 1)), color='red', label='Fitted RCP Line')


    
    ax.set_title(f'Device {device_num} photocurrent vs. wavelength')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Normalized Photocurrent')
    ax.legend()

plt.tight_layout()
plt.savefig(f'Devices_Plot.png')
plt.show()
  
    
# 输出模型参数
print(f'Device {device_num} LCP Model Coefficients: {lcp_model.coef_}, Intercept: {lcp_model.intercept_}')
print(f'Device {device_num} RCP Model Coefficients: {rcp_model.coef_}, Intercept: {rcp_model.intercept_}')
