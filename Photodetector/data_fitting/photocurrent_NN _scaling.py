'''
Author: Guo wenyu
Date: 2023-11-08 17:01:46
LastEditTime: 2023-11-08 19:09:47
LastEditors: GUOdeMacBook-Air.local
Description: In User Settings Edit
FilePath: /pythonProject/vscode/photocurrent/photocurrent_NN copy.py
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import os


plt.style.use('seaborn-darkgrid') 
current_dir = os.getcwd()
filePath = os.path.join(current_dir, 'data.xlsx')

# load data
data = pd.read_excel(filePath)

# set wavelength as x-axis
wavelength = data['Wavelength (µm)']
lcp_scaler = MinMaxScaler(feature_range=(-1, 1))
rcp_scaler = MinMaxScaler(feature_range=(-1, 1))

# new figure，3 subplots，size 15*5
fig, axs = plt.subplots(3, 1, figsize=(8, 15))

# loop for 3 devices
for device_num in range(1, 4):  # 1,2,3
    # get LCP and RCP data
    lcp_data = data.iloc[:, 2 * device_num - 1]  # column 2,4,6
    rcp_data = data.iloc[:, 2 * device_num]  # column 3,5,7

    # scale data to (-1, 1)
    max_abs_value = max(lcp_data.abs().max(), rcp_data.abs().max())

    lcp_data_scaled = lcp_data / max_abs_value
    rcp_data_scaled = rcp_data / max_abs_value
    # lcp_data_scaled = 2*(lcp_data - lcp_data.min())/(lcp_data.max() - lcp_data.min())-1

    # rcp_data_scaled = -1*((rcp_data-rcp_data).min())/(abs(rcp_data).max()-abs(rcp_data).min()))
    
    # multi-layer perceptron regressor model for LCP
    # lcp_model = MLPRegressor(hidden_layer_sizes=(1000, 1000), activation='relu', solver='adam', max_iter=3000, random_state=42)
    
    # 3 hidden layers model for LCP
    lcp_model =MLPRegressor(
        hidden_layer_sizes=(1000,1000,500),
        activation='relu', # 'identity', 'logistic', 'tanh', 'relu'
        solver='adam',
        alpha=0.0001, # L2 penalty (regularization term) parameter
        batch_size='auto',
        learning_rate='adaptive', # 'constant', 'invscaling', 'adaptive'
        max_iter=5000,
        random_state=42,
    )
    lcp_model.fit(wavelength.values.reshape(-1, 1), lcp_data_scaled.ravel())

    # multi-layer perceptron regressor model for RCP
    # rcp_model = MLPRegressor(hidden_layer_sizes=(1000, 1000), activation='relu', solver='adam', max_iter=3000, random_state=42)

    # 3 hidden layers model for RCP
    rcp_model =MLPRegressor(
        hidden_layer_sizes=(1000,1000,500),
        activation='relu', # 'identity', 'logistic', 'tanh', 'relu'
        solver='adam',
        alpha=0.0001, # L2 penalty (regularization term) parameter
        batch_size='auto',
        learning_rate='adaptive', # 'constant', 'invscaling', 'adaptive'
        max_iter=5000,
        random_state=42,
    )
    rcp_model.fit(wavelength.values.reshape(-1, 1), rcp_data_scaled.ravel())

    # plot
    ax = axs[device_num - 1]  # 0,1,2

    ax.scatter(wavelength, lcp_data_scaled, color='blue', marker='o', edgecolor='black', label='Normalized LCP Data')
    ax.plot(wavelength, lcp_model.predict(wavelength.values.reshape(-1, 1)), color='blue', label='Fitted LCP Curve')

    ax.scatter(wavelength, rcp_data_scaled, color='red', marker='o', edgecolor='black', label='Normalized RCP Data')
    ax.plot(wavelength, rcp_model.predict(wavelength.values.reshape(-1, 1)), color='red', label='Fitted RCP Curve')

    ax.set_title(f'Device {device_num} photocurrent vs. wavelength')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Scaled Photocurrent')
    ax.legend()

plt.tight_layout()
plt.savefig(f'Devices_scaled_NN_VPlot_3.png')
plt.show()


