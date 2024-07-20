import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
filePath = os.path.join(current_dir, 'data.xlsx')

# load data
data = pd.read_excel(filePath)

# set wavelength as x-axis
wavelength = data['Wavelength (µm)']
lcp_scaler = MinMaxScaler(feature_range=(-1, 1))
rcp_scaler = MinMaxScaler(feature_range=(-1, 1))

# new figure，3 subplots，size 15*5
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# loop for 3 devices
for device_num in range(1, 4):  # 1,2,3
    # get LCP and RCP data
    lcp_data = data.iloc[:, 2 * device_num - 1]  # column 2,4,6
    rcp_data = data.iloc[:, 2 * device_num]  # column 3,5,7

    # normalize data
    lcp_data_normalized = lcp_scaler.fit_transform(lcp_data.values.reshape(-1, 1))
    rcp_data_normalized = rcp_scaler.fit_transform(rcp_data.values.reshape(-1, 1))

    # multi-layer perceptron regressor model for LCP
    lcp_model = MLPRegressor(hidden_layer_sizes=(1000, 1000), activation='relu', solver='adam', max_iter=3000, random_state=42)
    lcp_model.fit(wavelength.values.reshape(-1, 1), lcp_data_normalized.ravel())

    # multi-layer perceptron regressor model for RCP
    rcp_model = MLPRegressor(hidden_layer_sizes=(1000, 1000), activation='relu', solver='adam', max_iter=3000, random_state=42)
    rcp_model.fit(wavelength.values.reshape(-1, 1), rcp_data_normalized.ravel())

    # plot
    ax = axs[device_num - 1]  # 0,1,2

    ax.scatter(wavelength, lcp_data_normalized, color='blue', label='Normalized LCP Data')
    ax.plot(wavelength, lcp_model.predict(wavelength.values.reshape(-1, 1)), color='blue', label='Fitted LCP Curve')

    ax.scatter(wavelength, rcp_data_normalized, color='red', label='Normalized RCP Data')
    ax.plot(wavelength, rcp_model.predict(wavelength.values.reshape(-1, 1)), color='red', label='Fitted RCP Curve')

    ax.set_title(f'Device {device_num} photocurrent vs. wavelength')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Normalized Photocurrent')
    ax.legend()

plt.tight_layout()
plt.savefig(f'Devices_NN_Plot_1.png')
plt.show()

