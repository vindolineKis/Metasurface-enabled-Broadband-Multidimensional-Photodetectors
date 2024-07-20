import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('seaborn-darkgrid') 
current_dir = os.getcwd()
filePath = os.path.join(current_dir, 'data.xlsx')

# load data
data = pd.read_excel(filePath)

# set wavelength as x-axis
wavelength = data['Wavelength (µm)']

# new figure, 3 subplots, size 15*5
fig, axs = plt.subplots(3, figsize=(8, 15))

# Keep history of each model
historys = []

# early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10)

# Loop for 3 devices
for device_num in range(1, 4):  # 1,2,3
    # get LCP and RCP data
    lcp_data = data.iloc[:, 2 * device_num - 1]  # column 2,4,6
    rcp_data = data.iloc[:, 2 * device_num]      # column 3,5,7

    # Scale data to (-1, 1)
    max_abs_value = max(lcp_data.abs().max(), rcp_data.abs().max())
    lcp_data_scaled = lcp_data / max_abs_value
    rcp_data_scaled = rcp_data / max_abs_value

    # Combined LCP and RCP data for training
    targets_combined = np.hstack((lcp_data_scaled, rcp_data_scaled))
    wavelengths_combined = np.vstack((wavelength, wavelength))

    # Create and compile the model
    model = Sequential([
        Dense(500, input_dim=1, activation='relu'), 
        Dense(500, activation='relu'), 
        Dense(500, activation='relu'), 
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(
        wavelengths_combined,
        targets_combined,
        epochs=500,
        verbose=0,
        callbacks=[early_stopping]
    )
    historys.append(history.history)

    # Predict using the trained model
    lcp_predictions = model.predict(wavelength.values.reshape(-1, 1))[:len(wavelength)]
    rcp_predictions = model.predict(wavelength.values.reshape(-1, 1))[len(wavelength):]

    # Plot the data and predictions
    ax = axs[device_num - 1]
    ax.scatter(wavelength, lcp_data_scaled, color='blue', marker='o', edgecolor='black', label='Normalized LCP Data')
    ax.plot(wavelength, lcp_predictions, color='blue', label='Fitted LCP Curve')
    ax.scatter(wavelength, rcp_data_scaled, color='red', marker='o', edgecolor='black', label='Normalized RCP Data')
    ax.plot(wavelength, rcp_predictions, color='red', label='Fitted RCP Curve')
    ax.set_title(f'Device {device_num} Photocurrent vs. Wavelength')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Scaled Photocurrent')
    ax.legend()

# Print training history for each device
for device_num, history in enumerate(historys, 1):
    print(f"Device {device_num} training history:")
    for key, values in history.items():
        print(f"{key}: {values[-1]}")  # Print the last value for simplicity

plt.tight_layout()
plt.savefig('Devices_scaled_Dense_Plot.png')
plt.show()
