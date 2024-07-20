import pandas as pd
import matplotlib.pyplot as plt
import os

# Improved style
plt.style.use('seaborn-v0_8-darkgrid')

# Current directory and data file path
current_dir = os.getcwd()
filePath = os.path.join(current_dir, 'data_1.xlsx')

# Load data
data = pd.read_excel(filePath)

# Set wavelength as x-axis
wavelength = data['Wavelength (µm)']

# Create a new figure with 3 subplots and specified figure size
fig, axs = plt.subplots(3, 1, figsize=(10, 12), dpi=100)  # Increased DPI for higher resolution

# Loop for 3 devices
for device_num in range(1, 4):  # Devices 1, 2, 3
    # Get LCP and RCP data
    lcp_data = data.iloc[:, 2 * device_num - 1]  # Columns 2, 4, 6
    rcp_data = data.iloc[:, 2 * device_num]      # Columns 3, 5, 7

    # Scale data to (-1, 1)
    max_abs_value = max(lcp_data.abs().max(), rcp_data.abs().max())
    lcp_data_scaled = lcp_data / max_abs_value
    rcp_data_scaled = rcp_data / max_abs_value

    # Plot
    ax = axs[device_num - 1]

    ax.scatter(wavelength, lcp_data_scaled, color='blue', marker='o', edgecolor='black', s=50, label='LCP', alpha=0.7)
    ax.scatter(wavelength, rcp_data_scaled, color='red', marker='o', edgecolor='black', s=50, label='RCP', alpha=0.9)

    # Customizing the subplot
    ax.set_title(f'Device {device_num} Photocurrent vs. Wavelength', fontsize=20)
    ax.set_xlabel('Wavelength (µm)', fontsize=18)
    ax.set_ylabel('Normalized Photocurrent', fontsize=18)
    ax.legend(frameon=True, loc='best', fontsize=12)

    # Improve readability
    ax.tick_params(axis='both', which='major', labelsize=15)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure with a specified DPI
plt.savefig('Devices_scaled_VPlot_new1.png', dpi=300)

# Show the plot
# plt.show()
