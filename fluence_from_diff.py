
'''
To compute the fluence of a MicroED experiment, follow these steps:
* Open the dataset in CAP
* run the following command:
    XX IS_RUNLIST 11 11 775 365
* Copy the output (EXCLUDING the table header lines) to a file named 'signal_out.csv' in the experiment folder.
    As CAPs copy function often works unreliably, it is recommended to copy it from the latest log file
    in the experiment's 'log' subfolder.
* Set the base_path variable to the path of the experiment folder and run the script.
'''

import configparser
import os
import csv

base_path = 'C:\\XcaliburData\\Marcus\\Cry11Aa\\grid_1\\exp_9745'
# base_path = 'C:\\XcaliburData\\SPERA\\exp_1426'
exp_name = os.path.split(base_path)[-1]

# Read intensity statistics
int_stats = []

int_stats = []
with open(os.path.join(base_path, 'signal_out.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=' ', 
                            skipinitialspace=True, 
                            fieldnames=['frame', 'sum', 'max', 'min', 'average', 
                                        'sigma', 'system_gain', 'gain', 
                                        'c_x', 'c_y', 'FWHMx', 'FWHMy'])
    
    for row in reader:
        # print(row)
        # processed_row = {key: float(value) for key, value in row.items()} # Convert values to float
        int_stats.append(row)

# Calculate total counts in the diffraction area
total_counts = sum(float(item['sum']) for item in int_stats)

# get diffraction area and gain
counts_per_e, aperture_demag = 2.9, 45
datacoll = configparser.ConfigParser()
datacoll.read(os.path.join(base_path, 'expinfo', f'{exp_name}_datacoll.ini'))
aperture = float(datacoll['MicroED']['Aperture SA info']) / aperture_demag # diffraction area radius in um
diff_area = 3.14 * (aperture*1e4/2) ** 2  # diffraction area in A^2

fluence = total_counts / counts_per_e / diff_area  # in e/A^2
print(f'Fluence in {exp_name} was: {fluence:.3f} e/A^2')