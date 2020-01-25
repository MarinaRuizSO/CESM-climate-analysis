# Python analysis of NetCDF files from climate models 

Python3 code for NetCDF files. Analysis of climatology behaviour, precipitation metrics and significance testing. 

1. `slope_summer_prec_2.py`: Reads the precipitation data from .nc files and calculates the summer climatology for the given time intervals and the two experiments. Produces plots and calculates the significance of the results.
2. `land_sea_mask_2.py`: Contains a function to calculate a land-sea mask matrix from the landfraction data file.
3. `metrics_analysis.py`: Calculates the precipitation metrics RX1day, RX5day, R95p, R99p and CDD from .pkl files created in `pickle_files_precipitation_mask_2.py`. Plots them for the two experiments for the given time periods and calculates the significance of the results.
4. `pickle_files_precipitation_mask_2.py`: Creates .pkl files for easier ingestion in other scripts. Each pickle file contains data from one experiment and one ensemble for the entire time period.


Note: The folders given in the scripts have paths with respect to my own directory. Make sure this is changed as needed.
In order to run `metrics_analysis.py`, you'll need to run `pickle_files_precipitation_mask_2.py` first.

## Prerequisites
Libraries needed (make sure the installed version is compatible with Python3): 
1. `pip3 install matplotlib`
2. `pip3 install numpy` 
3. `pip3 install netCDF4`
4. `pip3 install scipy`
5. `pip3 install scikit-learn`
6. `pip3 install Cartopy`

You should be able to install all the needed dependencies, but if not possible consider creating a virtual environment. Intructions for this are given below:


## Installing 
Create a Python3 virtual environment if you do not have admin permissions (instructions given for Linux). 

### Install virtualenv package with `pip`
```
python3 -m pip install --user virtualenv
```

### Create the virtual environment
```
python3 -m venv my_virtual_env
```

### Activate the virtual environment (Linux)
```
source my_virtual_env/bin/activate
```
The name of the environment will now appear in brackets on the terminal.

### Deactivate the virtual environment
```
deactivate
```


