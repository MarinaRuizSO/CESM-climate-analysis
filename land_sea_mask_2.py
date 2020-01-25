import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import netCDF4
from netCDF4 import Dataset
from itertools import chain
import gc


# creates a land sea mask based on the landfraction data file 

def mask():
    file_path = ('./LANDFRAC_BTAL_1.cam.h0.2000-01--2006-11.nc')
    data = Dataset(file_path)
    field50 = data.variables['LANDFRAC'][:] 
    field50 = np.mean(field50, axis = 0)
    print('max: {}'.format(np.max(field50)))
    print('shape: {}'.format(np.shape(field50)))
    mask = np.ma.masked_where(field50 == 0,field50 )
    return mask
