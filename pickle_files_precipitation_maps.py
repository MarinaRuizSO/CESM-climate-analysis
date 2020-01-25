import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import netCDF4
from netCDF4 import Dataset
from itertools import chain
import gc
import pickle


#===========================precipitation=============================

prec_AF = []

# get the data from All Forcing  starting from 1960 

for ensemble in range(1,6):
    gc.collect()

    ensemble_files =sorted(glob.glob('./AllForcing/r' + str(ensemble) + '/PREC*.nc'))
    data = [Dataset(ensemble_files[i]) for i in range(len(ensemble_files))]

    prec60 = data[2].variables['PRECC'][:] + data[11].variables['PRECL'][:]

    prec70 = data[3].variables['PRECC'][:] + data[12].variables['PRECL'][:]

    prec80 = data[4].variables['PRECC'][:] + data[13].variables['PRECL'][:]

    prec90 = data[5].variables['PRECC'][:] + data[14].variables['PRECL'][:]

    prec00 = data[7].variables['PRECC'][:] + data[16].variables['PRECL'][:]




    if ensemble==1:
        #import longitude and latitude files only once
        lats = data[0].variables['lat'][:]
        lons = data[0].variables['lon'][:]

    prec_AF.append(np.concatenate((prec60, prec70,prec80,prec90, prec00)))

# creates an array with the concatenated data for the given years
prec_AF = np.array(prec_AF)
prec_AF = np.reshape(prec_AF, (((5*(len(prec60)+len(prec70)+len(prec80)+len(prec90)+ len(prec00)), len(lats), len(lons)))))
total_length = len(prec60)+len(prec70)+len(prec80)+len(prec90)+ len(prec00)

# saves the data in separate pickle files

file_prec_AF_1 = open('prec_AF_1.pkl', 'wb')
pickle.dump(prec_AF[:total_length, :, :], file_prec_AF_1)

file_prec_AF_2 = open('prec_AF_2.pkl', 'wb')
pickle.dump(prec_AF[total_length:(2*total_length), :, :], file_prec_AF_2)

file_prec_AF_3 = open('prec_AF_3.pkl', 'wb')
pickle.dump(prec_AF[(2*total_length):(3*total_length), :, :], file_prec_AF_3)

file_prec_AF_4 = open('prec_AF_4.pkl', 'wb')
pickle.dump(prec_AF[(3*total_length):(4*total_length), :, :], file_prec_AF_4)

file_prec_AF_5 = open('prec_AF_5.pkl', 'wb')
pickle.dump(prec_AF[(4*total_length):(5*total_length), :, :], file_prec_AF_5)



# data for FixedEASO2

for ensemble in range(1,6):
    gc.collect()
    ensemble_files = sorted(glob.glob('./FixedEASO2/r' + str(ensemble) + '/PREC*.nc'))
    data = [Dataset(ensemble_files[i]) for i in range(len(ensemble_files))]

    prec60 = data[2].variables['PRECC'][:] + data[11].variables['PRECL'][:]

    prec70 = data[3].variables['PRECC'][:] + data[12].variables['PRECL'][:]

    prec80 = data[4].variables['PRECC'][:] + data[13].variables['PRECL'][:]

    prec90 = data[5].variables['PRECC'][:] + data[14].variables['PRECL'][:]

    prec00 = data[7].variables['PRECC'][:] + data[16].variables['PRECL'][:]


    if ensemble == 1:
        lats = data[0].variables['lat'][:]
        lons = data[0].variables['lon'][:]

    prec_FixedEASO2.append(np.concatenate((prec60, prec70, prec80, prec90, prec00)))

prec_FixedEASO2 = np.array(prec_FixedEASO2)
prec_FixedEASO2 = np.reshape(prec_FixedEASO2, (((5*(len(prec60)+len(prec70)+len(prec80)+len(prec90)+ len(prec00)), len(lats), len(lons)))))
total_length = len(prec60)+len(prec70)+len(prec80)+len(prec90)+ len(prec00)

file_prec_FixedEASO2_1 = open('prec_FixedEASO2_1.pkl', 'wb')
pickle.dump(prec_FixedEASO2[:total_length, :, :], file_prec_FixedEASO2_1)
print('done1')
file_prec_FixedEASO2_2 = open('prec_FixedEASO2_2.pkl', 'wb')
pickle.dump(prec_FixedEASO2[total_length:(2*total_length), :, :], file_prec_FixedEASO2_2)
print('done2')
file_prec_FixedEASO2_3 = open('prec_FixedEASO2_3.pkl', 'wb')
pickle.dump(prec_FixedEASO2[(2*total_length):(3*total_length), :, :], file_prec_FixedEASO2_3)
print('done3')
file_prec_FixedEASO2_4 = open('prec_FixedEASO2_4.pkl', 'wb')
pickle.dump(prec_FixedEASO2[(3*total_length):(4*total_length), :, :], file_prec_FixedEASO2_4)
print('done4')
file_prec_FixedEASO2_5 = open('prec_FixedEASO2_5.pkl', 'wb')
pickle.dump(prec_FixedEASO2[(4*total_length):(5*total_length), :, :], file_prec_FixedEASO2_5)
print('done5')

