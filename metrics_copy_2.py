import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import glob
import netCDF4
from netCDF4 import Dataset
from itertools import chain
import gc
import pickle
from scipy.stats import mannwhitneyu
from sklearn.utils import resample
from land_sea_mask import mask

#===================Get the precipitation metrics=============================

# Import Land-Sea Mask
masked_array = mask()

def plot_prec(ax, data, lon_significance, lat_significance, title, diff=False, reg=False, significance = False):
    if diff == False and reg == False: # Climatology only seasonal
        cmap = plt.cm.terrain_r
        extend = 'max'
        clevs = np.arange(0,20,1)


    elif diff == False and reg == True: # Regression only
        cmap = plt.cm.Blues
        extend = 'max'
        clevs = np.arange(0,50,5)
        #clevs = np.array([0,0.1,0.15,0.2,0.25,0.3,0.4,0.7,1])
    elif diff == True and reg == True: # Difference in regression
        cmap = plt.cm.RdBu
        extend = 'max'
        clevs = np.arange(0,50,5)
        #clevs = np.array([0,0.1,0.15,0.2,0.25,0.3,0.4,0.7,1])
    else:
        cmap = plt.cm.RdBu_r # coolwarm for CDD only Difference in climatology
        extend = 'both'
        clevs = np.arange(-4,5,1)

    #add coastlines
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.8, alpha=1)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.8, alpha=1)
    #define x ticks locations
    xticks = np.arange(-180,180,20)
    yticks = np.arange(-90,90,10)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    # label longitude and latitude
    lon_format = LongitudeFormatter(degree_symbol='')
    lat_format = LatitudeFormatter(degree_symbol='')
    ax.xaxis.set_ticklabels(ax.get_xticks(), fontsize=14.)
    ax.yaxis.set_ticklabels(ax.get_yticks(), fontsize=14.)
    ax.xaxis.set_major_formatter(lon_format)
    ax.yaxis.set_major_formatter(lat_format)
    # plot data as filled contours
    plot = ax.contourf(lons, lats, data, cmap=cmap, extend=extend, levels=clevs)
    if significance==True:
        ax.scatter(lon_significance,lat_significance, c='black',  marker="o",s=5)
    ax.set_extent([60,150,-5,45])
    #ax.set_title(title, fontsize=14.)
    cbar = plt.colorbar(plot, orientation = 'horizontal', pad = 0.07)
    cbar.ax.tick_params(labelsize=14.)


# the .pkl files are already created and have less data which has already been sorted
def get_next_dataset_AF(i = 0, names = ['prec_AF_1.pkl', 'prec_AF_2.pkl', 'prec_AF_3.pkl', 'prec_AF_4.pkl', 'prec_AF_5.pkl']):
    while i < len(names):
        yield pickle.load(open(names[i], 'rb'))
        i += 1

def get_next_dataset_FixedEASO2(i = 0, names = ['prec_FixedEASO2_1.pkl', 'prec_FixedEASO2_2.pkl', 'prec_FixedEASO2_3.pkl', 'prec_FixedEASO2_4.pkl', 'prec_FixedEASO2_5.pkl']):
    while i < len(names):
        yield pickle.load(open(names[i], 'rb'))
        i += 1
# uses this file just to get lon and lat values
file_path =('./AllForcing/r1/PRECC_BTAL_1.cam.h1.1950-1959.nc')
data = Dataset(file_path)
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]



# average for the summer for each of the grid points involved
def ensemble_metric (generator, metric, interval_number):

    if metric == 'RX1day':
        prec_summer_ensemble = []
        significance_data = []

        for dataset in generator():
            print('new dataset loading')
            prec_summer = []
            if interval_number == 1:
                int_start = 0
                int_end = 15
            else:
                int_start = 26
                int_end = 41
            for i in range(int_start, int_end):
                values = np.max(dataset[(151+(365*i)):(242+(365*i)),:,:], axis=0)
                prec_summer.append(values)
                significance_data.append(values)
            prec_summer_ensemble.append(np.mean(prec_summer, axis=0)) #this will have the highest temperature for each ensemble
            gc.collect() # <= this is a bit cheeky


    if metric == 'RX5day': # 5 day maximum average
        prec_summer_ensemble = []
        significance_data = []
        for dataset in generator(): #loads one ensemble at a time
            print('new dataset loading')
            #count = 0
            prec_summer = []
            yearly_5_day = [] # will produce array of size (15, 96, 144)
            if interval_number == 1:
                int_start = 0
                int_end = 15
            else:
                int_start = 26
                int_end = 41
            for i in range(int_start, int_end):
                values = dataset[(151+(365*i)):(242+(365*i)),:,:] # shape (91, 96, 144); precipitation for each summer
                values_5_day = [] # will produce array of size (86,96,144)
                for i in range(91-5):
                    values_5_day.append(np.sum(values[i:i+5,:,:], axis=0)) # loops over all the elements in 1 summer
                values_5_day = np.array(values_5_day)
                values_5_day = np.reshape(values_5_day,(86,96*144))
                values_5_day = np.reshape(values_5_day,(86,96,144))
                yearly_5_day.append(np.max(values_5_day, axis=0))
                print('new year')

            yearly_5_day = np.reshape(yearly_5_day,(15,96*144))
            yearly_5_day = np.reshape(yearly_5_day,(15,96,144))
            significance_data.append(yearly_5_day)
            prec_summer_ensemble.append(np.mean(yearly_5_day, axis=0))

            gc.collect()

    if metric == 'R95P': # 95% percentile
        prec_summer_ensemble = []
        significance_data = []
        for dataset in generator():
            print('new dataset loading')
            prec_summer = []

            if interval_number == 1:
                int_start = 0
                int_end = 15
            else:
                int_start = 26
                int_end = 41
            for i in range(int_start, int_end):
                percentile = np.percentile(dataset[(151+(365*i)):(242+(365*i)),:,:], 95, axis=0) # calculates the 95th percentile for each grid
                value = dataset[(151+(365*i)):(242+(365*i)),:,:]
                perc_value = []
                for j in range (96): # loops over every gridpoint first
                    for k in range (144):
                        perc_91_days = []
                        for l in range (91): # calculates the percentiles for all the summer days in theat grid
                            if value[l,j,k] > percentile[j,k]:
                               perc_91_days.append(value[l,j,k]) # appends value if above 95th percentile
                        perc_91_days = np.sum(perc_91_days) # cumulative sum of all the values above the 95th percentile
                        perc_value.append(perc_91_days)

                perc_value = np.array(perc_value)
                perc_value = np.reshape(perc_value, (96*144))
                perc_value = np.reshape(perc_value, (96, 144))
                prec_summer.append(perc_value)
            prec_summer = np.array(prec_summer)
            prec_summer = np.reshape(prec_summer,(15,96*144))
            prec_summer = np.reshape(prec_summer,(15,96,144))
            significance_data.append(prec_summer) # appends the percentile value for each summer
            prec_summer_ensemble.append(np.mean(prec_summer, axis=0)) #this will have the mean percentile for each ensemble
            gc.collect() # <= this is a bit cheeky
    if metric == 'R99P': # 99% percentile
        prec_summer_ensemble = []
        significance_data = []
        for dataset in generator():
            print('new dataset loading')
            prec_summer = []

            if interval_number == 1:
                int_start = 0
                int_end = 15
            else:
                int_start = 26
                int_end = 41
            for i in range(int_start, int_end):
                percentile = np.percentile(dataset[(151+(365*i)):(242+(365*i)),:,:], 99, axis=0) # calculates the 95th percentile for each grid
                value = dataset[(151+(365*i)):(242+(365*i)),:,:]
                perc_value = []
                for j in range (96): # loops over every gridpoint first
                    for k in range (144):
                        perc_91_days = []
                        for l in range (91): # calculates the percentiles for all the summer days in theat grid
                            if value[l,j,k] > percentile[j,k]:
                               perc_91_days.append(value[l,j,k]) # appends value if above 99th percentile
                        perc_91_days = np.sum(perc_91_days) # cumulative sum of all the values above the 99th percentile
                        perc_value.append(perc_91_days)

                perc_value = np.array(perc_value)
                perc_value = np.reshape(perc_value, (96*144))
                perc_value = np.reshape(perc_value, (96, 144))
                prec_summer.append(perc_value)

            prec_summer = np.array(prec_summer)
            prec_summer = np.reshape(prec_summer,(15,96*144))
            prec_summer = np.reshape(prec_summer,(15,96,144))
            significance_data.append(prec_summer) # appends the percentile value for each summer
            prec_summer_ensemble.append(np.mean(prec_summer, axis=0)) #this will have the mean percentile for each ensemble
            gc.collect() # <= this is a bit cheeky

    if metric == 'CDD': # number of consecutive das with precipitation less than 1 mm per day
        prec_summer_ensemble = []
        significance_data = []
        for dataset in generator():
            print('new dataset loading')
            prec_summer = []

            if interval_number == 1:
                int_start = 0
                int_end = 15
            else:
                int_start = 26
                int_end = 41
            for i in range(int_start, int_end):
                value = dataset[(151+(365*i)):(242+(365*i)),:,:]
                value = value*(86400*1000)
                max_number_cdd_days = []
                for j in range (96): # loops over every gridpoint first
                    for k in range (144):
                        number_cdd_days_summer = 0
                        cdd_summer_values = []
                        for l in range (91): # calculates the percentiles for all the summer days in that grid point
                            initial_value = 0
                            if value[l,j,k] < 1 and (value[l,j,k] > initial_value or value[l,j,k] < initial_value or value[l,j,k] == initial_value):
                               initial_value = value[l,j,k]
                               number_cdd_days_summer += 1
                               cdd_summer_values.append(number_cdd_days_summer)
                        max_number_cdd_days.append(number_cdd_days_summer)

                max_number_cdd_days = np.array(max_number_cdd_days)
                max_number_cdd_days = np.reshape(max_number_cdd_days, (96*144))
                max_number_cdd_days = np.reshape(max_number_cdd_days, (96, 144))
                prec_summer.append(max_number_cdd_days)

            prec_summer = np.array(prec_summer)
            prec_summer = np.reshape(prec_summer,(15,96*144))
            prec_summer = np.reshape(prec_summer,(15,96,144))
            significance_data.append(prec_summer) # appends the percentile value for each summer
            prec_summer_ensemble.append(np.mean(prec_summer, axis=0)) #this will have the mean percentile for each ensemble
            gc.collect()

    total_significance_data = np.array(significance_data)
    total_significance_data = np.reshape(total_significance_data, (15*5, 96*144))
    total_significance_data = np.reshape(total_significance_data, (15*5, 96,144))
    prec_summer_ensemble = np.array(prec_summer_ensemble)
    prec_summer_ensemble = np.reshape(prec_summer_ensemble, (5,96,144))
    prec_summer_ensemble_avg = np.mean(prec_summer_ensemble, axis = 0)
    return prec_summer_ensemble_avg, total_significance_data # need to multiply here by *(86400*1000) unless CDD

metric = input('Metric to be used: ')

#===============FixedEASO2=======================

# data for the first time interval (1960-1974) for FixedEASO2 simulations
print('starting ensemble_FixedEASO2_6074 max 1 day{}'.format(metric))
ensemble_FixedEASO2_6074, significance_FixedEASO2_6074 = ensemble_metric(get_next_dataset_FixedEASO2, metric,1)
ensemble_FixedEASO2_6074 = np.reshape(ensemble_FixedEASO2_6074, (96,144))
print('finished ensemble_FixedEASO2_6074 first interval')

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, masked_array*ensemble_FixedEASO2_6074, ensemble_FixedEASO2_6074, ensemble_FixedEASO2_6074, '{} AllForcing (mm)'.format(metric), diff=False, reg=False, significance=False)
plt.savefig('{}_6074_FixedEASO2.eps'.format(metric), bbox_inches='tight')



# data for the second time interval (1986-2000) for FixedEASO2 simulations
print('starting ensemble_FixedEASO2_8600 max 1 day')
ensemble_FixedEASO2_8600, significance_FixedEASO2_8600 = ensemble_metric(get_next_dataset_FixedEASO2, metric,0)
ensemble_FixedEASO2_8600 = np.reshape(ensemble_FixedEASO2_8600, (96,144))
print('finished ensemble_FixedEASO2_8600 max 1 day')

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, masked_array*ensemble_FixedEASO2_8600,ensemble_FixedEASO2_6074, ensemble_FixedEASO2_6074,  '{} AllForcing (mm)'.format(metric), diff=False, reg=False, significance=False)
plt.savefig('{}_8600_FixedEASO2.eps'.format(metric), bbox_inches='tight')


#=========================AllForcing=======================
# data for the first time interval (1960-1974) for AllForcing simulations
print('starting ensemble_AF_dry_6074 max 1 day')
ensemble_AF_6074, significance_AF_6074 = ensemble_metric(get_next_dataset_AF, metric,1)
print(np.shape(ensemble_AF_6074))
ensemble_AF_6074 = np.reshape(ensemble_AF_6074, (96,144))
print('finished ensemble_AF_6074 first interval')

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, masked_array*ensemble_AF_6074, ensemble_FixedEASO2_6074, ensemble_FixedEASO2_6074, '{} AllForcing (mm) '.format(metric), diff=False, reg=False, significance=False)
plt.savefig('{}_6074_AF.eps'.format(metric), bbox_inches='tight')

# data for the second time interval (1986-2000) for AllForcing simulations
print('starting ensemble_AF_8600 max 1 day')
ensemble_AF_8600, significance_AF_8600 = ensemble_metric(get_next_dataset_AF, metric,0)
ensemble_AF_8600 = np.reshape(ensemble_AF_8600, (96,144))
print('finished ensemble_AF_8600 max 1 day')

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, masked_array*ensemble_AF_8600, ensemble_FixedEASO2_6074, ensemble_FixedEASO2_6074, '{} AllForcing (mm/day) '.format(metric), diff=False, reg=False, significance=False)
plt.savefig('{}_8600_AF.eps'.format(metric), bbox_inches='tight')


#========== plot difference between time periods for AF and FixedEASO2 =============
print('Calculating time period differences')
AF_8600_6074 = ensemble_AF_8600 - ensemble_AF_6074
FixedEASO2_8600_6074 = ensemble_FixedEASO2_8600 - ensemble_FixedEASO2_6074

print('Calculating experiment differences with time differences')
AF_FixedEASO2_time_diff = AF_8600_6074 - FixedEASO2_8600_6074


print('Calculting experiment differences for 2 periods')
AF_FixedEASO2_6074= ensemble_AF_6074 - ensemble_FixedEASO2_6074

AF_FixedEASO2_8600 = ensemble_AF_8600 - ensemble_FixedEASO2_8600




#===========calculate the 95% significance for the two experiments=================


def significance_test(data_1,data_2):
    mannwhitney_data = []
    for i in range(96):
        for j in range (144):

            mannwhitney_grid = [] # will have 100 values for the mannwhitney test for each data point
            for k in range (10): # bootstrap
                boot_data_1 = resample(data_1[:,i,j], replace=True, n_samples=np.shape(data_1)[0]-1, random_state=1)
                boot_data_2 = resample(data_2[:,i,j], replace=True, n_samples=np.shape(data_2)[0]-1, random_state=1)
                if (boot_data_1 == boot_data_2).all():
                    continue
                u,p= mannwhitneyu(boot_data_1, boot_data_2)
                mannwhitney_grid.append(p) # for a two tailed test
            if len(mannwhitney_grid) == 0:
                mannwhitney_data.append(1) # this makes sure it's not significant
            else:
                mannwhitney_data.append(np.percentile(mannwhitney_grid, 95)) # 95 percentile of the mannwhitney p values
    mannwhitney_data = np.array(mannwhitney_data)
    mannwhitney_data = np.reshape(mannwhitney_data, (96,144))
    return mannwhitney_data

def significance_test_difference(data_1, data_2, data_3, data_4):
    mannwhitney_data_a = []
    mannwhitney_data_b = []
    mannwhitney_data_diff = []
    for i in range(96):
        for j in range (144):
            mannwhitney_grid_diff = []
            mannwhitney_grid_a = [] # will have 100 values for the mannwhitney test for each data point
            mannwhitney_grid_b = []
            for k in range (5): # bootstrap
                boot_data_1 = resample(data_1[:,i,j], replace=True, n_samples=np.shape(data_1)[0]-1, random_state=1)
                boot_data_2 = resample(data_2[:,i,j], replace=True, n_samples=np.shape(data_2)[0]-1, random_state=1)
                diff_boot_data_a = boot_data_1-boot_data_2 #difference for one time period between AF and Fixed
                boot_data_3 = resample(data_3[:,i,j], replace=True, n_samples=np.shape(data_3)[0]-1, random_state=1)
                boot_data_4 = resample(data_4[:,i,j], replace=True, n_samples=np.shape(data_4)[0]-1, random_state=1)
                diff_boot_data_b = boot_data_3-boot_data_4
                if ((boot_data_1 == boot_data_2).all()) or ((boot_data_3 == boot_data_4).all()) or ((diff_boot_data_a == diff_boot_data_b).all()):
                    continue
                u_a,p_a = mannwhitneyu(boot_data_1, boot_data_2)
                u_b, p_b = mannwhitneyu(boot_data_3, boot_data_4)
                mannwhitney_grid_a.append(p_a) # for a two tailed test
                mannwhitney_grid_b.append(p_b)
                u_diff, p_diff = mannwhitneyu(diff_boot_data_a, diff_boot_data_b)
                mannwhitney_grid_diff.append(p_diff)
            if (len(mannwhitney_grid_a) == 0):
                mannwhitney_data_a.append(1) # this makes sure it's not significant
            else:
                mannwhitney_data_a.append(np.percentile(mannwhitney_grid_a, 95)) # 95 percentile of the mannwhitney p values
            if (len(mannwhitney_grid_b) == 0):
                mannwhitney_data_b.append(1)
            else:
                mannwhitney_data_b.append(np.percentile(mannwhitney_grid_b, 95))
            if (len(mannwhitney_grid_diff) == 0):
                mannwhitney_data_diff.append(1)
            else:
                mannwhitney_data_diff.append(np.percentile(mannwhitney_grid_diff, 95))

    mannwhitney_data_a = np.array(mannwhitney_data_a)
    mannwhitney_data_a = np.reshape(mannwhitney_data_a, (96,144))

    mannwhitney_data_b = np.array(mannwhitney_data_b)
    mannwhitney_data_b = np.reshape(mannwhitney_data_b, (96,144))

    mannwhitney_data_diff = np.array(mannwhitney_data_diff)
    mannwhitney_data_diff = np.reshape(mannwhitney_data_diff, (96,144))
    return mannwhitney_data_a, mannwhitney_data_b, mannwhitney_data_diff


#============convert back to original latitudes and longitues========================

def convert_grid(lon,lat):
    lon_original = (lon*2.5)
    lat_original = (lat*1.875)-90
    return lon_original, lat_original


def convert_significance_values(data):
    lon_array = []
    lat_array = []
    for i in range(96):
        for j in range(144):
            if data[i,j] == 1:
                lon, lat = convert_grid(j,i)
                lon_array.append(lon)
                lat_array.append(lat)
    return lon_array, lat_array




#======plot the areas with over 95% significance ================

# All Forcing
mannwhitney_data_AF = significance_test(significance_AF_8600, significance_AF_6074)
threshold_data_AF = mannwhitney_data_AF < 0.05 # the null hypothesis is rejected - 2 tailed test
binary_data_AF = threshold_data_AF.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise.

longitude_AF, latitude_AF = convert_significance_values(binary_data_AF)

mannwhitney_data_6074, mannwhitney_data_8600, mannwhitney_data_diff = significance_test_difference(significance_AF_6074, significance_FixedEASO2_6074,significance_AF_8600, significance_FixedEASO2_8600)


# 8600 diff
threshold_data_8600 = mannwhitney_data_8600 < 0.05 # the null hypothesis is rejected - 2 tailed test
binary_data_8600 = threshold_data_8600.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise.

longitude_8600, latitude_8600 = convert_significance_values(binary_data_8600)

# 6074 diff
threshold_data_6074 = mannwhitney_data_6074 < 0.05 # the null hypothesis is rejected - 2 tailed test
binary_data_6074 = threshold_data_6074.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise.

longitude_6074, latitude_6074 = convert_significance_values(binary_data_6074)

# Total Difference
threshold_data_diff = mannwhitney_data_diff < 0.05 # the null hypothesis is rejected - 2 tailed test
binary_data_diff = threshold_data_diff.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise.

longitude_diff, latitude_diff = convert_significance_values(binary_data_diff)

# plots the graphs of the metric and the significance
fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, AF_8600_6074, longitude_AF, latitude_AF, '{} precipitation difference 1960-74 and 1986-2000 AF'.format(metric), diff=True, reg=False, significance=True)
plt.savefig('{}_AF_8600_6074_prec_sig.eps'.format(metric), bbox_inches = 'tight')

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, FixedEASO2_8600_6074,longitude_8600, latitude_8600,  '{} precipitation difference 1960-74 and 1986-2000 FixedEASO2'.format(metric), diff=True, reg=False, significance=False)
plt.savefig('{}_FixedEASO2_8600_6074_prec.eps'.format(metric), bbox_inches = 'tight')

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, AF_FixedEASO2_6074, longitude_6074, latitude_6074,'{} precipitation difference 1960-74 AF-FixedEASO2'.format(metric), diff=True, reg=False, significance=True)
plt.savefig('{}_AF_FixedEASO2_6074_prec_significance.eps'.format(metric), bbox_inches = 'tight')

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, AF_FixedEASO2_8600,longitude_8600, latitude_8600, '{} precipitation difference 1986-2000 AF-FixedEASO2'.format(metric), diff=True, reg=False, significance=True)
plt.savefig('{}_AF_FixedEASO2_8600_prec_significance.eps'.format(metric), bbox_inches = 'tight')

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, AF_FixedEASO2_time_diff, longitude_diff, latitude_diff, '{} precipitation difference 1960-74 and 1986-2000 AF-FixedEASO2'.format(metric), diff=True, reg=False, significance=True)
plt.savefig('{}_AF_FixedEASO2_time_diff_prec_sig.eps'.format(metric), bbox_inches = 'tight')








