import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import glob
import netCDF4
from netCDF4 import Dataset
from sklearn.linear_model import LinearRegression
from itertools import chain
import gc
import pickle
from scipy.stats import mannwhitneyu
from sklearn.utils import resample

'''
Creates climatology maps for differences and linear regression
'''
def plot_field(ax, data, lon_significance, lat_significance, title, diff=False, reg=False, significance = False):
    if diff == False and reg == False: # Climatology only seasonal
        cmap = plt.cm.Blues
        extend = 'max'
        clevs = np.arange(0,21,2)

    elif diff == False and reg == True: # Regression only
        cmap = plt.cm.RdBu
        extend = 'both'
        clevs = np.array([-0.9,-0.4,-0.25,-0.2,-0.15,-0.1,0.1,0.15,0.2, 0.25, 0.4,0.9])
    elif diff == True and reg == True: # Difference in regression
        cmap = plt.cm.seismic
        extend = 'both'
        clevs = np.array([-0.9,-0.4,-0.25,-0.2,-0.15,-0.1,0.1,0.15,0.2, 0.25, 0.4,0.9])

    else:
        cmap = plt.cm.RdBu # Difference in climatology
        extend = 'both'
        clevs = np.arange(-0.9,1.0,0.1)

    #add coastlines
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.8, alpha=1)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.8, alpha=1)
    #define x ticks locations
    xticks = np.arange(-180,180, 20)
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
    #data_smooth = ndimage.gaussian_filter(data, sigma=1, order=0,smoothing the data
    plot = ax.contourf(lons, lats, data, cmap=cmap, extend=extend, levels=clevs)
    if significance == True:
        ax.scatter(lon_significance,lat_significance, c='black', marker="o", s=5, alpha = 0.7)
    # sets the region shown in the map
    ax.set_extent([60,150,-5,45])
    #ax.set_title(title, fontsize=14.)
    cbar = plt.colorbar(plot, orientation = 'horizontal', pad = 0.07)
    cbar.ax.tick_params(labelsize=14.)
    #cbar.set_label('precipitation (mm/day)')

'''
def plot_prec(ax, data, title, diff=False, reg=False):
    if diff == False and reg == False: # Climatology only seasonal
        cmap = plt.cm.Blues
        extend = 'max'
        clevs = np.arange(0,21,2)

    elif diff == False and reg == True: # Regression only
        cmap = plt.cm.RdBu
        extend = 'both'
        clevs = np.array([-0.9,-0.4,-0.25,-0.2,-0.15,-0.1,0.1,0.15,0.2, 0.25, 0.4,0.9])
    elif diff == True and reg == True: # Difference in regression
        cmap = plt.cm.RdBu
        extend = 'both'
        clevs = np.array([-0.9,-0.4,-0.25,-0.2,-0.15,-0.1,0.1,0.15,0.2, 0.25, 0.4,0.9])

    else:
        cmap = plt.cm.RdBu # Difference in climatology
        extend = 'both'
        clevs = np.arange(-0.9, 1.0, 0.2)

    #add coastlines
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.8, alpha=1)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.8, alpha=1)
    #define x ticks locations
    xticks = np.arange(-160,170, 10)
    yticks = np.arange(-80,90,10)
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
    # sets the region shown in the map
    ax.set_extent([60,170,-10,55])
    ax.set_title(title, fontsize=14.)
    cbar = plt.colorbar(plot, orientation = 'horizontal', shrink=0.5)
    cbar.set_label('precipitation (mm/day)')
'''
def linear_regression(time, precipitation_array):
    regression_values = []
    for i in range (96):
        for j in range (144):
            regression = LinearRegression()
            y_values= np.reshape(precipitation_array[:, i,j], (-1,1))
            regression.fit(time, y_values)
            regression_values.append(regression.coef_)
    regression_values = np.array(regression_values)
    regression_values = np.reshape(regression_values, (96,144))
    return regression_values

# get the data from each of the ensembles from 1990 to 2019 from the 2 models, AF and FixedEASO2

for ensemble in range(1,6):
    gc.collect()

    ensemble_files =sorted(glob.glob('../AllForcing/r' + str(ensemble) + '/PREC*.nc'))
    data = [Dataset(ensemble_files[i]) for i in range(len(ensemble_files))]
    if ensemble != 5:
        prec50 = data[1].variables['PRECC'][5:] + data[10].variables['PRECL'][5:]
    else:
        prec50 = data[1].variables['PRECC'][:] + data[10].variables['PRECL'][:]
    for i in range (5):
        prec50 = np.delete(prec50[:,:,:], 1820, axis=0)
    prec60 = data[2].variables['PRECC'][:] + data[11].variables['PRECL'][:]
    prec70 = data[3].variables['PRECC'][:] + data[12].variables['PRECL'][:]
    prec80 = data[4].variables['PRECC'][:] + data[13].variables['PRECL'][:]
    prec90 = data[5].variables['PRECC'][:] + data[14].variables['PRECL'][:]
    prec00 = data[7].variables['PRECC'][:] + data[16].variables['PRECL'][:]
    prec10 = data[8].variables['PRECC'][:] + data[17].variables['PRECL'][:]

    if ensemble==1:
        #import longitude and latitude files only once
        lats = data[0].variables['lat'][:]
        lons = data[0].variables['lon'][:]
        prec_AF = np.zeros((len(prec50)+len(prec60)+len(prec70)+len(prec80)+len(prec90)+ len(prec00)+len(prec10), len(lats), len(lons)))

    prec_AF += np.concatenate((prec50, prec60, prec70, prec80, prec90, prec00, prec10))

# converts units and gets ensemble average
prec_AF = (prec_AF*(84600*1000))/5
print(np.shape(prec_AF))


for ensemble in range(1,6):
    gc.collect()
    #files for FixedEASO2
    ensemble_files = sorted(glob.glob('../FixedEASO2/r' + str(ensemble) + '/PREC*.nc'))
    data = [Dataset(ensemble_files[i]) for i in range(len(ensemble_files))]
    if ensemble == 5 :
        prec50 = data[1].variables['PRECC'][:] + data[10].variables['PRECL'][:]
        for i in range (5):
            prec50 = np.delete(prec50[:,:,:], 1820, axis=0)
    else:
        prec50 = data[1].variables['PRECC'][:] + data[10].variables['PRECL'][:]
    prec60 = data[2].variables['PRECC'][:] + data[11].variables['PRECL'][:]
    prec70 = data[3].variables['PRECC'][:] + data[12].variables['PRECL'][:]
    prec80 = data[4].variables['PRECC'][:] + data[13].variables['PRECL'][:]
    prec90 = data[5].variables['PRECC'][:] + data[14].variables['PRECL'][:]
    prec00 = data[7].variables['PRECC'][:] + data[16].variables['PRECL'][:]
    prec10 = data[8].variables['PRECC'][:] + data[17].variables['PRECL'][:]
    if ensemble == 1:
        lats = data[0].variables['lat'][:]
        lons = data[0].variables['lon'][:]
        prec_FixedEASO2 = np.zeros((len(prec50)+len(prec60)+len(prec70)+len(prec80)+len(prec90)+ len(prec00)+len(prec10), len(lats), len(lons)))

    prec_FixedEASO2 += np.concatenate((prec50, prec60, prec70, prec80, prec90, prec00, prec10))

prec_FixedEASO2 = (prec_FixedEASO2*(84600*1000))/5
print(np.shape(prec_FixedEASO2))


'''
 summer for first time interval All Forcing
'''

precip_avg_AF_summer_1 = np.zeros((15,96,144))


# average for the summer for each of the grid points involved
count=0
for i in range(9,24):
    precip_avg_AF_summer_1[count] = np.average(prec_AF[(151+(365*i)):(242+(365*i)),:,:], axis=0)
    count+=1

# average over all years
significance_data_AF_1 = precip_avg_AF_summer_1
precip_avg_AF_summer_1 = np.average(precip_avg_AF_summer_1, axis=0)


'''
 summer for second time interval All Forcing
'''
precip_avg_AF_summer_2 = np.zeros((15,96,144))


count = 0
for i in np.arange(35,50):
    precip_avg_AF_summer_2[count] = np.average(prec_AF[(151+(365*i)):(242+(365*i)),:,:], axis=0)
    count+=1

# average to get all spatial points over all years
significance_data_AF_2 = precip_avg_AF_summer_2
precip_avg_AF_summer_2 = np.average(precip_avg_AF_summer_2, axis=0)

'''
summer averages first time interval FixedEASO2
'''

precip_avg_Fi_summer_1 = np.zeros((15,96,144))

count=0
for i in range(9,24):
    precip_avg_Fi_summer_1[count] = np.average(prec_FixedEASO2[(151+(365*i)):(242+(365*i)),:,:], axis=0)
    count+=1

# average to get all spatial points over all years
significance_data_Fi_1 = precip_avg_Fi_summer_1 # no average over the time axis
precip_avg_Fi_summer_1 = np.average(precip_avg_Fi_summer_1, axis=0)

'''
 summer for second time interval FixedEASO2
'''
precip_avg_Fi_summer_2 = np.zeros((15,96,144))


count = 0
for i in np.arange(35,50):
    precip_avg_Fi_summer_2[count] = np.average(prec_FixedEASO2[(151+(365*i)):(242+(365*i)),:,:], axis=0)
    count+=1
# average to get all spatial points over all years
significance_data_Fi_2 = precip_avg_Fi_summer_2 # no average over the time axis
precip_avg_Fi_summer_2 = np.average(precip_avg_Fi_summer_2, axis=0)



diff_AF_summer = precip_avg_AF_summer_2 - precip_avg_AF_summer_1 # have changed from precip_avg_AF_summer_1 - precip_avg_AF_summer_2
diff_Fi_summer = precip_avg_Fi_summer_2 - precip_avg_Fi_summer_1

diff_AF_Fi_8600 = precip_avg_AF_summer_2 - precip_avg_Fi_summer_2
diff_AF_Fi_6074 = precip_avg_AF_summer_1 - precip_avg_Fi_summer_1
'''
fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, diff_AF_Fi_9610, 'summer 8600 difference AF-FixedEASO2', diff=True, reg=False)
plt.savefig('mean_AF_FixedEASO2_8600_prec.png')

fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, diff_AF_summer, 'summer 6074-8600 daily difference AF', diff=True, reg=False)
plt.savefig('daily_AF_6074_8600_prec.png')

fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, diff_Fi_summer, 'summer 6074-8600 daily difference Fi', diff=True, reg=False)
plt.savefig('daily_Fi_6074_8600_prec.png')

fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, diff_AF_Fi_5064, 'summer 6074 difference AF-FixedEASO2', diff=True, reg=False)
plt.savefig('mean_AF_FixedEASO2_6074_prec.png')

fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, diff_AF_summer-diff_Fi_summer, 'summer 6074-8600 daily total difference AF-FixedEASO2', diff=True, reg=False)
plt.savefig('daily_totaldiff_AF_FixedEASO2_6074_8600_prec.png')

'''
def significance_test(data_1, data_2):
    mannwhitney_data = []
    for i in range(96):
        for j in range (144):
            #print(data_1[:,i,j])

            mannwhitney_grid = [] # will have 100 values for the mannwhitney test for each data point
            for k in range (10): # bootstrap
                #print(k) 
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
    print(mannwhitney_data)
    mannwhitney_data = np.reshape(mannwhitney_data, (96,144))
    return mannwhitney_data




def significance_test_difference(data_1, data_2, data_3, data_4):
    mannwhitney_data_a = []
    mannwhitney_data_b = []
    mannwhitney_data_diff = []
    for i in range(96):
        for j in range (144):
            #print(data_1[:,i,j])
            mannwhitney_grid_diff = []
            mannwhitney_grid_a = [] # will have 100 values for the mannwhitney test for each data point
            mannwhitney_grid_b = []
            for k in range (5): # bootstrap
                #print(k) 
                boot_data_1 = resample(data_1[:,i,j], replace=True, n_samples=np.shape(data_1)[0]-1, random_state=1)
                boot_data_2 = resample(data_2[:,i,j], replace=True, n_samples=np.shape(data_2)[0]-1, random_state=1)    
                diff_boot_data_a = boot_data_1-boot_data_2 #difference for one time period between AF and Fixed
                print(np.shape(diff_boot_data_a))
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
    print(mannwhitney_data_a)
    mannwhitney_data_a = np.reshape(mannwhitney_data_a, (96,144))

    mannwhitney_data_b = np.array(mannwhitney_data_b)
    print(mannwhitney_data_b)
    mannwhitney_data_b = np.reshape(mannwhitney_data_b, (96,144))
   
    mannwhitney_data_diff = np.array(mannwhitney_data_diff)
    print(mannwhitney_data_diff)
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
'''

#======calculates areas with over 95% significance ================
mannwhitney_data_AF = significance_test(significance_data_AF_2, significance_data_AF_1)
threshold_data_AF = mannwhitney_data_AF < 0.05 # the null hypothesis is rejected - 2 tailed test 
binary_data_AF = threshold_data_AF.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise. 

longitude_AF, latitude_AF = convert_significance_values(binary_data_AF)

'''
mannwhitney_data_AF = significance_test(significance_data_AF_2, significance_data_AF_1)
threshold_data_AF = mannwhitney_data_AF < 0.05 # the null hypothesis is rejected - 2 tailed test 
binary_data_AF = threshold_data_AF.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise. 

longitude_AF, latitude_AF = convert_significance_values(binary_data_AF)





mannwhitney_data_6074, mannwhitney_data_8600, mannwhitney_data_diff = significance_test_difference(significance_data_AF_1, significance_data_Fi_1,significance_data_AF_2, significance_data_Fi_2)

threshold_data_8600 = mannwhitney_data_8600 < 0.05 # the null hypothesis is rejected - 2 tailed test 
binary_data_8600 = threshold_data_8600.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise. 

longitude_8600, latitude_8600 = convert_significance_values(binary_data_8600)



threshold_data_6074 = mannwhitney_data_6074 < 0.05 # the null hypothesis is rejected - 2 tailed test 
binary_data_6074 = threshold_data_6074.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise. 

longitude_6074, latitude_6074 = convert_significance_values(binary_data_6074)

threshold_data_diff = mannwhitney_data_diff < 0.05 # the null hypothesis is rejected - 2 tailed test 
binary_data_diff = threshold_data_diff.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise. 

longitude_diff, latitude_diff = convert_significance_values(binary_data_diff)



fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_field(ax, diff_AF_summer,longitude_AF, latitude_AF, 'summer mean prec 8600-6074 AF', diff=True, reg=False, significance=True)
plt.savefig('mean_prec_8600_6074_AF_sig.eps', bbox_inches='tight')
print('hello')
plt.clf()

fig = plt.figure(figsize=[7.5,7.5])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_field(ax, diff_AF_Fi_8600-diff_AF_Fi_6074,longitude_diff, latitude_diff, 'summer mean prec 8600-6074 AF-FixedEASO2', diff=True, reg=False, significance=True)
plt.savefig('total_diff_mean_8600_6074_AF_Fi_sig.eps', bbox_inches='tight')
print('hello')
plt.clf()


'''
#===========calculate the 95% significance for the two experiments=================


def significance_test(data_1,data_2):
    mannwhitney_data = []
    for i in range(96):
        for j in range (144):
            #print(data_1[:,i,j])
            print(np.shape(data_1))
            mannwhitney_grid = [] # will have 1000 values for the mannwhitney test for each data point
            for k in range (100): # bootstrap 
                boot_data_1 = resample(data_1[:,i,j], replace=True, n_samples=np.shape(data_1)[0]-1, random_state=1)
                boot_data_2 = resample(data_2[:,i,j], replace=True, n_samples=np.shape(data_2)[0]-1, random_state=1)
                u,p= mannwhitneyu(boot_data_1, boot_data_2)
                mannwhitney_grid.append(p) # for a two tailed test
            mannwhitney_data.append(np.percentile(mannwhitney_grid, 95)) # 95 percentile of the mannwhitney p values 
    mannwhitney_data = np.array(mannwhitney_data)
    print(mannwhitney_data)
    mannwhitney_data = np.reshape(mannwhitney_data, (96,144))
    return mannwhitney_data


#mannwhitney_data1 = significance_test(significance_data_AF_2, significance_data_Fi_2)

#threshold_data1 = mannwhitney_data1 < 0.05 # the null hypothesis is rejected
#binary_data1 = threshold_data1.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise. 


#======plot the areas with over 95% significance ================
fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, binary_data1, 'AF_FixedEASO2_9610_mean_significance_bootstrap', diff=False, reg=True)
plt.savefig('AF_FixedEASO2_9610_mean_significance_bootstrap.png')



mannwhitney_data2 = significance_test(significance_data_AF_1, significance_data_Fi_1)

threshold_data2 = mannwhitney_data2 < 0.05 # the null hypothesis is rejected
binary_data2 = threshold_data2.astype(int) # converts to binary values: 1 if null hypothesis is rejected. 0 otherwise. 


#======plot the areas with over 95% significance ================
fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
plot_prec(ax, binary_data2, 'AF_FixedEASO2_5064_mean_significance_bootstrap', diff=False, reg=True)
plt.savefig('AF_FixedEASO2_5064_mean_significance_bootstrap.png')
'''

"""
Calculates a least square linear fit through the selected data
"""
'''

# times goes from the midpoint of each of the time intervals selected
time = np.arange(1960,2001,1)
time = np.reshape(time, (-1,1))

count = 0
# first element of array is the number of years
precip_avg_AF_summer = np.zeros((41,96,144))

for i in np.arange(9,50):
    precip_avg_AF_summer[count] = np.average(prec_AF[(151+(365*i)):(242+(365*i)),:,:], axis=0)

    count+= 1

precip_avg_Fi_summer = np.zeros((41,96,144))


count = 0
for i in np.arange(9,50):
    precip_avg_Fi_summer[count] = np.average(prec_FixedEASO2[(151+(365*i)):(242+(365*i)),:,:], axis=0)

    count+= 1

#calculates regression 
regression_values_summer_AF = linear_regression(time, precip_avg_AF_summer)

regression_values_summer_Fi = linear_regression(time, precip_avg_Fi_summer)

'''
'''
fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
#plots precipitation for AF
plot_prec(ax, diff_AF_summer, 'summer 5069_8100 difference AF', diff=False, reg=True)
plt.savefig('difference_AF_5069_8100_Asia.png')


fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())
#plots precipitation for FixedEASO2
plot_prec(ax, diff_Fi_summer, 'summer 5069_8100 difference Fixed', diff=False, reg=True)
plt.savefig('difference_Fi_5069_8100_Asia.png')


fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())

# plots difference between AF and FixedEASO2
plot_prec(ax, (diff_AF_summer-diff_Fi_summer), 'summer 5069_8100 difference AF-Fixed', diff=True, reg=True)
plt.savefig('total_difference_5069_8100_Asia.png')

'''
'''
#plots precipitation for AF
fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())

plot_prec(ax, regression_values_summer_AF*41, 'daily summer 1960-2000 regression AF', diff=False, reg=True)
plt.savefig('daily_regression_AF_1960-2000_Asia.png')


#plots precipitation for FixedEASO2
fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())

plot_prec(ax, regression_values_summer_Fi*41, 'daily summer 1960-2000 regression Fixed', diff=False, reg=True)
plt.savefig('daily_regression_Fi_1960-2000_Asia.png')


# plots difference between AF and FixedEASO2
fig = plt.figure(figsize=[18,9])
ax = plt.axes(projection=ccrs.PlateCarree())

plot_prec(ax, (regression_values_summer_AF*41 -regression_values_summer_Fi*41), 'daily summer 1960-2000 regression AF-Fixed', diff=True, reg=True)
plt.savefig('daily_total_regression_1960-2000_Asia.png')


'''
