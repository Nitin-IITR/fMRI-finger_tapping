import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import nilearn
from nilearn import plotting
from nilearn import image
from nilearn import datasets
import nibabel as nib
import matplotlib.image as mpimg
import h5py
import imageio
import scipy.misc as spmi
import nibabel as nib
from nilearn.image import get_data
import os
import random
from random import seed
from nideconv.utils import roi
from itertools import chain


# Locate the data of the first subject
sub=[1,2,3,4,5,6,7,8,9,10,11,12,13]
ses=[1,2,3,4]

main_folder= 'D:\Finger New fmri'
func=[]

for subjects in sub:
    subject=str("{:02d}".format(subjects))
    
    for sessions in ses:
        session=str(sessions)
        
        func.append(main_folder+'\sub-'+subject +'\\func\swarsub-'+subject+'_task-bilateralfingertapping_echo-'+session +'_bold.nii')
#        func = r'D:\Finger New fmri\sub-01\func\swarsub-01_task-bilateralfingertapping_echo-1_bold.nii'



# Use the cortical Harvard-Oxford atlas
atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
atlas_harvard_oxford.labels=['ROI']
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ROI\tap_combine.nii' #1) 86.8% ACCURACY
atlas_harvard_oxford.maps=r'D:\Finger New fmri\ROI\Taproi.nii'  #2) 83.5% ACCURACY
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ROI\Taproi2.nii' #3) 87% Accuracy
## 2) and 3) as feature the net accuracy is 87.5%


#plotting.plot_prob_atlas(atlas_harvard_oxford.maps)

# ROI time series
ts=[]
for bold in func:  
    ts.append(roi.extract_timecourse_from_nii(atlas_harvard_oxford, bold, t_r=4))




###########################################################################################
###########################################################################################
##############    CONSIDERING THE ONSET POINTS IN RESTING     #############################
#
#                             BEST ONE
#

# Extacting tapping time series from each subject and storing it into a list 'tap_series_all'
tap_roi_all=[]

for k in range(len(ts)):
    ts_frame=ts[k]
#    ts_frame.index=ts_frame.index+4
    #plt.plot(ts_frame['MyROI'])
    
    tap_roi_each=[]
    for j in range(4):
        a=range(44+80*j,84+80*j,4)
        tap_roi_each.append(ts_frame.loc[[i for i in ts_frame.index if i in a]].values)
    
    tap_roi_each1=list(chain.from_iterable(tap_roi_each))
    tap_roi_each1=np.array(tap_roi_each1)
    
    tap_roi_all.append(tap_roi_each1)


tap_roi_all=np.array(tap_roi_all)
tap_roi_all12=tap_roi_all.ravel()               
        
# Extacting resting time series from each subject and storing it into a list 'rest_series_all'
rest_roi_all=[]

for k in range(len(ts)):
    ts_frame=ts[k]
#    ts_frame.index=ts_frame.index+4
    #plt.plot(ts_frame['MyROI'])
    
    rest_roi_each=[]
    for j in range(5):
        a=range(4+80*j,44+80*j,4)
        rest_roi_each.append(ts_frame.loc[[i for i in ts_frame.index if i in a]].values)
    
#    rest_series_each1=np.array(rest_series_each)
#    rest_series_each1=rest_series_each1.reshape(49,1)
#    
    
    rest_roi_each1=list(chain.from_iterable(rest_roi_each))
    rest_roi_each1=np.array(rest_roi_each1)
    
    rest_roi_all.append(rest_roi_each1)   
    
    
rest_roi_all=np.array(rest_roi_all)
rest_roi_all12=rest_roi_all.ravel()     

    
#tap_series_all11=tap_series_all
#rest_series_all11=rest_series_all
        
plt.plot(rest_roi_all11)        
plt.plot(tap_roi_all11)     


# BELOW ONE WILL MAKE ACCURACY WORSE BECAUSE IT GAVE RISE TO OUTLIERS AND AS MANY NUMBERS ARE IN BETWEEN 
# 0 AND 1, SO IT GONNA MAKE BOTH CLASSES CLOSE TO ZERO AND HARDER TO SEPARATE
plt.plot( np.power(rest_series_all,3))   
plt.plot( np.power(tap_series_all,3)) 
        
        
###############################################################################
################  EEG TYPE FEATURE EXTRACTION (Ultimate RESULTS) 98.5 % accuracy+
############## Use features which make difference in positive and negative


tap_roi_com=pd.DataFrame( {'col1':tap_roi_all11, 'col2':tap_roi_all12})
rest_roi_com=pd.DataFrame({'col1':rest_roi_all11, 'col2':rest_roi_all12})


rest_roi_com1=pd.DataFrame([])

for i in range(53):    
    rest_roi_com1=pd.concat([rest_roi_com1,rest_roi_com[49*i:(49*i+40)] ])
    
rest_roi_com=rest_roi_com1.reset_index(drop=True)

#com_frame = tap_roi_com.append(rest_roi_com)
#
#com_frame= com_frame.reset_index(drop=True)

#############################################
points_at_time=40
n_regions_extracted=2

def feature_extraction(method):
    a=[]
    for i in range(n_regions_extracted):
        b=[]
        for j in range(int(len(tap_roi_com)/points_at_time)):  # total length devided by 25(the value used below)
            b.append(method(tap_roi_com.iloc[:,i].values[points_at_time*j:(points_at_time*(j+1))]))
        a.append(b)

    c=pd.concat([pd.DataFrame(a[i]) for i in range(n_regions_extracted)],ignore_index=True,axis=1) 
    
    return c

############################################
points_at_time1=40
n_regions_extracted1=2

def feature_extraction1(method):
    a=[]
    for i in range(n_regions_extracted1):
        b=[]
        for j in range(int(len(rest_roi_com)/points_at_time1)):  # total length devided by 25(the value used below)
            b.append(method(rest_roi_com.iloc[:,i].values[points_at_time1*j:(points_at_time1*(j+1))]))
        a.append(b)

    c=pd.concat([pd.DataFrame(a[i]) for i in range(n_regions_extracted1)],ignore_index=True,axis=1) 
    
    return c


############################################
#HIGHEST ACCURACY: Using INTEGRATION ====>Using only Taproi2.nii' 87% wala mask ####  points_at_time1=40 98.5% :40 points each <======
    

# Integration
# Accuracy 95.68% with points_at_time1=10  :40 points each|| points_at_time1=40 97.54% :40 points each|| 40,49 diff points then 95.06%
# Using only Taproi2.nii' #### Accuracy 95.6% with points_at_time1=10  :40 points each|| ====> points_at_time1=40 98.5% :40 points each <======|| 40,49 diff points then 96.4%
#                                                                                      -------------------------------------------------------
tap_inte = feature_extraction(sp.integrate.simps)
rest_inte= feature_extraction1(sp.integrate.simps)

# Approximate entropy
from entropy import app_entropy
tap_app_ent = feature_extraction(app_entropy)
rest_app_ent= feature_extraction1(app_entropy)

# Sample entropy
import nolds
tap_sam_ent = feature_extraction(nolds.sampen)
rest_sam_ent= feature_extraction1(nolds.sampen)

# Iqr
tap_iqr = feature_extraction(sp.stats.iqr)
rest_iqr= feature_extraction1(sp.stats.iqr)

# Mode
AB_mode = feature_extraction(sp.stats.mode)
AB_mode= AB_mode.iloc[:,[0,2]].values
tap_mode= AB_mode.astype(np.float) 

AB_mode = feature_extraction1(sp.stats.mode)
AB_mode= AB_mode.iloc[:,[0,2]].values
rest_mode= AB_mode.astype(np.float) 


# Mean #
# Accuracy 95.42% with points_at_time1=10  :40 points each|| points_at_time1=40 97.5% :40 points each|| 40,49 diff points then 97.1%
# Using only Taproi2.nii' #### Accuracy 95.75% with points_at_time1=10  :40 points each|| points_at_time1=40 98.2% :40 points each|| 40,49 diff points then 98.3%
# Using only Taproi2.nii' #### BUT BOTH INTEGRATION and MEAN: Accuracy: 98.37% => points_at_time1=40 98.2% :40 points each
import statistics
tap_mean = feature_extraction(statistics.mean)
rest_mean = feature_extraction1(statistics.mean)

# Std
from astropy.stats import mad_std
AB_std = feature_extraction(mad_std)


###########################################################################################        
###########################################################################################        
###########################################################################################

###########################################################################################
###########################################################################################
##############    CONSIDERING THE ONSET POINTS IN TAPPING      #############################

# Extacting tapping time series from each subject and storing it into a list 'tap_series_all'
tap_series_all=[]

for k in range(16):
    ts_frame=ts[k]
#    ts_frame.index=ts_frame.index+4
    #plt.plot(ts_frame['MyROI'])
    
    tap_series_each=[]
    for j in range(4):
        a=range(40+80*j,80+80*j,4)
        tap_series_each.append(ts_frame.loc[[i for i in ts_frame.index if i in a]].values)
    
    tap_series_each1=list(chain.from_iterable(tap_series_each))
    tap_series_each1=np.array(tap_series_each1)
    
    tap_series_all.append(tap_series_each1)


tap_series_all=np.array(tap_series_all)
tap_series_all=tap_series_all.ravel()              
        
# Extacting resting time series from each subject and storing it into a list 'rest_series_all'
rest_series_all=[]

for k in range(16):
    ts_frame=ts[k]
#    ts_frame.index=ts_frame.index+4
    #plt.plot(ts_frame['MyROI'])
    
    rest_series_each=[]
    for j in range(5):
        a=range(80*j,40+80*j,4)
        rest_series_each.append(ts_frame.loc[[i for i in ts_frame.index if i in a]].values)
    
#    rest_series_each1=np.array(rest_series_each)
#    rest_series_each1=rest_series_each1.reshape(49,1)
#    
    
    rest_series_each1=list(chain.from_iterable(rest_series_each))
    rest_series_each1=np.array(rest_series_each1)
    
    rest_series_all.append(rest_series_each1)   
    
    
rest_series_all=np.array(rest_series_all)
rest_series_all=rest_series_all.ravel()
    
        
plt.plot(rest_series_all)        
plt.plot(tap_series_all)     




###########################################################################################
###########################################################################################
##############    WITHOUT CONSIDERING THE ONSET POINTS      #############################


# Extacting tapping time series from each subject and storing it into a list 'tap_series_all'
tap_series_all=[]

for k in range(16):
    ts_frame=ts[k]
#    ts_frame.index=ts_frame.index+4
    #plt.plot(ts_frame['MyROI'])
    
    tap_series_each=[]
    for j in range(4):
        a=range(44+80*j,80+80*j,4)
        tap_series_each.append(ts_frame.loc[[i for i in ts_frame.index if i in a]].values)
    
    tap_series_each1=list(chain.from_iterable(tap_series_each))
    tap_series_each1=np.array(tap_series_each1)
    
    tap_series_all.append(tap_series_each1)


tap_series_all=np.array(tap_series_all)
tap_series_all=tap_series_all.ravel()             
        
# Extacting resting time series from each subject and storing it into a list 'rest_series_all'
rest_series_all=[]

for k in range(16):
    ts_frame=ts[k]
#    ts_frame.index=ts_frame.index+4
    #plt.plot(ts_frame['MyROI'])
    
    rest_series_each=[]
    for j in range(5):
        a=range(4+80*j,40+80*j,4)
        rest_series_each.append(ts_frame.loc[[i for i in ts_frame.index if i in a]].values)
    

    rest_series_each1=list(chain.from_iterable(rest_series_each))
    rest_series_each1=np.array(rest_series_each1)
    
    rest_series_all.append(rest_series_each1)   
    
    
rest_series_all=np.array(rest_series_all)
rest_series_all=rest_series_all.ravel()     
    
        
plt.plot(rest_series_all)        
plt.plot(tap_series_all)        
