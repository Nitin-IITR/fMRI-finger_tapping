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
atlas_harvard_oxford.labels=['ROI1']

#1) Total Extracted regions are 28 and column wise accuracy in decreasing order is =[82.5,82.3,63.1,60.9....]
# and the combined accuracy of first two is 82.7% only
# IMPORTANT: USING ALL 28 FEATURES GIVE 90.4% ACCURACY (SVM)
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ROI\Tap_ICA_3_comp.nii' 


#2) Total Extracted regions are 4 and columwise accuracy is [83.6,57.7,56.6,54.7] 
# and all combined accuracy is 85.71%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ROI\Tap_ICA_2_comp.nii'





############################################################
# NEW ICA COMPONENTS OBTAIN BY WHOLE DATASET ########

#1) ACCURACY WITH ALL FEATURES IS 92.11% and 23 features
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_3_comp.nii' 


#2) ACCURACY WITH ALL FEATURES IS 90.9% and 12 features
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_2_comp.nii' 

#3) ACCURACY WITH ALL FEATURES IS 92.86% and 32 features || Also including ROI 87% wala Taproi2 then give
# 93.1 % accuracy || including Taproi also with it roi 93.1%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_4_comp.nii' 

#4) ACCURACY WITH ALL FEATURES IS 93.2% || Only 87% wala ROI its 93.5% || Including both ROI's 93.6%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_5_comp.nii' 

#5) ACCURACY WITH ALL FEATURES IS 93.5% || Only 87% wala ROI its 93.7% || Including both ROI's 93.72%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_6_comp.nii' 

#7) ACCURACY WITH ALL 47 FEATURES IS 93.55% || Only 87% wala ROI its 93.86% || Including both ROI's 93.86%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_7_comp.nii' 

#8) ACCURACY WITH ALL FEATURES IS 93.97% || Only 87% wala ROI its 94.3% || Including both ROI's 94.33%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_8_comp.nii' 

#8) ACCURACY WITH ALL FEATURES IS 94% || Only 87% wala ROI its 94.24% || Including both ROI's 94.27%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_9_comp.nii' 

#8) ACCURACY WITH ALL FEATURES IS 94.95% || Only 87% wala ROI its  || Including both ROI's 95.14
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_12_comp.nii' 

#8) ACCURACY WITH ALL FEATURES IS 95% || Only 87% wala ROI its  || Including both ROI's 95.03%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_14_comp.nii' 

#8) ACCURACY WITH ALL FEATURES IS 95.3% || Only 87% wala ROI its  || Including both ROI's 95.4%
#atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_20_comp.nii' 


#8) ACCURACY WITH ALL FEATURES IS 95.57% || Only 87% wala ROI its  || Including both ROI's %
# Threshold =2: ACCURACY WITH 183 FEATURES IS %
atlas_harvard_oxford.maps=r'D:\Finger New fmri\ICA\ICA_tap_25_comp.nii' 

#plotting.plot_prob_atlas(atlas_harvard_oxford.maps)

#FURTHER ACCURACY CAN BE INCREASED BY LITTLE BIT BY USING SPHERE DOTS ROI USING NEUROSYNTH OR 2ND LEVEL SPM





# ROI time series
#ts=[]
#for bold in func:  
#    ts.append(roi.extract_timecourse_from_nii(atlas_harvard_oxford, bold, t_r=4))
#


#######################################################################################
#######################  REGION EXTRACTOR  ######################################

################################################################################
# Extract regions from networks
# ------------------------------

# Import Region Extractor algorithm from regions module
# threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
# maps, less the threshold means that more intense non-voxels will be survived.
from nilearn.regions import RegionExtractor

extractor = RegionExtractor(atlas_harvard_oxford.maps, threshold=2,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=1350)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Each region index is stored in index_
regions_index = extractor.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, 3))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)

################################################################################
# Compute correlation coefficients
# ---------------------------------

# First we need to do subjects timeseries signals extraction and then estimating
# correlation matrices on those signals.
# To extract timeseries signals, we call transform() from RegionExtractor object
# onto each subject functional data stored in func_filenames.
# To estimate correlation matrices we import connectome utilities from nilearn
from nilearn.connectome import ConnectivityMeasure

time_series=[]
correlations = []
# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
for filename in func:
    # call transform from RegionExtractor object to extract timeseries signals
    timeseries_each_subject = extractor.transform(filename)
    
    # append time series
    time_series.append(timeseries_each_subject)
    
    # call fit_transform from ConnectivityMeasure object
    correlation = connectome_measure.fit_transform([timeseries_each_subject])
    # saving each subject correlation to correlations
    correlations.append(correlation)

# Mean of all correlations
import numpy as np
mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)

###############################################################################
###############################################################################
###############################################################################
# NOW WE HAVE TIME SERIES, SO WE CAN USE OUR PREVIOUS CODE AFTER MAKING ITS INDEX LIKE PREVIOUS


# Extacting tapping time series from each subject and storing it into a list 'tap_series_all'
tap_series_allx=[]

for filename in func:
  
    timeseries_each_subject = extractor.transform(filename)
    ts_frame= pd.DataFrame(timeseries_each_subject,index=range(0,360,4))
    
    tap_series_each=[]
    for j in range(4):
        a=range(44+80*j,84+80*j,4)
        tap_series_each.append(ts_frame.loc[[i for i in ts_frame.index if i in a]].values)
    
    tap_series_each1=list(chain.from_iterable(tap_series_each))
    tap_series_each1=np.array(tap_series_each1)
    
    tap_series_allx.append(tap_series_each1)


tap_series_allx=np.array(tap_series_allx)
tap_series_all88=tap_series_allx.reshape(2080,n_regions_extracted)               
        
# Extacting resting time series from each subject and storing it into a list 'rest_series_all'
rest_series_allx=[]

for filename in func:
  
    timeseries_each_subject = extractor.transform(filename)
    ts_frame= pd.DataFrame(timeseries_each_subject,index=range(0,360,4))
    
    rest_series_each=[]
    for j in range(5):
        a=range(4+80*j,44+80*j,4)
        rest_series_each.append(ts_frame.loc[[i for i in ts_frame.index if i in a]].values)
    
    rest_series_each1=list(chain.from_iterable(rest_series_each))
    rest_series_each1=np.array(rest_series_each1)
    
    rest_series_allx.append(rest_series_each1)


rest_series_allx=np.array(rest_series_allx)
rest_series_all88=rest_series_allx.reshape(2548,n_regions_extracted)        

####################################################################################
##########  PCA  #######################################
## LOW ACCURACY: ONLY 63.11%

from sklearn.decomposition import PCA
pca = PCA(n_components='mle',svd_solver= 'full')
tap_all_pca=pca.fit_transform(tap_series_all22)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

#######################################


rest_all_pca=pca.fit_transform(rest_series_all22)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

####################################################################################
####################################################################################


    
#tap_series_all11=tap_series_all
#rest_series_all11=rest_series_all
        
plt.plot(rest_series_all22[1])        
plt.plot(tap_series_all22[1])     


# BELOW ONE WILL MAKE ACCURACY WORSE BECAUSE IT GAVE RISE TO OUTLIERS AND AS MANY NUMBERS ARE IN BETWEEN 
# 0 AND 1, SO IT GONNA MAKE BOTH CLASSES CLOSE TO ZERO AND HARDER TO SEPARATE
plt.plot( np.power(rest_series_all,3))   
plt.plot( np.power(tap_series_all,3)) 
        
        


















###############################################################################
# Plot resulting connectomes
# ----------------------------

title = 'Correlation between %d regions' % n_regions_extracted

# First plot the matrix
display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
                               colorbar=True, title=title)

# Then find the center of the regions and plot a connectome
regions_img = regions_extracted_img
coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)

plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title)

################################################################################
# Plot regions extracted for only one specific network
# ----------------------------------------------------
components_img=atlas_harvard_oxford.maps
# First, we plot a network of index=4 without region extraction (left plot)
from nilearn import image

img = image.index_img(components_img, 1)
coords = plotting.find_xyz_cut_coords(img)
display = plotting.plot_stat_map(img, cut_coords=coords, colorbar=False,
                                 title='Showing one specific network')

################################################################################
# Now, we plot (right side) same network after region extraction to show that
# connected regions are nicely seperated.
# Each brain extracted region is identified as separate color.

# For this, we take the indices of the all regions extracted related to original
# network given as 4.
regions_indices_of_map3 = np.where(np.array(regions_index) == 1)

display = plotting.plot_anat(cut_coords=coords,
                             title='Regions from this network')

# Add as an overlay all the regions of index 4
colors = 'rgbcmyk'
for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
    display.add_overlay(image.index_img(regions_extracted_img, each_index_of_map3),
                        cmap=plotting.cm.alpha_cmap(color))

plotting.show()




















































