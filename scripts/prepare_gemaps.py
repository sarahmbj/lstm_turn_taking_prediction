# -*- coding: utf-8 -*-
import os
import time as t
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import scipy.io as io
from multiprocessing import Pool
import sys
num_workers = 4

# takes about 5 mins for 10ms, 1 min for 50ms
if len(sys.argv)==2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0 # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    input_gemaps_dir = './data/signals/gemaps_features_50ms/' # takes raw features in csv form
    output_gemaps_dir = './data/signals/gemaps_features_processed_50ms/'
elif speed_setting == 1:
    input_gemaps_dir = './data/signals/gemaps_features_10ms/'
    output_gemaps_dir = './data/signals/gemaps_features_processed_10ms/'
    shift_back_amount = 4 # The window_size is 50ms long so needs to be shifted back to avoid looking into future.

annotation_dir = './data/extracted_annotations/voice_activity/' # voice activity is already here ? not created?

csv_files=os.listdir(input_gemaps_dir)
voice_activity_files = [file.split('.')[0]+'.'+file.split('.')[1]+'.csv' for file in csv_files] # essentially creates a list with files in the same order as feature files but for voice activity
skip_normalization_list = [] # there are some features we might not want to z-normalise -- thus 'skipping them'
if not(os.path.exists(output_gemaps_dir)):
    os.mkdir(output_gemaps_dir)
for feature_set in ['raw','znormalized','/znormalized_pooled']:
    if not(os.path.exists(output_gemaps_dir+'/'+feature_set)):
        os.mkdir(output_gemaps_dir+'/'+feature_set)

### whats up with the masks? ask catherine or matthew!

frequency_features_list = ['F0semitoneFrom27.5Hz','jitterLocal','F1frequency',
                           'F1bandwidth','F2frequency','F3frequency']
frequency_mask_list = [0,0,0,
                       0,0,0]
energy_features_list = ['Loudness','shimmerLocaldB', 'HNRdBACF']
energy_mask_list = [0,0,0]
spectral_features_list = ['alphaRatio','hammarbergIndex','spectralFlux',
                          'slope0-500', 'slope500-1500','F1amplitudeLogRelF0',
                          'F2amplitudeLogRelF0','F3amplitudeLogRelF0','mfcc1',
                          'mfcc2','mfcc3', 'mfcc4']
spectral_mask_list = [0,0,0,
                      0,0,-201,
                      -201,-201,0,
                      0,0,0]

### whats up with the masks? ask catherine or matthew!

gemaps_full_feature_list = frequency_features_list + energy_features_list + spectral_features_list
gemaps_feat_name_list = frequency_features_list + energy_features_list + spectral_features_list
#full_mask_list = frequency_mask_list + energy_mask_list + spectral_mask_list

#%% get the names of features and test the alignment of features

test_covarep = pd.read_csv(input_gemaps_dir+csv_files[0],delimiter=',')
test_gemaps = pd.read_csv(input_gemaps_dir+csv_files[0],delimiter=',')



#%% loop through files

curTime = t.time()
missing_count = 0


def loop_func_one(data):
    #%%
    target_file, annotation_file = data

    # Get files and annotations
#    target_mat_covarep = io.loadmat(input_gemaps_dir+target_file)
#    target_csv_gemaps = target_mat_covarep['features']
    target_csv_gemaps = pd.read_csv(input_gemaps_dir+target_file,delimiter=',') # csv feature files output by opensmile
    mean_list, max_list, min_list, std_list, num_vals = [], [], [], [], []
    num_vals.append( len(target_csv_gemaps)) # as far as I can tell, just the no. of features in the utt extracted -- possibly also no of rows
    # raw features
    temp_dict = {}
    temp_dict['frame_time'] = target_csv_gemaps['frameTime'] # the frame time vector retrieved from the csv for which features are extracted
    for feature in gemaps_full_feature_list:
        if speed_setting ==1:
            tmp = np.zeros(target_csv_gemaps[feature].shape)
            tmp[:-shift_back_amount] = target_csv_gemaps[feature][shift_back_amount:]
            temp_dict[feature] = tmp
        else:
            temp_dict[feature] = target_csv_gemaps[feature] # just copying feature values to dictionary
    outputcsv = pd.DataFrame(temp_dict)
    outputcsv[list(temp_dict.keys())].to_csv(output_gemaps_dir+'raw/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True) # output raw csvs now with features better organised? or just the same?

    # znormalized
    temp_dict = {}
    covarep_std_list,covarep_mean_list,covarep_max_list,covarep_min_list,gemaps_feat_name_list = [],[],[],[],[]
    temp_dict['frame_time'] = target_csv_gemaps['frameTime'] # everything is just redefined in above three lines to 'empty it out'
    for feature in gemaps_full_feature_list:
        if feature in skip_normalization_list:
            if speed_setting == 1:
                temp_dict[feature] = np.zeros(target_csv_gemaps[feature].shape)
                temp_dict[feature][:-shift_back_amount] = target_csv_gemaps[feature][shift_back_amount:] # whats going on here? don't really understand!
            else:
                temp_dict[feature] = target_csv_gemaps[feature] # same as in raw csv case
        else:
            if speed_setting == 1:
                temp_dict[feature] = np.zeros(target_csv_gemaps[feature].shape)
                temp_dict[feature][:-shift_back_amount] = preprocessing.scale(target_csv_gemaps[feature])[shift_back_amount:]
            else:
                temp_dict[feature] = preprocessing.scale(target_csv_gemaps[feature] ) # scales means to 0 and SDs to 1

            gemaps_feat_name_list.append(feature)
            covarep_std_list.append(np.std(target_csv_gemaps[feature],axis=0)) # SD of feature throughout file
            covarep_mean_list.append(np.mean(target_csv_gemaps[feature],axis=0)) # Mean of feature through file
            covarep_max_list.append(np.max(target_csv_gemaps[feature],axis=0)) # Max of feature through file
            covarep_min_list.append(np.min(target_csv_gemaps[feature],axis=0)) # Min of feature through file

    mean_list.append(covarep_mean_list)
    std_list.append(covarep_std_list)
    min_list.append(covarep_min_list)
    max_list.append(covarep_max_list)# min and max are both appended to -- but not output by file -- this is weird because there is a global min_list var but it gets overwritten in this function?

    outputcsv = pd.DataFrame(temp_dict)
    outputcsv[list(temp_dict.keys())].to_csv(output_gemaps_dir+'znormalized/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True) # into a csv again

    return std_list,mean_list,num_vals # returned to calculate loop 2

#for target_file, annotation_file in zip(csv_files,voice_activity_files):
def loop_func_two(data):
    target_file, annotation_file,variance_pd,mean_pd = data
    target_csv_gemaps = pd.read_csv(input_gemaps_dir+target_file,delimiter=',')

    # znormalized_pooled
    temp_dict = {}
#    covarep_std_list,covarep_mean_list,covarep_max_list,covarep_min_list,gemaps_feat_name_list = [],[],[],[],[]
    temp_dict['frame_time'] = target_csv_gemaps['frameTime']
    for feature in gemaps_full_feature_list:
        if feature in skip_normalization_list:
            if speed_setting == 1:
                temp_dict[feature] = np.zeros(target_csv_gemaps[feature].shape)
                temp_dict[feature][:-shift_back_amount] = target_csv_gemaps[shift_back_amount:]  # done to prevent looking in future -- given window is 50 ms
            else:
                temp_dict[feature] = np.array(target_csv_gemaps[feature]) # basically this is just adding the feature data under the umbrella of each feature in a dictionary format for better processing (i assume)
        else:
            if speed_setting ==1:
                tmp = np.zeros(target_csv_gemaps[feature].shape)
                tmp2 = (np.array(target_csv_gemaps[feature]) - np.array(mean_pd[feature]))/np.array(variance_pd[feature])
                tmp[:-shift_back_amount] = tmp2[shift_back_amount:]
                temp_dict[feature] = tmp
            else:
                temp_dict[feature] =  (np.array(target_csv_gemaps[feature]) - np.array(mean_pd[feature]))/np.array(variance_pd[feature]) # alters all values of feature to normalise manually, since we want to normalise using values across different dialogues and participants.


    outputcsv = pd.DataFrame(temp_dict)
    outputcsv = outputcsv[['frame_time']+gemaps_full_feature_list]
    outputcsv = outputcsv.fillna(0)
    outputcsv.to_csv(output_gemaps_dir+'znormalized_pooled/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True)


#loop_func_one(my_data_one[0])


mean_list, max_list, min_list, std_list,num_vals = [],[],[],[],[] # define some containers
if __name__ == '__main__':
    my_data_one = []
    for target_file, annotation_file in zip(csv_files,voice_activity_files):
        my_data_one.append([target_file,annotation_file])
    p = Pool(num_workers)
    multi_output=p.map(loop_func_one,my_data_one) # get out big nested lists
    std_list,mean_list,num_vals = [],[],[]
    for l in multi_output: # over all files
        std_list.append(l[0][0]) # pulls out SD for each feature
        mean_list.append(l[1][0]) # pull out mean for each feature
        num_vals.append(l[2][0]) # pull out number of rows?

    totalTime = t.time() - curTime
    print('Time taken:')
    print(totalTime)

    #%% Reprocess values

    for feature_set in ['znormalized_pooled']:
        if not(os.path.exists(output_gemaps_dir+'/'+feature_set)):
            os.mkdir(output_gemaps_dir+'/'+feature_set)

    #max_pd=pd.DataFrame(columns=gemaps_feat_name_list)
    #max_pd.loc[0] = np.transpose(np.max(np.array(max_list),axis=0))
    #min_pd=pd.DataFrame(columns=gemaps_feat_name_list)
    #min_pd.loc[0] = np.transpose(np.min(np.array(max_list),axis=0))

    numerator_variance = np.sum(np.tile(np.array(num_vals)-1,[np.array(std_list).shape[1],1]).transpose() * np.array(std_list)**2,axis=0) # function to square values -- get squared value over all features # num vals - 1 because pull out feature description col. do by no. of columns? -- multiply by standard deviations?
    pooled_variance = np.sqrt(numerator_variance/(sum(num_vals)-len(num_vals))) # process to get variance over all samples

    numerator_mean = np.sum(np.tile(np.array(num_vals),[np.array(mean_list).shape[1],1]).transpose() * np.array(mean_list),axis=0)
    pooled_mean = numerator_mean/(sum(num_vals)) # mean across all diff dialogue samples

    mean_pd,variance_pd = {},{}
    for feat_name,p_mean,p_var in zip(gemaps_feat_name_list,pooled_mean,pooled_variance): # [each feature name, feature mean pooled across all files, feature variance pooled across all files]
        mean_pd[feat_name] = p_mean
        variance_pd[feat_name] = p_var

    my_data_two = []
    for target_file, annotation_file in zip(csv_files,voice_activity_files):
        my_data_two.append([target_file,annotation_file,variance_pd,mean_pd]) # each element in list picks out 2 files (with same name), the pooled var dic and pooled mean dic

    p.map(loop_func_two,my_data_two)

#mean_pd=pd.DataFrame(columns=gemaps_feat_name_list)
#mean_pd.loc[0] = np.transpose(pooled_mean)
#variance_pd=pd.DataFrame(columns=gemaps_feat_name_list)
#variance_pd.loc[0] = np.transpose(pooled_variance)
