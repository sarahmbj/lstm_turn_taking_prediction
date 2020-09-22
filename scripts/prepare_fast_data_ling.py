# -*- coding: utf-8 -*-
"""This script takes in csv files for speakers f and g from './data/extracted_annotations', 
time_scale_folder' (time_scale_folder used in prepare_data.py is words_advanced_50ms_averaged 
and then words_advanced_10ms_averaged).
The output of this script is a vector (h5py dataset) for each file, for each speaker, 
for each feature of those featur≠e values over time, in h5py dataset format, stored as 
'./data/datasets/*.hdf5' (file name is whatever output_name is set to: when run from 
prepare_data.py it creates words_split_50ms.hdf5 and words_split_10ms_5_chunked.hdf5).
Don't think this does any processing over the data, other than storing it in a different format?"""

import os
import numpy as np
import pandas as pd
import h5py 
from itertools import zip_longest
import time as t
import multiprocessing
import sys

# load data: hold_shift, onsets, overlaps
# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
num_workers = 4
data_select = 0

dataset = "switchboard_data"  # default should be "data" - change if doing cross-corpora tests

# %% Settings
if len(sys.argv) == 2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0 


if speed_setting == 0:  # regular/fast_irregular 50ms
    max_len = 1
    annotations_dir = f'./{dataset}/extracted_annotations/voice_activity'
    time_scale_folder = 'words_advanced_50ms_averaged'
    output_name = 'words_split_50ms'
    features_list = ['word']

elif speed_setting == 1:  # regular 10ms
    max_len = 2
    annotations_dir = f'./{dataset}/extracted_annotations/voice_activity'  # for chunking into 50ms chunks
    time_scale_folder = 'words_advanced_10ms_averaged'
    output_name = 'words_split_10ms_5_chunked'
    features_list = ['word']

elif speed_setting == 2:  # irregular 50ms
    max_len = 2
    annotations_dir = f'./{dataset}/extracted_annotations/voice_activity'
    time_scale_folder = 'words_advanced_50ms_raw'
    output_name = 'words_split_irreg_50ms'
    # features_list = ['frameTimes', 'word']
    features_list = ['word']
# todo: irregular 10ms

time_label_select = 2
file_list = list(pd.read_csv(f'./{dataset}/splits/complete.txt', header=None, dtype=str)[0])


# %% Funcs
# load data: hold_shift, onsets, overlaps
# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
if 'of_split' in locals():
    if isinstance(of_split, h5py.File):  # Just HDF5 files
        try:
            of_split.close()
        except:
            pass  # Was already closed


def find_max_len(df_list):
    max_len = 0
    for df in df_list:
        max_len = max(max_len, len(df))
    return max_len


def fill_array(arr, seq):
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = np.nan
    else:
        for subarr, subseq in zip_longest(arr, seq, fillvalue=()):
            fill_array(subarr, subseq)


data_select_dict = {0: ['f', 'g'],
                    1: ['c1', 'c2']}
time_label_select_dict = {0: 'frame_time',  # gemaps
                          1: 'timestamp',  # openface
                          2: 'frameTimes'}  # annotations

if os.path.exists('./{dataset}/datasets/' + output_name + '.hdf5'):
    os.remove('./{dataset}/datasets/' + output_name + '.hdf5')

out_split = h5py.File('./{dataset}/datasets/' + output_name + '.hdf5', 'w')
folder_path = os.path.join('./{dataset}/extracted_annotations', time_scale_folder)

# structure: of_split/file/[f,g][x,x_i]/feature/matrix[T,4]
#%% run
t_1=t.time()
#for file_name in file_list:
#file_name = file_list[0]
def file_run(file_name):
    """Takes a file name as an argument and uses it to define filepaths to load and save data in csv format. Returns a dictionary
    containing the f (information follower) & g (information giver) word features. Annot_f is a column of the ‘follower’ csv with the heading
    ‘frameTimes’. Each item in the column is added to the list ‘split_indices’. Max_len is set equal to the length of the item in data_f_temp 
    with the most words(?). Function iterates over features_list (but it is a list with a single item, ['word']) and fills arrays with the f&g data for each feature. """
    annot_f = pd.read_csv(annotations_dir+'/'+file_name+'.'+data_select_dict[data_select][0]+'.csv',delimiter=',')
#    annot_g = pd.read_csv(annotations_dir+'/'+file_name+'.'+data_select_dict[data_select][1]+'.csv',delimiter=',')
                    
    data_f_temp = pd.read_csv(folder_path+'/'+file_name+'.'+data_select_dict[data_select][0]+'.csv')
    data_g_temp = pd.read_csv(folder_path+'/'+file_name+'.'+data_select_dict[data_select][1]+'.csv')
    
    data_f_split_list,data_g_split_list = [],[]
    split_indices=[]
    for f_time in annot_f['frameTimes'][1:]:
        split_indices.append(np.max(np.where(data_f_temp[time_label_select_dict[time_label_select]] < f_time))+1)
    data_f_split_list = np.split(data_f_temp,split_indices)
    data_g_split_list = np.split(data_g_temp,split_indices)
    max_len = find_max_len(data_f_split_list)
    data_dict_f = {'x':{},'x_i':{}}
    data_dict_g = {'x':{},'x_i':{}}
    for feature_name in features_list:    
        data_dict_f['x'][feature_name] = np.zeros((len(annot_f['frameTimes']),max_len))
        data_dict_f['x_i'][feature_name] = np.zeros((len(annot_f['frameTimes']),max_len))
        data_dict_g['x'][feature_name] = np.zeros((len(annot_f['frameTimes']),max_len))
        data_dict_g['x_i'][feature_name] = np.zeros((len(annot_f['frameTimes']),max_len)) # !!! I don't actually need to save x_i for each feature, rerun at some stage
        for ind, (data_f_split, data_g_split) in enumerate(zip(data_f_split_list,data_g_split_list)):
            data_dict_f['x'][feature_name][ind,:len(data_f_split)] = data_f_split[feature_name]
            data_dict_f['x_i'][feature_name][ind,:len(data_f_split)] = 1 
            data_dict_g['x'][feature_name][ind,:len(data_g_split)] = data_g_split[feature_name]
            data_dict_g['x_i'][feature_name][ind,:len(data_g_split)] = 1 
            
    data_fg = {file_name:{data_select_dict[data_select][0]:data_dict_f,data_select_dict[data_select][1]:data_dict_g}}

    return data_fg
         
if __name__=='__main__':
    p = multiprocessing.Pool(num_workers)
    results = p.map(file_run,(file_list),chunksize=1)
#     file_run(file_list[0])
    for file_name,data_fg in zip(file_list,results):
        for f_g in data_select_dict[data_select]:
            for x_type in ['x','x_i']:
                for feat in features_list:
                    print(' ')
                    out_split.create_dataset(file_name+'/'+f_g+'/'+x_type+'/'+feat ,data=np.array(data_fg[file_name][f_g][x_type][feat]))
out_split.close()
