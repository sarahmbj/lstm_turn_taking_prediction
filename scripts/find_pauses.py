# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import os


#%% Load file
file_list_path = './data/splits/complete.txt'
va_annotations_path = './data/extracted_annotations/voice_activity/'

# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
if 'file_hold_shift' in locals():
    if isinstance(file_hold_shift,h5py.File):   # Just HDF5 files
        try:
            file_hold_shift.close()
        except:
            pass # Was already closed

if not(os.path.exists('./data/datasets/')):
    os.makedirs('./data/datasets/')
file_hold_shift = h5py.File('./data/datasets/hold_shift.hdf5','w')

pause_str_list = ['50ms','250ms','500ms']
pause_length_list = [1,5,10] # one frame, five frames or ten frames
length_of_future_window = 20 # (one second) find instances where only one person speaks after the pause within this future window

#%% get Annotations
# Load list of test files
test_file_list = list(pd.read_csv(file_list_path,header=None)[0])
annotations_dict = {}
frame_times_dict = {}
for file_name in test_file_list:
    data_f = pd.read_csv(va_annotations_path+'/'+file_name+'.f.csv')
    data_g = pd.read_csv(va_annotations_path+'/'+file_name+'.g.csv')
    annotations = np.column_stack([np.array(data_g)[:,1].astype(bool),np.array(data_f)[:,1].astype(bool)]) #combining voice activity arrays for f and g into a matrix
    annotations_dict[file_name] = annotations
    frame_times_dict[file_name] = np.array(data_g)[:,0]

#%% loop through pause lengths

for pause_str, pause_length in zip(pause_str_list,pause_length_list):
    pause_seq_list = list()
    g_turns_list = list()
    f_turns_list = list()
    g_hold_shift_list = list()
    f_hold_shift_list = list()
    turns_sum = 0
    list_sum = 0
    uncertain_sum = 0
    num_holds = 0
    num_shifts = 0
    
    for seq in test_file_list: #looking through the list of all dialogue files
        annotations = annotations_dict[seq] #pulls out a matrix for each dialogue in the loop
            
        #%% find pauses
        pauses = ~(annotations[:,0] | annotations[:,1]) #pulls the matrix for each dialogue into the respective speakers, and checks if anyone is speaking
        # False is when someone is speaking
        # pauses is a vector of True and False for each frame of the dialogue
        
        pause_list = list()
        g_hold_shift = list()
        f_hold_shift = list()
        uncertain = list()
        
        
        for i in range(pause_length,len(pauses)):    #e.g. if pause_length = 10 (for 500ms pause), will loop through indices from frame 10 to the end
            if all(pauses[i-pause_length:i]==True) & (pauses[i-pause_length-1]==False): #second condition checks someone was speaking before the pause (i.e. it isn't a longer pause)
                
                if (annotations[i-pause_length-1,0] & annotations[i-pause_length-1,1]): # check if last speech frame had speech by both speakers
                    uncertain.append(i)
                else:
                    pause_list.append(i) # start calculating who takes turn from i (inclusive)
                        
        pause_seq_list.append(pause_list) # will be a list, containing a list of pause indexes (i.e. frame number) for each dialogue
        list_sum +=len(pause_list)
        uncertain_sum += len(uncertain)
        
        for i in pause_list: #for each pause, works out if it is a hold or a shift
            if annotations[i-pause_length-1,0]: #this if statement works out who current speaker is (0 or 1 refers to speaker)
                if (any(annotations[i:i+length_of_future_window,0]) and not(any(annotations[i:i+length_of_future_window,1]))):
                    g_hold_shift.append([i,0]) # [index of pause, hold(0) or shift(1)]
                    num_holds += 1
                    turns_sum += 1
                if any(annotations[i:i+length_of_future_window,1]) and not(any(annotations[i:i+length_of_future_window,0])):
                    g_hold_shift.append([i,1]) # [index of pause, hold(0) or shift(1)]
                    num_shifts += 1
                    turns_sum += 1
            elif annotations[i-pause_length-1,1]: #this if statement works out who current speaker is (0 or 1 refers to speaker)
                if any(annotations[i:i+length_of_future_window,1]) and not(any(annotations[i:i+length_of_future_window,0])):
                    f_hold_shift.append([i,0]) # [index of pause, hold(0) or shift(1)]
                    num_holds += 1
                    turns_sum += 1                
                if any(annotations[i:i+length_of_future_window,0]) and not(any(annotations[i:i+length_of_future_window,1])):
                    f_hold_shift.append([i,1]) # [index of pause, hold(0) or shift(1)]
                    num_shifts += 1
                    turns_sum += 1
            else:
                raise('pause list indexing problem')
        
        g_hold_shift_list.append(g_hold_shift)
        f_hold_shift_list.append(f_hold_shift)
        
    print(pause_str)
    print('num of instances: {}'.format(list_sum))  #total pauses across all dialogues (apart from uncertains)
    print('num Uncertain: {}'.format(uncertain_sum))
    print('num of instances where only one speaker continued: {}'.format(turns_sum))
    print('num holds: {}'.format(num_holds))
    print('num shifts: {}'.format(num_shifts))
    #%% Write to file
    # structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
    dset_stats = file_hold_shift[pause_str+'/stats/num_holds'] = num_holds
    dset_stats = file_hold_shift[pause_str+'/stats/num_shifts'] = num_shifts
    dset_stats = file_hold_shift[pause_str+'/stats/num_instances_total'] = list_sum
    dset_stats = file_hold_shift[pause_str+'/stats/num_instances_one_speaker_continues'] = turns_sum
    
    for seq,indx in zip(test_file_list,range(len(test_file_list))): # writes a file for each dialogue (seq) and speaker (indx) of where the speaker is holding and shifting
        file_hold_shift.create_dataset(pause_str+'/hold_shift/'+seq+'/'+'g',data=np.array(g_hold_shift_list[indx]))
        file_hold_shift.create_dataset(pause_str+'/hold_shift/'+seq+'/'+'f',data=np.array(f_hold_shift_list[indx]))

file_hold_shift.close()

        
