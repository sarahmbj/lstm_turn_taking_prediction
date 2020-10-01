import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t
import pickle
import sys
import json

"""Takes in raw word embeddings and outputs averaged word embeddings.
Averaged word embeddings are the async word representations collected at the end of each word (rather than at each time interval).
Averaging is done for each time interval (10ms/50ms) so the words can be input to the LSTMs along with other features being 
measured at those intervals (i.e. acoustic features)."""

dataset = "switchboard_data"  # the cross corpus test set that is being prepared (training data will be in ./data/)

# select settings for 50ms (0) or 10ms (1) features
# takes 1.5 mins for 50ms, 3 mins for 10ms setting
if len(sys.argv) == 2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0  # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    path_to_features = f'./{dataset}/signals/gemaps_features_processed_50ms/znormalized/'
    path_to_orig_embeds = f'./{dataset}/extracted_annotations/words_advanced_50ms_raw/'
    path_to_extracted_annotations = f'./{dataset}/extracted_annotations/words_advanced_50ms_averaged/'
    set_dict_path = f'./data/extracted_annotations/set_dict_50ms.p'  # training data set dict

elif speed_setting == 1:
    path_to_features = f'./{dataset}/signals/gemaps_features_processed_10ms/znormalized/'
    path_to_orig_embeds = f'./{dataset}/extracted_annotations/words_advanced_10ms_raw/'
    path_to_extracted_annotations = f'./{dataset}/extracted_annotations/words_advanced_10ms_averaged/'
    set_dict_path = f'./data/extracted_annotations/set_dict_10ms.p'  # training data set dict

t_1 = t.time()


if not(os.path.exists(path_to_extracted_annotations)):
    os.mkdir(path_to_extracted_annotations)

files_feature_list = os.listdir(path_to_orig_embeds)
files_annotation_list = list()
files_output_list = list()
for file in files_feature_list:
    base_name = os.path.basename(file)
    files_annotation_list.append(os.path.splitext(base_name)[0]+'.timed-units.xml')
    files_output_list.append(os.path.splitext(base_name)[0]+'.csv')

word_to_ix = pickle.load(open('./data/extracted_annotations/word_to_ix.p', 'rb'))
# glove_embed_table = pickle.load(open('./extracted_annotations/glove_embed_table.p','rb'))

# Create delayed frame annotations #DO WE NEED THIS IN THE CROSS CORP CASE?
max_len = 0
total_list = []
for i in range(0, len(files_feature_list)):

    print('percent done files create:'+str(i/len(files_feature_list))[0:4])
    orig_file = pd.read_csv(path_to_orig_embeds+files_feature_list[i], delimiter=',')
    combins = np.array(orig_file[orig_file.columns[1:]])[list(set(np.nonzero(np.array
                                                                             (orig_file[orig_file.columns[1:]]))[0]))]
    local_set = [frozenset(i) for i in combins]
    total_list.extend(local_set)

total_set = set(total_list)

# create new averaged glove embedding dict (can maybe try different approaches apart from averaging in future)
# set_dict, glove_embed_dict_50ms = {}, {}
# for indx, glove_combination in enumerate(total_set):
#     set_dict[glove_combination] = indx+1
#
# set_dict[frozenset([0])] = 0

# load in the set dict created for the training data
set_dict = pickle.load(open(set_dict_path, 'rb'))
# load in word_to_ix from training data to get the index of --unk--
word_to_ix = pickle.load(open('./data/extracted_annotations/word_to_ix.p', 'rb'))
unk_index = word_to_ix['--unk--']

print('set dict len: ', len(set_dict))
print('word to ix len: ', len(word_to_ix))
print("unk index: ", unk_index)

# get new word_reg annotations for new embedding dict
for i in range(0, len(files_feature_list)):

    print('percent done files create:' + str(i/len(files_feature_list))[0:4])
    orig_file = pd.read_csv(path_to_orig_embeds + files_feature_list[i], delimiter=',')
    frame_times = orig_file['frameTimes']
    word_annotations = np.zeros(frame_times.shape)
    # 'indices' is all the frame times with a non-zero annotation (i.e. there is a word there)
    indices = list(set(np.nonzero(np.array(orig_file[orig_file.columns[1:]]))[0]))
    # for all frame times where there are words, add the token index from the new set dict
    key_error_count = 0
    for indx in indices:  # deal with unknown words TODO
        try:
            word_annotations[indx] = set_dict[frozenset(np.array(orig_file[orig_file.columns[1:]])[indx])]
        except KeyError:
                key_error_count += 1
                print(frozenset(np.array(orig_file[orig_file.columns[1:]])[indx]))
                word_annotations[indx] = set_dict[frozenset([unk_index])]

    output = pd.DataFrame(np.vstack([frame_times, word_annotations]).transpose())
    output.columns = ['frameTimes', 'word']
    output.to_csv(path_to_extracted_annotations + files_output_list[i], float_format='%.6f', sep=',', index=False,
                  header=True)

print('key error count: ', key_error_count)
print('total_time: ' + str(t.time()-t_1))

