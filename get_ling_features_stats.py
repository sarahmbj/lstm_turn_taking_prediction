# designed to be read from both_sets folder, to get data about both corpora

import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t
import pickle
import sys
import json
import nltk
nltk.download('punkt')


def find_nearest(array,value):  # function returns the index of an element in an array which most closely matches a specified value
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()


# get vocabulary for maptask
# TODO: CHANGE TO ONLY USE FILES IN A GIVEN LIST
# TODO: MAKE INTO A FUNCTION SO WE CAN DO FOR MULTIPLE COMBINATIONS EASILY
path_to_features_maptask = './data/signals/gemaps_features_processed_50ms/znormalized/'  # uses processed features
path_to_annotations_maptask = './data/maptaskv2-1/Data/timed-units/'  # voice activity files

files_feature_list_maptask = os.listdir(path_to_features_maptask)  # file list
files_annotation_list_maptask = list()
for file in files_feature_list_maptask:
    base_name = os.path.basename(file)
    files_annotation_list_maptask.append(os.path.splitext(base_name)[0]+'.timed-units.xml')

no_change, disfluency_count, multi_word_count = 0, 0, 0
words_from_annotations_maptask = []
for i in range(0,len(files_feature_list_maptask)):
    print('percent done vocab build:'+str(i/len(files_feature_list_maptask))[0:4])
    # returns all roots from XML file -- can parse this tree to browse more specific constituents
    e = xml.etree.ElementTree.parse(path_to_annotations_maptask+files_annotation_list_maptask[i]).getroot()
    for atype in e.findall('tu'): # tu corresponds to transcribed utterances
        target_word = atype.text # harvest the text from the utterances
        target_word = target_word.strip() # strip spaces around
        target_word = target_word.lower()
        if '--' in target_word:
            target_word ='--disfluency_token--'
            words_from_annotations_maptask.append(target_word)
            disfluency_count += 1
        else:
            target_words = nltk.word_tokenize(target_word) # normalises words, converts into list if multiple
            words_from_annotations_maptask.extend(target_words)

vocab_maptask = set(words_from_annotations_maptask)  # turn words into set

# get vocabulary for Switchboard
#TODO: make switchboard function (or combine with the maptask function?)
#TODO: ADD counter to get frequencies
#TODO: Plot frequencies for each data set to see if it is Zipf-like
#TODO: calculate overlap between various sets (train and test for each experiment, and overlaps between datasets)
#TODO: metric for similarity of vocab between conversations (DICE/JACCARD MAYBE?)
