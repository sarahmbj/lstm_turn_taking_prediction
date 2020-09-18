from os import walk
import pandas as pd
import pickle

"""
Script Modified from script by Lena Smith https://github.com/lnaclst
Script creates 3 dictionaries: conv_dict, which contains word count by conversation, total_dict, which contains
total word count in the data set, and conv_vecs, which contains vectors with the counts of each word by index in each
conversation. Pickles all 3 dictionaries.
"""

# Total vocab count
# Vocab count for role
# Vocab count by conversation by role

base_path = './data/extracted_annotations/'

words_folder = base_path + 'words_advanced_50ms_averaged/'

conversations_to_include = './data/splits/training.txt'
include_f = True
include_g = True

#get list of file names to consider for the stats
files_to_include = []
if include_f is True:
    for line in files_to_include:
        conversations_to_include.append(words_folder + line.strip() + '.f.csv')
if include_g is True:
    for line in files_to_include:
        conversations_to_include.append(line.strip() + '.g.csv')


ix_to_word_file = open(base_path + 'ix_to_word.p', 'rb')
ix_to_word = pickle.load(ix_to_word_file)

print(len(ix_to_word))

conv_vecs = {}

for word in ix_to_word.values():
    conv_vecs[word] = 0

conv_dict = {}
total_dict = {}

for filename in files_to_include:
    print(filename)
    if filename not in conv_dict:
        conv_dict[filename] = {}
    csv = pd.read_csv(words_folder + filename, usecols=['word'])

    if filename not in conv_vecs:
        print("if filename not in conv_vecs")
        conv_vecs[filename] = [0]*len(ix_to_word)

    for x in csv['word']:
        if x != 0:  # don't count timestamps with no associated word
            conv_vecs[filename][int(x)] += 1

            if ix_to_word[x] not in total_dict:
                total_dict[ix_to_word[x]] = 1
            else:
                total_dict[ix_to_word[x]] += 1
            if ix_to_word[x] not in conv_dict[filename]:
                conv_dict[filename][ix_to_word[x]] = 1
            else:
                conv_dict[filename][ix_to_word[x]] += 1


print(len(conv_vecs))
print(len(conv_dict))
print(conv_dict)
print(len(total_dict))
print(total_dict)

pickle.dump(conv_vecs, open(base_path + 'conv_vectors.p', 'wb'))
pickle.dump(conv_dict, open(base_path + 'conv_count_dict.p', 'wb'))
pickle.dump(total_dict, open(base_path + 'total_count_dict.p', 'wb'))

