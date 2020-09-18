from os import walk
import pandas as pd
import pickle
import numpy as np

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
conversations_list = 'complete'
conversations_list_file = f'./data/splits/{conversations_list}.txt'
include_f = False
include_g = True

# get list of file names to consider for the stats
conversations_to_include = []
with open(conversations_list_file, "r") as file:
    for line in file:
        conversations_to_include.append(line)
files_to_include = []
if include_f is True:
    for line in conversations_to_include:
        print(line)
        files_to_include.append(line.strip() + '.f.csv')
        print(files_to_include[-1])
if include_g is True:
    for line in conversations_to_include:
        print(line)
        files_to_include.append(line.strip() + '.g.csv')
        print(files_to_include[-1])


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
print(len(total_dict))

pickle.dump(conv_vecs, open(base_path + 'conv_vectors.p', 'wb'))
pickle.dump(conv_dict, open(base_path + 'conv_count_dict.p', 'wb'))
pickle.dump(total_dict, open(base_path + 'total_count_dict.p', 'wb'))

# Write stats to file

total_tokens = sum(total_dict.values())
total_types = len(total_dict)
no_conversations = len(conversations_to_include)

types_per_convo = []
tokens_per_convo = []
for conversation in conv_dict:
    types_per_convo.append(len(conversation)) #TODO: CHECK THIS!
    tokens_per_convo.append(sum(conversation.values))

with open(f"{conversations_list}_vocab_stats.txt", "a") as file:
    file.write(f"Vocab stats for ~~ {conversations_list} ~~ \n")
    file.write(f"include_f is: {str(include_f)}\t include_g is: {str(include_g)}\n")
    file.write(f"Number of conversations: {no_conversations}\n")
    file.write(f"Total tokens in dataset: {total_tokens}\n")
    file.write(f"Total types in dataset: {total_types}\n")
    file.write(f"Mean tokens per conversation: {np.mean(tokens_per_convo)}\tRange: {np.range(tokens_per_convo)}\t"
               f"Standard dev: {np.std(tokens_per_convo)}\n")
    file.write(f"Mean types per conversation: {np.mean(types_per_convo)}\tRange: {np.range(types_per_convo)}"
               f"\tStandard dev: {np.std(types_per_convo)}\n")

    file.write("Number of types appearing once only: \n")
    file.write("Percentage of total types: \t Percentage of total tokens: \t\n ")

    file.write("****************************************************\n\n\n")

