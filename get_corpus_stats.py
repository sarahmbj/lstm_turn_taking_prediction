import sys
import numpy as np

base_path = './data/extracted_annotations/voice_activity/'
conversations_list = sys.argv[1]  # complete, testing or training
conversations_list_file = f'./data/splits/{conversations_list}.txt'


# get list of file names to consider for the stats
files_to_include = []
with open(conversations_list_file, "r") as file:
    for line in file:
        files_to_include.append(line.strip() + '.f.csv') # could use f or g - both are same length

#loop through every conversation in the set (f or g, it doesnt matter)
file_lengths = []
for file in files_to_include:
    with open(base_path + file, "r") as f:
        last_line = f.readlines()[-1]
        print(file)
        file_length = float(last_line.split(",")[0])
        file_lengths.append(file_length)

print(sum(file_lengths)/60)
print(np.mean(file_lengths)/60)
print(min(file_lengths)/60)
print(max(file_lengths)/60)
print(np.std(file_lengths)/60)


