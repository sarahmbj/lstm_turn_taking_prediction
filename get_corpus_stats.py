base_path = './data/extracted_annotations/voice_activity/'
conversations_list = sys.argv[1]  # complete, testing or training
conversations_list_file = f'.data/splits/{conversations_list}.txt'


# get list of file names to consider for the stats
files_to_include = []
with open(conversations_list_file, "r") as file:
    for line in file:
        files_to_include.append(line.strip() + '.f.csv') # could use f or g - both are same length

#loop through every conversation in the set (f or g, it doesnt matter)
file_lengths = []
for file in files_to_include:
    with open(file, "r") as f:
        last_line = f.readlines()[-1]
        print(file)
        print(last_line)
        quit()
        # get frame number of last line, append to file lengths

# get sum of file lengths
# get mean of file lengths


