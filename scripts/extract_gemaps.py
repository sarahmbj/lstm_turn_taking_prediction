# -*- coding: utf-8 -*-
# takes about 30 mins
import os
import time as t
import sys

if len(sys.argv)==2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0 # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    output_files_dir = './data/signals/gemaps_features_50ms/' # first run of features straight into a time-specified directory
    # smile_command = 'SMILExtract -C ./opensmile-2.3.0/config/gemaps_50ms/eGeMAPSv01a.conf'
    smile_command = './utils/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract -C ./utils/opensmile-2.3.0/config/gemaps_50ms/eGeMAPSv01a.conf'
else:
    output_files_dir = './data/signals/gemaps_features_10ms/'
    # smile_command = 'SMILExtract -C ./opensmile-2.3.0/config/gemaps_10ms/eGeMAPSv01a.conf'
    smile_command = './utils/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract -C ./utils/opensmile-2.3.0/config/gemaps_10ms/eGeMAPSv01a.conf'


audio_files_dir = './data/signals/dialogues_mono/' # path to the files containing the audio data -- note we are using the mono -- not regular -- whats the diff? Mono only includes one speaker!
audio_files=os.listdir(audio_files_dir) #get the names of the files
csv_file_list = [ file.split('.')[0]+'.'+file.split('.')[1]+'.csv' for file in audio_files] # turn e.g. q1ec7.f.wav into q1ec7.f.csv

if not(os.path.exists(output_files_dir)):
    os.mkdir(output_files_dir)

t_1=t.time()
total_num_files = len(audio_files)
file_indx = 0
for input_file,output_file in zip(audio_files,csv_file_list): # zip together audio files in dialogues_mono and csv files to output
    #os.system('ls')
    file_indx +=1
    t_2 = t.time()
    print('processing file '+str(file_indx)+' out of '+str(total_num_files))
#    subprocess.check_output(smile_command + ' -I '+audio_files_dir+input_file+' -D '+output_files_dir+output_file)
    os.system(smile_command + ' -I '+audio_files_dir+input_file+' -D '+output_files_dir+output_file) # extract the features from input wav file and output csv to specified directory
    print('time taken for file: '+str(t.time()-t_2))

print('total time taken: '+str(t.time()-t_1))
