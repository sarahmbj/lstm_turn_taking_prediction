import glob
import os
import wave

wav_directory = "./data/signals/dialogues_stereo"
conversation_list = os.listdir(wav_directory)
print(conversation_list)

for file in conversation_list:
    wav_file = wave.open(f'{wav_directory}/{file}')
    print(dir(wav_file))
