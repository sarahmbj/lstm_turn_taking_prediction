import platform
import xml.etree.ElementTree as ET
import re
from pprint import pprint
from collections import defaultdict

# set paths to metadata
if platform.system() == 'Linux':
    print('Running on DICE')
    dialogue_tree = ET.parse("/group/corpora/public/switchboard/nxt/xml/corpus-resources/dialogues.xml") # on DICE
    swbd_complete_path = "./data/splits/complete.txt"
else:
    print('Running on local')
    dialogue_tree = ET.parse("./data/dialogues_excerpt.xml")  # local
    swbd_complete_path = "./data/splits/complete_swbd.txt"

# settings
test_split = 0.25

# get list of dialogues we want to use
dialogues_to_use = []
with open(swbd_complete_path) as file:
    for line in file:
        dialogues_to_use.append(line.strip())
total_dialogues = len(dialogues_to_use)

# get dictionary of dialogues and their speakers, and dictionary of speakers and their dialogues
dialogue_root = dialogue_tree.getroot()
dialogue_dict = {}
speaker_dict = defaultdict(list)
speaker_regex = re.compile(r"speakers.xml#id\((.*)\)")
for dialogue in dialogue_root:
    speakers = list()
    dialogue_id = f'sw0{dialogue.attrib["swbdid"]}'
    if dialogue_id in dialogues_to_use:
        for pointer in dialogue:
            pointer_value = pointer.attrib['href']
            speaker = re.search(speaker_regex, pointer_value)
            if speaker:
                speakers.append(speaker[1])
        dialogue_dict[dialogue_id] = speakers
        speaker_dict[speakers[0]].append(dialogue_id)
        speaker_dict[speakers[1]].append(dialogue_id)


pprint(dialogue_dict)
pprint(speaker_dict)

#work out which speakers are in multiple conversations
speakers_in_multiple_dialogues = set()
for speaker in speaker_dict:
    if len(speaker_dict[speaker]) > 1:
        speakers_in_multiple_dialogues.add(speaker)
print(f"there are {len(speakers_in_multiple_dialogues)} speakers in multiple dialogues: {speakers_in_multiple_dialogues}")

#allocate all conversations for these speakers to either train or test sets

#allocate remaining dialogues to train/test sets
max_test_dialogues = int(total_dialogues * test_split)
max_train_dialogues = total_dialogues - max_test_dialogues
print(f"total dialogues: {total_dialogues}, max test: {max_test_dialogues}, max train: {max_train_dialogues}")
test_speaker_set = set()
train_speaker_set = set()
test_dialogues = set()
train_dialogues = set()
# for dialogue in dialogue_dict:
#     speakers = dialogue_dict[dialogue]
#     print( dialogue, speakers[0], speakers[1])
#     if speakers[0] in test_speaker_set:



