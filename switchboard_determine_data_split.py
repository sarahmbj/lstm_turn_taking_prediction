import platform
import xml.etree.ElementTree as ET
import re
from pprint import pprint
from collections import defaultdict
import random

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
print(f"there are {len(speaker_dict)} distinct speakers in the data set")
print(f"there are {len(speakers_in_multiple_dialogues)} speakers in multiple dialogues: {speakers_in_multiple_dialogues}")

# work out desired size of each split
max_test_dialogues = int(total_dialogues * test_split)
max_train_dialogues = total_dialogues - max_test_dialogues
print(f"total dialogues: {total_dialogues}, max test: {max_test_dialogues}, max train: {max_train_dialogues}")

# allocate dialogues randomly to each set
all_dialogues = list(dialogue_dict.keys())
random.shuffle(all_dialogues)
test_set = set(all_dialogues[0:max_test_dialogues])
train_set = set(all_dialogues[max_test_dialogues:])
print(test_set)
print(train_set)
print(len(test_set) + len(train_set))


# check how many speakers appear in both sets
def check_speaker_overlaps(test_dialogues, train_dialogues):
    test_speakers = set()
    train_speakers = set()
    for dialogue in test_dialogues:
        test_speakers.add(dialogue_dict[dialogue][0])
        test_speakers.add(dialogue_dict[dialogue][1])
    for dialogue in train_dialogues:
        train_speakers.add(dialogue_dict[dialogue][0])
        train_speakers.add(dialogue_dict[dialogue][1])
    overlap = test_speakers.intersection(train_speakers)
    print(f'There are {len(overlap)} speakers in both test sets.')
    return overlap

overlap_speakers = check_speaker_overlaps(test_set, train_set)

while overlap_speakers:
    speaker = overlap_speakers.pop()
    print(speaker)
    dialogues = speaker_dict[speaker]
    print(dialogues)
    for dialogue in dialogues[0]:
        if dialogue in test_set:
            test_set.remove(dialogue)
            train_set.add(dialogue)
            swap = test_set.pop()
            train_set.add(swap)
            print(f"swap in test_set loop: {swap}")

        else:
            train_set.remove(dialogue)
            test_set.add(dialogue)
            swap = test_set.pop()
            test_set.add(swap)
            print(f"swap in train_set loop: {swap}")

    overlap_speakers = check_speaker_overlaps(test_set, train_set)




