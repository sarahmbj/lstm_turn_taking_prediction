import os
import xml.etree.ElementTree as ET

dialogue_tree = ET.parse("/group/corpora/public/switchboard/nxt/xml/corpus-resources/dialogues.xml")
dialogue_root = dialogue_tree.getroot()
# /group/corpora/public/switchboard/nxt/xml/corpus-resources/speakers.xml

for dialogue in dialogue_root:
    print(dialogue.tag, dialogue.attrib)
    