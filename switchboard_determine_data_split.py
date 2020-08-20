import os
import xml.etree.ElementTree as ET

dialogue_tree = ET.parse("/group/corpora/public/switchboard/nxt/xml/corpus-resources/dialogues.xml")
root = dialogue_tree.getroot()
# /group/corpora/public/switchboard/nxt/xml/corpus-resources/speakers.xml

for child in root:
    print(child)