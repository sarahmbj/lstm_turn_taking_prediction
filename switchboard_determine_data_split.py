import os
import xml.etree.ElementTree as ET

dialogue_tree = ET.parse("/group/corpora/public/switchboard/nxt/xml/corpus-resources/dialogues.xml")
# /group/corpora/public/switchboard/nxt/xml/corpus-resources/speakers.xml

print(dialogue_tree)