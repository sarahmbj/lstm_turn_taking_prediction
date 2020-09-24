#!/bin/sh

cd ../dev && python maptask_tests.py && python onset_prediction_length_tests.py

cd ../maptask_unk && python maptask_unk_tests.py

cd ../switchboard_dev && python switchboard_tests.py && python onset_prediction_length_tests.py

cd ../switchboard_maptask_unk && python switchboard_unk_tests.py

cd ../both_sets && python both_tests.py

cd ../both_unk && python both_unk_tests.py