import torch
from pprint import pprint
from lstm_model import LSTMPredictor
from sys import argv
from data_loader import TurnPredictionDataset



def load_model(pickled_model, args):

    model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, feature_size_dict=feature_size_dict,
                      batch_size=train_batch_size, seq_length=sequence_length, prediction_length=prediction_length,
                      embedding_info=embedding_info)
    with open(pickled_model, "rb") as model_file:
        if torch.cuda.is_available():
            model.load_state_dict(model_file)
        else:
            model.load_state_dict(model_file, map_location=torch.device('cpu'))
        print(type(model))
        pprint(model)


def load_test_set():
    test_dataset = TurnPredictionDataset(feature_dict_list, annotations_dir, test_list_path, sequence_length,
                                         prediction_length, 'test', data_select=data_set_select, test_on_f=test_on_f,
                                         test_on_g=test_on_g)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                 pin_memory=p_memory)
    pass


proper_num_args = 2  # when called as subprocess, this is './test_on_existing_model.py' and a dict of the other args
print('Number of arguments is: ' + str(len(argv)))
with open("model_location.txt", "r") as file:
    model_location = file[-1]  # this only works if the most recent run_json was for this model

load_model('results/example_pickled_model/model.p', argv[1])
load_test_set()




