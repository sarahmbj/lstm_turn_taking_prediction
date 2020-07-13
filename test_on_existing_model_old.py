import torch
from pprint import pprint
from lstm_model import LSTMPredictor
from data_loader import TurnPredictionDataset
from torch.utils.data import DataLoader
import json
import os


num_layers = 1
annotations_dir = './data/extracted_annotations/voice_activity/'
test_list_path = './data/splits/testing.txt'
prediction_length = 60  # (3 seconds of prediction)
data_set_select = 0  # 0 for maptask, 1 for mahnob, 2 for switchboard
p_memory = True
train_batch_size = 128


def create_results_directory(directory, test_set, experiment_path):
    print(f"{directory}/{test_set}/{experiment_path}")
    os.makedirs(f"{directory}/{test_set}/{experiment_path}")


def remove_results_directory(directory, test_set, experiment_path):
    """reverses create_results_directory - for use when testing """
    print(f"{directory}/{test_set}/{experiment_path}")
    try:
        os.rmdir(f"{directory}/{test_set}/{experiment_path}")
    except FileNotFoundError:
        pass


def load_args(args_path):
    with open(args_path, "rb") as json_file:
        args_string = json.load(json_file)

    args_dict = json.loads(args_string)
    return args_dict

def load_model(pickled_model, args_dict, test_data):

    lstm_settings_dict = {  # this works because these variables are set in locals from json_dict
        'no_subnets': args_dict['no_subnets'],
        'hidden_dims': {
            'master': args_dict['hidden_nodes_master'],
            'acous': args_dict['hidden_nodes_acous'],
            'visual': args_dict['hidden_nodes_visual'],
        },
        'uses_master_time_rate': {},
        'time_step_size': {},
        'is_irregular': {},
        'layers': num_layers,
        'dropout': args_dict['dropout_dict'],
        'freeze_glove': args_dict['freeze_glove_embeddings']
    }
    lstm_settings_dict = test_data.get_lstm_settings_dict(
        lstm_settings_dict)  # add some extra items to the lstm settings related to the dataset

    # TODO check we can get feature_size_dict from test set (it comes from train set in run_json.py)
    model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, feature_size_dict=test_data.get_feature_size_dict(),
                          batch_size=train_batch_size, seq_length=args_dict['sequence_length'],
                          prediction_length=prediction_length, embedding_info=test_data.get_embedding_info())
    with open(pickled_model, "rb") as model_file:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_file))
        else:
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    return model


def load_test_set(args_dict, test_on_g=True, test_on_f=True):
    test_dataset = TurnPredictionDataset(args['feature_dict_list'], annotations_dir, test_list_path, args_dict['sequence_length'],
                                         prediction_length, 'test', data_select=data_set_select, test_on_f=test_on_f,
                                         test_on_g=test_on_g)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                 pin_memory=p_memory)

    return test_dataset, test_dataloader


if __name__ == "__main__":
    trial_path = './two_subnets_complete/1_Acous_50ms_Ling_50ms'
    test_path = f'{trial_path}/test'
    for directory in os.listdir(test_path):
        # remove_results_directory(trial_path, 'test_on_both', directory) # shouldn't need this in final script - used when testing functionality
        model_path = f'{test_path}/{directory}/model.p'
        settings_path = f'{test_path}/{directory}/settings.json'
        results_path = f"{trial_path}/test_on_both/{directory}"
        print('*********', results_path)
        try:
            os.rmdir(results_path)
        except FileNotFoundError:
            pass
        args = load_args(settings_path)
        test_set, test_loader = load_test_set(args, test_on_g=True, test_on_f=True)
        model = load_model(model_path, args, test_set)
        os.makedirs(results_path)

