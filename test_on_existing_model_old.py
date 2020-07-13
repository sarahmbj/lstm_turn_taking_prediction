import torch
from pprint import pprint
from lstm_model import LSTMPredictor
from data_loader import TurnPredictionDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import json
import os
import pandas as pd
import numpy as np


num_layers = 1
annotations_dir = './data/extracted_annotations/voice_activity/'
test_list_path = './data/splits/testing.txt'
prediction_length = 60  # (3 seconds of prediction)
data_set_select = 0  # 0 for maptask, 1 for mahnob, 2 for switchboard
p_memory = True
train_batch_size = 128

# Decide whether to use cuda or not
use_cuda = torch.cuda.is_available()
print('Use CUDA: ' + str(use_cuda))
if use_cuda:
    #    torch.cuda.device(randint(0,1))
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor


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


def test(model, test_dataset, test_dataloader):
    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    losses_mse, losses_l1 = [], []
    model.eval()
    # setup results_dict
    results_lengths = test_dataset.get_results_lengths()

    test_file_list = list(pd.read_csv(test_list_path, header=None, dtype=str)[0])
    data_select_dict = {0: ['f', 'g'],
                        1: ['c1', 'c2'],
                        2: ['A', 'B']}
    for file_name in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            # create new arrays for the results
            results_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])
            losses_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])

    for batch_indx, batch in enumerate(test_dataloader):

        model_input = []

        for b_i, bat in enumerate(batch):
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(torch.squeeze(bat, 0).transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(torch.squeeze(bat, 0).type(dtype)).transpose(0, 2).transpose(1, 2))

        y_test = Variable(torch.squeeze(batch[4].type(dtype), 0))

        info_test = batch[-1]
        batch_length = int(info_test['batch_size'])
        if batch_indx == 0:
            model.change_batch_size_reset_states(batch_length)
        else:
            model.change_batch_size_no_reset(batch_length)

        out_test = model(model_input)
        out_test = torch.transpose(out_test, 0, 1)

        if test_dataset.set_type == 'test':
            file_name_list = [info_test['file_names'][i][0] for i in range(len(info_test['file_names']))]
            gf_name_list = [info_test['g_f'][i][0] for i in range(len(info_test['g_f']))]
            time_index_list = [info_test['time_indices'][i][0] for i in range(len(info_test['time_indices']))]
        else:
            file_name_list = info_test['file_names']
            gf_name_list = info_test['g_f']
            time_index_list = info_test['time_indices']

        # Should be able to make other loss calculations faster
        # Too many calls to transpose as well. Should clean up loss pipeline
        y_test = y_test.permute(2, 0, 1)
        loss_no_reduce = loss_func_L1_no_reduce(out_test, y_test.transpose(0, 1))

        for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                 gf_name_list,
                                                                 time_index_list,
                                                                 range(batch_length)):

            results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = out_test[
                batch_indx].data.cpu().numpy()
            losses_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = loss_no_reduce[
                batch_indx].data.cpu().numpy()

        loss = loss_func_BCE(F.sigmoid(out_test), y_test.transpose(0, 1))
        # loss = loss_func_BCE_Logit(out_test,y_test.transpose(0,1))
        losses_test.append(loss.data.cpu().numpy())
        batch_sizes.append(batch_length)

        loss_l1 = loss_func_L1(out_test, y_test.transpose(0, 1))
        losses_l1.append(loss_l1.data.cpu().numpy())

    # get weighted mean
    loss_weighted_mean = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_test))) / np.sum(batch_sizes)
    loss_weighted_mean_l1 = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_l1))) / np.sum(batch_sizes)
    #    loss_weighted_mean_mse = np.sum( np.array(batch_sizes)*np.squeeze(np.array(losses_mse))) / np.sum( batch_sizes )

    for conv_key in test_file_list:

        results_dict[conv_key + '/' + data_select_dict[data_set_select][1]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][1]]).reshape(-1, prediction_length)
        results_dict[conv_key + '/' + data_select_dict[data_set_select][0]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][0]]).reshape(-1, prediction_length)

    # get hold-shift f-scores
    for pause_str in pause_str_list:
        true_vals = list()
        predicted_class = list()
        for conv_key in test_file_list:
            for g_f_key in list(hold_shift[pause_str + '/hold_shift' + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in hold_shift[pause_str + '/hold_shift' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if frame_indx < len(results_dict[conv_key + '/' + g_f_key]):
                        true_vals.append(true_val)
                        if np.sum( # using model outputs to decide a prediction of hold or shift
                                results_dict[conv_key + '/' + g_f_key][frame_indx, 0:length_of_future_window]) > np.sum(
                                results_dict[conv_key + '/' + g_f_key_not[0]][frame_indx, 0:length_of_future_window]):
                            predicted_class.append(0)
                        else:
                            predicted_class.append(1)
        f_score = f1_score(true_vals, predicted_class, average='weighted')
        results_save['f_scores_' + pause_str].append(f_score)
        tn, fp, fn, tp = confusion_matrix(true_vals, predicted_class).ravel() # true negative, false positive etc.
        results_save['tn_' + pause_str].append(tn)
        results_save['fp_' + pause_str].append(fp)
        results_save['fn_' + pause_str].append(fn)
        results_save['tp_' + pause_str].append(tp)
        print('majority vote f-score(' + pause_str + '):' + str(
            f1_score(true_vals, np.zeros([len(predicted_class)]).tolist(), average='weighted')))
    # get prediction at onset f-scores
    # first get best threshold from training data
    if onset_test_flag: # this is set to true at top of file
        onset_train_true_vals = list()
        onset_train_mean_vals = list()
        onset_threshs = []
        for conv_key in list(set(train_file_list).intersection(onsets['short_long'].keys())):
            for g_f_key in list(onsets['short_long' + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds

                    if (frame_indx < len(train_results_dict[conv_key + '/' + g_f_key])) and not (
                    np.isnan(np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
                        onset_train_true_vals.append(true_val)
                        onset_train_mean_vals.append(
                            np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))
        if not(len(onset_train_true_vals)==0):
            fpr, tpr, thresholds = roc_curve(np.array(onset_train_true_vals), np.array(onset_train_mean_vals))
        else:
            fpr,tpr,thresholds = 0,0,[0]
        thresh_indx = np.argmax(tpr - fpr)
        onset_thresh = thresholds[thresh_indx]
        onset_threshs.append(onset_thresh)

        true_vals_onset, onset_test_mean_vals, predicted_class_onset = [], [], []
        for conv_key in list(set(test_file_list).intersection(onsets['short_long'].keys())):
            for g_f_key in list(onsets['short_long' + '/' + conv_key].keys()):
                #                g_f_key_not = ['g','f']
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if (frame_indx < len(results_dict[conv_key + '/' + g_f_key])) and not (
                    np.isnan(np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
                        true_vals_onset.append(true_val)
                        onset_mean = np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :])
                        onset_test_mean_vals.append(onset_mean)
                        if onset_mean > onset_thresh:
                            predicted_class_onset.append(1)  # long
                        else:
                            predicted_class_onset.append(0)  # short
        f_score = f1_score(true_vals_onset, predicted_class_onset, average='weighted')
        print(onset_str_list[0] + ' f-score: ' + str(f_score))
        print('majority vote f-score:' + str(
            f1_score(true_vals_onset, np.zeros([len(true_vals_onset), ]).tolist(), average='weighted')))
        results_save['f_scores_' + onset_str_list[0]].append(f_score)
        if not(len(true_vals_onset) == 0):
            tn, fp, fn, tp = confusion_matrix(true_vals_onset, predicted_class_onset).ravel()
        else:
            tn,fp,fn,tp, = 0,0,0,0
        results_save['tn_' + onset_str_list[0]].append(tn)
        results_save['fp_' + onset_str_list[0]].append(fp)
        results_save['fn_' + onset_str_list[0]].append(fn)
        results_save['tp_' + onset_str_list[0]].append(tp)

    # get prediction at overlap f-scores

    for overlap_str in overlap_str_list:
        true_vals_overlap, predicted_class_overlap = [], []

        for conv_key in list(set(list(overlaps[overlap_str].keys())).intersection(set(test_file_list))):
            for g_f_key in list(overlaps[overlap_str + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for eval_indx, true_val in overlaps[overlap_str + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if eval_indx < len(results_dict[conv_key + '/' + g_f_key]):
                        true_vals_overlap.append(true_val)
                        if np.sum(results_dict[conv_key + '/' + g_f_key][eval_indx,
                                  eval_window_start_point: eval_window_start_point + eval_window_length]) \
                                > np.sum(results_dict[conv_key + '/' + g_f_key_not[0]][eval_indx,
                                         eval_window_start_point: eval_window_start_point + eval_window_length]):
                            predicted_class_overlap.append(0)
                        else:
                            predicted_class_overlap.append(1)
        f_score = f1_score(true_vals_overlap, predicted_class_overlap, average='weighted')
        print(overlap_str + ' f-score: ' + str(f_score))

        print('majority vote f-score:' + str(
            f1_score(true_vals_overlap, np.ones([len(true_vals_overlap), ]).tolist(), average='weighted')))
        results_save['f_scores_' + overlap_str].append(f_score)
        tn, fp, fn, tp = confusion_matrix(true_vals_overlap, predicted_class_overlap).ravel()
        results_save['tn_' + overlap_str].append(tn)
        results_save['fp_' + overlap_str].append(fp)
        results_save['fn_' + overlap_str].append(fn)
        results_save['tp_' + overlap_str].append(tp)
    # get error per person (to use with plot_person_error())
    bar_chart_labels = []
    bar_chart_vals = []
    for conv_key in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            losses_dict[conv_key + '/' + g_f] = np.array(losses_dict[conv_key + '/' + g_f]).reshape(-1,
                                                                                                    prediction_length)
            bar_chart_labels.append(conv_key + '_' + g_f)
            bar_chart_vals.append(np.mean(losses_dict[conv_key + '/' + g_f]))

    results_save['test_losses'].append(loss_weighted_mean)
    results_save['test_losses_l1'].append(loss_weighted_mean_l1)
    #    results_save['test_losses_mse'].append(loss_weighted_mean_mse)

    indiv_perf = {'bar_chart_labels': bar_chart_labels,
                  'bar_chart_vals': bar_chart_vals}
    results_save['indiv_perf'].append(indiv_perf)
    # majority baseline:
    # f1_score(true_vals,np.zeros([len(true_vals),]).tolist(),average='weighted')


if __name__ == "__main__":
    trial_path = './two_subnets_complete/1_Acous_50ms_Ling_50ms'
    test_path = f'{trial_path}/test'
    for directory in os.listdir(test_path):
        # paths to stored models, settings, and location for new results
        model_path = f'{test_path}/{directory}/model.p'
        settings_path = f'{test_path}/{directory}/settings.json'
        results_path = f"{trial_path}/test_on_both/{directory}"

        # remove existing directories (only needed for debugging purposes)
        try:
            os.rmdir(results_path)
        except FileNotFoundError:
            pass

        # load settings, model, data and create directories for results
        args = load_args(settings_path)
        test_set, test_loader = load_test_set(args, test_on_g=True, test_on_f=True)
        model = load_model(model_path, args, test_set)
        os.makedirs(results_path)

        # perform test on loaded model
        model.eval()
        test(model, test_set, test_loader)


