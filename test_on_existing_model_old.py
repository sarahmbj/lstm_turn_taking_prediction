import torch
from lstm_model import LSTMPredictor
from data_loader import TurnPredictionDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import pandas as pd
import numpy as np
from copy import deepcopy
import h5py
from sklearn.metrics import f1_score, roc_curve, confusion_matrix
from pprint import pprint
import pickle
import shutil
import matplotlib.pyplot as plt


num_layers = 1
annotations_dir = './data/extracted_annotations/voice_activity/'
test_list_path = './data/splits/testing.txt'
train_list_path = './data/splits/training.txt' #only used for onsets evaluation
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


def load_evaluation_data():
    if 'hold_shift' in locals():
        if isinstance(hold_shift, h5py.File):  # Just HDF5 files
            try:
                hold_shift.close()
            except:
                pass  # Was already closed
    if 'onsets' in locals():
        if isinstance(onsets, h5py.File):  # Just HDF5 files
            try:
                onsets.close()
            except:
                pass  # Was already closed
    if 'overlaps' in locals():
        if isinstance(overlaps, h5py.File):  # Just HDF5 files
            try:
                overlaps.close()
            except:
                pass  # Was already closed

    hold_shift = h5py.File('./data/datasets/hold_shift.hdf5', 'r')
    onsets = h5py.File('./data/datasets/onsets.hdf5', 'r')
    overlaps = h5py.File('./data/datasets/overlaps.hdf5', 'r')

    return hold_shift, onsets, overlaps


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
    test_dataset = TurnPredictionDataset(args_dict['feature_dict_list'], annotations_dir, test_list_path,
                                         args_dict['sequence_length'], prediction_length, 'test',
                                         data_select=data_set_select, test_on_f=test_on_f, test_on_g=test_on_g)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                 pin_memory=p_memory)

    return test_dataset, test_dataloader


def load_training_set(args_dict, train_on_f=True, train_on_g=True):
    """needed to get the threshold value for prediction at onsets"""
    train_dataset = TurnPredictionDataset(args_dict['feature_dict_list'], annotations_dir, train_list_path,
                                          args_dict['sequence_length'], prediction_length, 'train',
                                          data_select=data_set_select, train_on_f=train_on_f, train_on_g=train_on_g)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0,
                                  drop_last=True, pin_memory=p_memory)
    return train_dataset, train_dataloader


def plot_person_error(name_list, data, results_path, results_key='barchart'):
    y_pos = np.arange(len(name_list))
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.barh(y_pos, data, align='center', alpha=0.5)
    plt.yticks(y_pos, name_list, fontsize=5)
    plt.xlabel('mean abs error per time frame', fontsize=7)
    plt.xticks(fontsize=7)
    plt.title('Individual Error')
    plt.savefig(results_path + '/' + results_key + '.pdf')


def test(model, test_dataset, test_dataloader, onset_test_flag=True):
    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    losses_mse, losses_l1 = [], []

    pause_str_list = ['50ms', '250ms', '500ms']
    overlap_str_list = ['overlap_hold_shift', 'overlap_hold_shift_exclusive']
    onset_str_list = ['short_long']
    results_save = dict()
    for pause_str in pause_str_list + overlap_str_list + onset_str_list:
        results_save['f_scores_' + pause_str] = list()
        results_save['tn_' + pause_str] = list()
        results_save['fp_' + pause_str] = list()
        results_save['fn_' + pause_str] = list()
        results_save['tp_' + pause_str] = list()
    results_save['train_losses'], results_save['test_losses'], results_save['indiv_perf'], results_save[
        'test_losses_l1'] = [], [], [], []

    # Decide whether to use cuda or not
    use_cuda = torch.cuda.is_available()
    print('Use CUDA: ' + str(use_cuda))
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        dtype_long = torch.LongTensor

    # initialise loss functions
    loss_func_L1 = nn.L1Loss()
    loss_func_L1_no_reduce = nn.L1Loss(reduce=False)
    loss_func_BCE = nn.BCELoss()

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

    hold_shift, onsets, overlaps = load_evaluation_data()

    # get hold-shift f-scores
    length_of_future_window = 20  # (1 second)

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
    train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])
    # if onset_test_flag: #TODO: fix this
    #     onset_train_true_vals = list()
    #     onset_train_mean_vals = list()
    #     onset_threshs = []
    #     for conv_key in list(set(train_file_list).intersection(onsets['short_long'].keys())):
    #         for g_f_key in list(onsets['short_long' + '/' + conv_key].keys()):
    #             g_f_key_not = deepcopy(data_select_dict[data_set_select])
    #             g_f_key_not.remove(g_f_key)
    #             for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
    #                 # make sure the index is not out of bounds
    #
    #                 if (frame_indx < len(train_results_dict[conv_key + '/' + g_f_key])) and not (
    #                         np.isnan(np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
    #                     onset_train_true_vals.append(true_val)
    #                     onset_train_mean_vals.append(
    #                         np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))
    #     if not(len(onset_train_true_vals) == 0):
    #         fpr, tpr, thresholds = roc_curve(np.array(onset_train_true_vals), np.array(onset_train_mean_vals))
    #     else:
    #         fpr, tpr, thresholds = 0, 0, [0]
    #     thresh_indx = np.argmax(tpr - fpr)
    #     onset_thresh = thresholds[thresh_indx]
    #     onset_threshs.append(onset_thresh)
    #
    #     true_vals_onset, onset_test_mean_vals, predicted_class_onset = [], [], []
    #     for conv_key in list(set(test_file_list).intersection(onsets['short_long'].keys())):
    #         for g_f_key in list(onsets['short_long' + '/' + conv_key].keys()):
    #             #                g_f_key_not = ['g','f']
    #             g_f_key_not = deepcopy(data_select_dict[data_set_select])
    #             g_f_key_not.remove(g_f_key)
    #             for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
    #                 # make sure the index is not out of bounds
    #                 if (frame_indx < len(results_dict[conv_key + '/' + g_f_key])) and not (
    #                 np.isnan(np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
    #                     true_vals_onset.append(true_val)
    #                     onset_mean = np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :])
    #                     onset_test_mean_vals.append(onset_mean)
    #                     if onset_mean > onset_thresh:
    #                         predicted_class_onset.append(1)  # long
    #                     else:
    #                         predicted_class_onset.append(0)  # short
    #     f_score = f1_score(true_vals_onset, predicted_class_onset, average='weighted')
    #     print(onset_str_list[0] + ' f-score: ' + str(f_score))
    #     print('majority vote f-score:' + str(
    #         f1_score(true_vals_onset, np.zeros([len(true_vals_onset), ]).tolist(), average='weighted')))
    #     results_save['f_scores_' + onset_str_list[0]].append(f_score)
    #     if not(len(true_vals_onset) == 0):
    #         tn, fp, fn, tp = confusion_matrix(true_vals_onset, predicted_class_onset).ravel()
    #     else:
    #         tn,fp,fn,tp, = 0,0,0,0
    #     results_save['tn_' + onset_str_list[0]].append(tn)
    #     results_save['fp_' + onset_str_list[0]].append(fp)
    #     results_save['fn_' + onset_str_list[0]].append(fn)
    #     results_save['tp_' + onset_str_list[0]].append(tp)

    # get prediction at overlap f-scores
    short_class_length = 20
    overlap_min = 2
    eval_window_start_point = short_class_length - overlap_min
    eval_window_length = 10
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
    return(results_save)


if __name__ == "__main__":
    trial_path = './two_subnets_complete/1_Acous_50ms_Ling_50ms'
    test_path = f'{trial_path}/test'

    # Loop through all the trained models in this trial path
    for directory in os.listdir(test_path):
        # paths to stored models, settings, and location for new results
        model_path = f'{test_path}/{directory}/model.p'
        settings_path = f'{test_path}/{directory}/settings.json'
        results_path = f"{trial_path}/test_on_both/{directory}"

        # remove existing directories (only needed for debugging purposes)
        try:
            shutil.rmtree(results_path)
        except FileNotFoundError:
            pass

        # load settings, model, data and create directories for results
        args = load_args(settings_path)
        test_set, test_loader = load_test_set(args, test_on_g=True, test_on_f=True)
        model = load_model(model_path, args, test_set)
        os.makedirs(results_path)

        # use training set to get threshold for onset evaluation
        train_set, train_loader = load_training_set(args, train_on_g=True, train_on_f=True) #TODO: needs to be what the original test set was
        print(type(train_set))
        quit()

        # perform test on loaded model
        model.eval()
        test_results = test(model, test_set, test_loader) #TODO: fix onset evaluation (needs train_results_dict)
        with open(results_path + '/results.txt', 'w') as file:
            file.write(str(test_results))
        pickle.dump(test_results, open(results_path + '/results.p', 'wb'))
        plot_person_error(test_results['indiv_perf'][-1]['bar_chart_labels'],
                          test_results['indiv_perf'][-1]['bar_chart_vals'], results_path,
                          results_key='person_error_barchart')

        # store metrics to average across trials
        total_num_epochs = len(test_results['test_losses'])
        best_loss_indx = np.argmin(test_results['test_losses'])
        eval_metric_list = ['f_scores_50ms', 'f_scores_250ms', 'f_scores_500ms', 'f_scores_overlap_hold_shift',
                            'f_scores_overlap_hold_shift_exclusive', 'f_scores_short_long', 'train_losses',
                            'test_losses', 'test_losses_l1']
    #     for eval_metric in eval_metric_list:
    #         best_vals_dict[eval_metric] += float(test_results[eval_metric][best_loss_indx]) * (
    #                     1.0 / float(len(test_indices)))
    #         last_vals_dict[eval_metric] += float(test_results[eval_metric][-1]) * (1.0 / float(len(test_indices)))
    #         best_vals_dict_array[eval_metric].append(float(test_results[eval_metric][best_loss_indx]))
    #         best_fscore_array[eval_metric].append(float(np.amax(test_results[eval_metric])))
    #
    # combine metrics across trials
    report_dict = {'experiment_name': experiment_name,
                   'best_vals': best_vals_dict,
                   'last_vals': last_vals_dict,
                   'best_vals_array': best_vals_dict_array,
                   'best_fscore_array': best_fscore_array,
                   'best_fscore_500_average': np.mean(best_fscore_array['f_scores_500ms']),
                   'best_test_loss_average': np.mean(best_vals_dict['test_losses']),
                   'best_indx': int(best_loss_indx),
                   'num_epochs_total': int(total_num_epochs),
                   'selected_lr': best_lr,
                   'selected_master_node_size': int(best_master_node_size)
                   }

    json.dump(report_dict, open(trial_path + '/report_dict.json', 'w'), indent=4, sort_keys=True)
        # TODO: need to do averaging across trials
        # TODO: do for each training set (f,g,both)
        # TODO: fix evaluation metrics for f v. g (is this needed?)


