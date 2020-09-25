from test_on_existing_model import test_on_existing_models

# can run on maptask and switchboard
# remove acous only tests to make a version to run on unk
# remove role tests to make a version to run on combined data set

train_data = 'data'
test_data = 'data'

#to get results with different onset prediction times:
onset_prediction_length = [0, 21]  # default is [0, 60] (60 frames is 3 seconds)
trial_path = './no_subnets/2_Acous_10ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_20_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_20_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_20_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

onset_prediction_length = [20, 41]  # default is [0, 60] (60 frames is 3 seconds)
trial_path = './no_subnets/2_Acous_10ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_20_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_20_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_20_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

onset_prediction_length = [40, 61]  # default is [0, 60] (60 frames is 3 seconds)
trial_path = './no_subnets/2_Acous_10ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_40_60_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_40_60_test_on_both_roles')

trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_40_60_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

onset_prediction_length = [0, 41]  # default is [0, 60] (60 frames is 3 seconds)
trial_path = './no_subnets/2_Acous_10ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)

trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)