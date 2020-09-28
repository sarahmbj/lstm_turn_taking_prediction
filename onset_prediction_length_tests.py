from test_on_existing_model import test_on_existing_models

# can run on maptask and switchboard
# remove acous only tests to make a version to run on unk
# remove role tests to make a version to run on combined data set

train_data = 'data'
test_data = 'data'

#to get results with different onset prediction times:
# onset_prediction_length = [0, 21]  # default is [0, 60] (60 frames is 3 seconds)
# trial_path = './no_subnets/2_Acous_10ms'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_both_roles',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

# trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_test_on_both_roles',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)
#
# trial_path = './no_subnets/3_Ling_50ms'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_test_on_both_roles',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_20_test_on_f,',
#                         test_data_dir=test_data, train_data_dir=train_data)

onset_prediction_length = [20, 41]  # default is [0, 60] (60 frames is 3 seconds)
trial_path = './no_subnets/2_Acous_10ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_20_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_20_40_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_20_40_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_20_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_20_40_test_on_bg',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_20_40_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_20_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_20_40_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_20_40_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

onset_prediction_length = [40, 61]  # default is [0, 60] (60 frames is 3 seconds)
trial_path = './no_subnets/2_Acous_10ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_40_60_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_40_60_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_40_60_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_40_60_test_on_both_roles')
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_40_60_test_on_g')
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_40_60_test_on_f')

trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_40_60_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_40_60_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_40_60_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

onset_prediction_length = [0, 41]  # default is [0, 60] (60 frames is 3 seconds)
trial_path = './no_subnets/2_Acous_10ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_40_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_40_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_40_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_40_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length,
                        report_dict_name='onset_0_40_test_on_both_roles',
                        test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_40_test_on_g',
#                         test_data_dir=test_data, train_data_dir=train_data)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length,
#                         report_dict_name='onset_0_40_test_on_f',
#                         test_data_dir=test_data, train_data_dir=train_data)

# trial_path = './f_and_g_no_subnets/3_Acous_10ms_ftrain'
# TODO: add report_dict_name to prevent saving over previous results, if using these
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
#
# trial_path = './f_and_g_no_subnets/5_Ling_50ms_ftrain'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
#
# trial_path = './f_and_g_no_subnets/4_Acous_10ms_gtrain'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)
#
# trial_path = './f_and_g_no_subnets/6_Ling_50ms_gtrain'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)
#
# trial_path = './f_and_g_two_subnets/1_Acous_10ms_Ling_50ms_ftrain'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False,
#                         onset_prediction_frames=onset_prediction_length)
#
# trial_path = './f_and_g_two_subnets/2_Acous_10ms_Ling_50ms_gtrain'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=False, trained_on_g=True,
#                         onset_prediction_frames=onset_prediction_length)