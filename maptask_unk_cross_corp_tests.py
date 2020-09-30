from test_on_existing_model import test_on_existing_models

#ran succesfully:


#still to run:

# train_data = 'data' #TODO: re-run with correct onset_thresh (SHORT_LONG SHOULD CHANGE)
# test_data = 'both_data'
# trial_path = './no_subnets/2_Acous_10ms'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data,
#                         report_dict_name="test_on_combined_data")
#
# train_data = 'data'  #TODO: re-run with correct onset_thresh (SHORT_LONG SHOULD CHANGE)
# test_data = 'switchboard_data'
# trial_path = './no_subnets/3_Ling_50ms'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data,
#                         report_dict_name="test_on_switchboard_data")

# train_data = 'data'  #TODO: re-run with correct onset_thresh (SHORT_LONG SHOULD CHANGE)
# test_data = 'switchboard_data'
# trial_path = './no_subnets/2_Acous_10ms'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data,
#                         report_dict_name="test_on_switchboard_data")

# train_data = 'data'  # TODO: re-run with correct onset_thresh (SHORT_LONG SHOULD CHANGE)
# test_data = 'switchboard_data'
# trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data,
#                         report_dict_name="test_on_switchboard_data")

# train_data = 'data'  # TODO: fix index out of range error
# test_data = 'both_data'
# trial_path = './no_subnets/3_Ling_50ms'
# test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data,
#                         report_dict_name="test_on_combined_data")
train_data = 'data'  # TODO: fix index out of range error (same as above)
test_data = 'both_data'
trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data,
                        report_dict_name="test_on_combined_data")




