from test_on_existing_model import test_on_existing_models

# cross corpus tests - acoustic only models
trial_path = './no_subnets/2_Acous_10ms'

train_data = 'data'
test_data = 'switchboard_data'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)

train_data = 'data'
test_data = 'both_data'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)


trial_path = './f_and_g_no_subnets/4_Acous_10ms_gtrain'

train_data = 'data'
test_data = 'switchboard_data'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True,
                        test_data_dir=test_data, train_data_dir=train_data)

train_data = 'data'
test_data = 'both_data'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True,
                        test_data_dir=test_data, train_data_dir=train_data)

trial_path = './f_and_g_no_subnets/3_Acous_10ms_ftrain'

train_data = 'data'
test_data = 'switchboard_data'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False,
                        test_data_dir=test_data, train_data_dir=train_data)

train_data = 'data'
test_data = 'both_data'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False,
                        test_data_dir=test_data, train_data_dir=train_data)


# models trained on both roles
# train_data = 'data'
# test_data = 'data'
# trial_path = './no_subnets/3_Ling_50ms'
# test_on_existing_models(trial_path, test_data_dir=test_data, train_data_dir=train_data,
#                         test_on_f=True, test_on_g=True, report_dict_name=f'test_on_{test_data}')
# trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
# test_on_existing_models(trial_path, test_data_dir=test_data, train_data_dir=train_data,
#                         test_on_f=True, test_on_g=True, report_dict_name=f'test_on_{test_data}')


train_data = 'data'
test_data = 'switchboard_data'
trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_data_dir=test_data, train_data_dir=train_data,
                        test_on_f=True, test_on_g=True, report_dict_name=f'test_on_{test_data}')
trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_data_dir=test_data, train_data_dir=train_data,
                        test_on_f=True, test_on_g=True, report_dict_name=f'test_on_{test_data}')

train_data = 'data'
test_data = 'both_data'
trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_data_dir=test_data, train_data_dir=train_data,
                        test_on_f=True, test_on_g=True, report_dict_name=f'test_on_{test_data}')
trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_data_dir=test_data, train_data_dir=train_data,
                        test_on_f=True, test_on_g=True, report_dict_name=f'test_on_{test_data}')

# could also add models trained/tested on one role only #TODO: decide whether to do this