from test_on_existing_model import test_on_existing_models

# These tests can be run on the models trained on the combined dataset
# no tests on single roles, as I have not trained these models for the combined dataset

train_data = 'data'
test_data = 'data'

# models trained on both roles
trial_path = './no_subnets/2_Acous_10ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)
test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)
test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, test_data_dir=test_data, train_data_dir=train_data)

trial_path = './no_subnets/3_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)
test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)
test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)
test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)
test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, test_data_dir=test_data, train_data_dir=train_data)

# cross corpus tests - acoustic only models
trial_path = './no_subnets/2_Acous_10ms'

train_data = 'data'
test_data = 'switchboard_data'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)

train_data = 'data'
test_data = 'maptask_data'
test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, test_data_dir=test_data, train_data_dir=train_data)
