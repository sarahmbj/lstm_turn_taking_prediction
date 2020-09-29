from test_on_existing_model import test_on_existing_models

train_data = 'data'
test_data = 'data'

trial_path = './no_subnets/2_Acous_10ms'
for i in range(4):
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name=f'onset_stability{i}',
                            test_data_dir=test_data, train_data_dir=train_data)

trial_path = './no_subnets/3_Ling_50ms'
for i in range(4):
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name=f'onset_stability{i}',
                            test_data_dir=test_data, train_data_dir=train_data)

trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
for i in range(4):
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name=f'onset_stability{i}',
                            test_data_dir=test_data, train_data_dir=train_data)