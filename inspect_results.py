# to load in pickled results files and inspect them
import pickle
import numpy as np
from pprint import pprint


file_to_inspect = "./results/sample_results/results.p"

with open(file_to_inspect, "rb") as results_file:
    results = pickle.load(results_file)

print(len(results))
print(type(results))
print(results.keys())

# print(results['indiv_perf']) # a list of dictionaries - is it a dictionary for each run of the experiment?
print(results['indiv_perf'][0]) # these are the labels and values for the per speaker bar chart - mean absolute error per speaker
# would need to decide best epoch, and then get the mean error for f and for g by aggregating across all speakers
#but do you want to report other values e.g. f-scores for things like in roddy paper? that wont be available here
# would really like to do error curve - but that is tricky (is the reporting at different delays a less granular version of that?

print(results['indiv_perf'][0]['bar_chart_vals'])

both_errors = results['indiv_perf'][0]['bar_chart_vals']
f_errors = []  # 0,2,4,6 etc.
g_errors = []  # 1,3,5,7 etc.

for index, error in enumerate(results['indiv_perf'][0]['bar_chart_vals']):
    if index % 2 == 0:
        f_errors.append(error)
    else:
        g_errors.append(error)

mean_f_error = sum(f_errors)/len(f_errors)
mean_g_error = sum(g_errors)/len(g_errors)
mean_error = sum(both_errors)
std_f_error = np.std(f_errors)
std_g_error = np.std(g_errors)
std_error = np.std(both_errors)

print(f'average error on f: {mean_f_error} standard deviation: {std_f_error}')
print(f'average error on f: {mean_g_error} standard deviation: {std_g_error}')
print(f'average error on f: {mean_error} standard deviation: {std_error}')
print(f'Difference between average error f and g: {abs(mean_f_error-mean_g_error)}')
