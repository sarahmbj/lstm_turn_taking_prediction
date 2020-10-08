from scipy.stats import spearmanr
from scipy.stats import pearsonr

# to investigate the correlations between different metrics used in my results

# On Roddy results
# print("\n\n***** Roddy Results *****")
# BCE_loss_roddy = [0.5456, 0.5351, 0.5779, 0.5839, 0.5823, 0.5411, 0.5321, 0.542, 0.5291, 0.5416, 0.5296, 0.531]
# F1_50_roddy = [0.7907, 0.8154, 0.7234, 0.7101, 0.7072, 0.7957, 0.8194, 0.7916, 0.8323, 0.7949, 0.8307, 0.8285]
# F1_500_roddy = [0.8165, 0.8428, 0.7547, 0.7341, 0.7391, 0.8354, 0.8465, 0.8303, 0.8526, 0.8385, 0.8553, 0.847]
# F1_onset_roddy = [0.7926, 0.8126, 0.7249, 0.7174, 0.7111, 0.8101, 0.8141, 0.8019, 0.8236, 0.7993, 0.8232, 0.8189]

# print(spearmanr(BCE_loss_roddy, F1_onset_roddy))
# print(pearsonr(BCE_loss_roddy, F1_onset_roddy))
#
# print(spearmanr(F1_500_roddy, F1_onset_roddy))
# print(pearsonr(F1_500_roddy, F1_onset_roddy))
#
# print(spearmanr(F1_50_roddy, F1_onset_roddy))
# print(pearsonr(F1_50_roddy, F1_onset_roddy))
#
# print(spearmanr(F1_50_roddy, BCE_loss_roddy))
# print(pearsonr(F1_50_roddy, BCE_loss_roddy))
#
# print(spearmanr(F1_500_roddy, BCE_loss_roddy))
# print(pearsonr(F1_500_roddy, BCE_loss_roddy))
#
# print(spearmanr(F1_50_roddy, F1_500_roddy))
# print(pearsonr(F1_50_roddy, F1_500_roddy))

# On Maptask results
# print("\n\n***** Maptask Results *****")
# BCE_loss_maptask = [0.5465, 0.5376, 0.5783, 0.5785, 0.5824, 0.5431, 0.5358, 0.5419, 0.5311, 0.5463, 0.5313, 0.5316]
# F1_50_maptask = [0.7678, 0.8056, 0.7116, 0.7137, 0.6912, 0.7801, 0.7992, 0.7804, 0.8188, 0.7685, 0.8203, 0.8205]
# F1_500_maptask = [0.8125, 0.8282, 0.7522, 0.7441, 0.7252, 0.826, 0.8325, 0.823, 0.8417, 0.8111, 0.8437, 0.8381]
# F1_onset_maptask = [0.768, 0.7897, 0.6885, 0.697, 0.693, 0.7883, 0.7922, 0.7873, 0.8014, 0.7724, 0.7935, 0.7975]

# print(spearmanr(BCE_loss_maptask, F1_onset_maptask))
# print(pearsonr(BCE_loss_maptask, F1_onset_maptask))
#
# print(spearmanr(F1_500_maptask, F1_onset_maptask))
# print(pearsonr(F1_500_maptask, F1_onset_maptask))
#
# print(spearmanr(F1_50_maptask, F1_onset_maptask))
# print(pearsonr(F1_50_maptask, F1_onset_maptask))

# print(spearmanr(F1_50_maptask, BCE_loss_maptask))
# print(pearsonr(F1_50_maptask, BCE_loss_maptask))
#
# print(spearmanr(F1_500_maptask, BCE_loss_maptask))
# print(pearsonr(F1_500_maptask, BCE_loss_maptask))
#
# print(spearmanr(F1_50_maptask, F1_500_maptask))
# print(pearsonr(F1_50_maptask, F1_500_maptask))

# On Switchboard results
# print("\n\n***** Switchboard Results *****")
# BCE_loss_switchboard = [0.4818, 0.4717, 0.5032, 0.5088, 0.5072, 0.4774, 0.4705, 0.4746, 0.4653, 0.4802, 0.4679, 0.4694]
# F1_50_switchboard = [0.829, 0.8577, 0.8191, 0.8175, 0.8159, 0.8445, 0.8633, 0.848, 0.8703, 0.8373, 0.867, 0.8665]
# F1_500_switchboard = [0.8814, 0.8986, 0.8485, 0.8444, 0.8498, 0.8758, 0.896, 0.8897, 0.9046, 0.8373, 0.9017, 0.9014]
# F1_onset_switchboard = [0.7916, 0.7981, 0.7692, 0.7576, 0.7711, 0.8151, 0.8031, 0.8277, 0.8491, 0.798, 0.8414, 0.8204]

# print(spearmanr(BCE_loss_switchboard, F1_onset_switchboard))
# print(pearsonr(BCE_loss_switchboard, F1_onset_switchboard))
#
# print(spearmanr(F1_500_switchboard, F1_onset_switchboard))
# print(pearsonr(F1_500_switchboard, F1_onset_switchboard))
#
# print(spearmanr(F1_50_switchboard, F1_onset_switchboard))
# print(pearsonr(F1_50_switchboard, F1_onset_switchboard))

# print(spearmanr(F1_50_switchboard, BCE_loss_switchboard))
# print(pearsonr(F1_50_switchboard, BCE_loss_switchboard))
#
# print(spearmanr(F1_500_switchboard, BCE_loss_switchboard))
# print(pearsonr(F1_500_switchboard, BCE_loss_switchboard))
#
# print(spearmanr(F1_50_switchboard, F1_500_switchboard))
# print(pearsonr(F1_50_switchboard, F1_500_switchboard))

# On Combined results
# print("\n\n***** Combined Results *****")
# BCE_loss_combined = [0.5235, 0.5132, 0.5517, 0.552, 0.5566, 0.5172, 0.5087, 0.5179, 0.5077, 0.5212, 0.5089, 0.5088]
# F1_50_combined = [0.7852, 0.8213, 0.7475, 0.7485, 0.7206, 0.8044, 0.8285, 0.8016, 0.8361, 0.7957, 0.8288, 0.8048]
# F1_500_combined = [0.8295, 0.8506, 0.7781, 0.7814, 0.7582, 0.848, 0.8581, 0.8417, 0.8607, 0.8331, 0.8614, 0.8592]
# F1_onset_combined = [0.7724, 0.7939, 0.7167, 0.7105, 0.7038, 0.8015, 0.8047, 0.8031, 0.8137, 0.7744, 0.8029, 0.8048]

# print(spearmanr(BCE_loss_combined, F1_onset_combined))
# print(pearsonr(BCE_loss_combined, F1_onset_combined))
#
# print(spearmanr(F1_500_combined, F1_onset_combined))
# print(pearsonr(F1_500_combined, F1_onset_combined))
#
# print(spearmanr(F1_50_combined, F1_onset_combined))
# print(pearsonr(F1_50_combined, F1_onset_combined))

# print(spearmanr(F1_50_combined, BCE_loss_combined))
# print(pearsonr(F1_50_combined, BCE_loss_combined))
#
# print(spearmanr(F1_500_combined, BCE_loss_combined))
# print(pearsonr(F1_500_combined, BCE_loss_combined))
#
# print(spearmanr(F1_50_combined, F1_500_combined))
# print(pearsonr(F1_50_combined, F1_500_combined))

# For Roddy reproduction experiment:
# print("\n\n***** RODDY REPRODUCION *****")
# print(spearmanr(BCE_loss_roddy, BCE_loss_maptask))
# print(pearsonr(BCE_loss_roddy, BCE_loss_maptask))
#
# print(spearmanr(F1_50_roddy, F1_50_maptask))
# print(pearsonr(F1_50_roddy, F1_50_maptask))
#
# print(spearmanr(F1_500_roddy, F1_500_maptask))
# print(pearsonr(F1_500_roddy, F1_500_maptask))
#
# print(spearmanr(F1_onset_roddy, F1_onset_maptask))
# print(pearsonr(F1_onset_roddy, F1_onset_maptask))

# # For Roddy reproduction experiment:
# print("\n\n***** RODDY v. Switchboard *****")
# print(spearmanr(BCE_loss_roddy, BCE_loss_switchboard))
# print(pearsonr(BCE_loss_roddy, BCE_loss_switchboard))
#
# print(spearmanr(F1_50_roddy, F1_50_switchboard))
# print(pearsonr(F1_50_roddy, F1_50_switchboard))
#
# print(spearmanr(F1_500_roddy, F1_500_switchboard))
# print(pearsonr(F1_500_roddy, F1_500_switchboard))
#
# print(spearmanr(F1_onset_roddy, F1_onset_switchboard))
# print(pearsonr(F1_onset_roddy, F1_onset_switchboard))
#
# # For Roddy reproduction experiment:
# print("\n\n***** RODDY v. combined *****")
# print(spearmanr(BCE_loss_roddy, BCE_loss_combined))
# print(pearsonr(BCE_loss_roddy, BCE_loss_combined))
#
# print(spearmanr(F1_50_roddy, F1_50_combined))
# print(pearsonr(F1_50_roddy, F1_50_combined))
#
# print(spearmanr(F1_500_roddy, F1_500_combined))
# print(pearsonr(F1_500_roddy, F1_500_combined))
#
# print(spearmanr(F1_onset_roddy, F1_onset_combined))
# print(pearsonr(F1_onset_roddy, F1_onset_combined))
#
# print("\n\n***** Maptask v. Switchboard *****")
# print(spearmanr(BCE_loss_switchboard, BCE_loss_maptask))
# print(pearsonr(BCE_loss_switchboard, BCE_loss_maptask))
#
# print(spearmanr(F1_50_switchboard, F1_50_maptask))
# print(pearsonr(F1_50_switchboard, F1_50_maptask))
#
# print(spearmanr(F1_500_switchboard, F1_500_maptask))
# print(pearsonr(F1_500_switchboard, F1_500_maptask))
#
# print(spearmanr(F1_onset_switchboard, F1_onset_maptask))
# print(pearsonr(F1_onset_switchboard, F1_onset_maptask))
#
# print("\n\n***** Maptask v. Combined *****")
# print(spearmanr(BCE_loss_combined, BCE_loss_maptask))
# print(pearsonr(BCE_loss_combined, BCE_loss_maptask))
#
# print(spearmanr(F1_50_combined, F1_50_maptask))
# print(pearsonr(F1_50_combined, F1_50_maptask))
#
# print(spearmanr(F1_500_combined, F1_500_maptask))
# print(pearsonr(F1_500_combined, F1_500_maptask))
#
# print(spearmanr(F1_onset_combined, F1_onset_maptask))
# print(pearsonr(F1_onset_combined, F1_onset_maptask))
#
# print("\n\n***** Switchboard v. Combined *****")
# print(spearmanr(BCE_loss_combined, BCE_loss_switchboard))
# print(pearsonr(BCE_loss_combined, BCE_loss_switchboard))
#
# print(spearmanr(F1_50_combined, F1_50_switchboard))
# print(pearsonr(F1_50_combined, F1_50_switchboard))
#
# print(spearmanr(F1_500_combined, F1_500_switchboard))
# print(pearsonr(F1_500_combined, F1_500_switchboard))
#
# print(spearmanr(F1_onset_combined, F1_onset_switchboard))
# print(pearsonr(F1_onset_combined, F1_onset_switchboard))

# data-set here refers to what the model was trained on
print("\n\n***** F1 Onset *****")
F1_onset_cross_corp_maptask = [0.7923, 0.6646, 0.8009, 0.6936, 0.6121, 0.7479, 0.7538, 0.6475, 0.7782]
F1_onset_cross_corp_switchboard = [0.68, 0.5834, 0.702, 0.7981, 0.7692, 0.8491, 0.7293, 0.6593, 0.7604]
F1_onset_cross_corp_combined = [0.786, 0.654, 0.7894, 0.8037, 0.7954, 0.8757, 0.7939, 0.7167, 0.8137]

print(spearmanr(F1_onset_cross_corp_maptask, F1_onset_cross_corp_switchboard))
print(pearsonr(F1_onset_cross_corp_maptask, F1_onset_cross_corp_switchboard))

print(spearmanr(F1_onset_cross_corp_maptask, F1_onset_cross_corp_combined))
print(pearsonr(F1_onset_cross_corp_maptask, F1_onset_cross_corp_combined))

print(spearmanr(F1_onset_cross_corp_switchboard, F1_onset_cross_corp_combined))
print(pearsonr(F1_onset_cross_corp_switchboard, F1_onset_cross_corp_combined))

print("\n\n***** avg F1 *****")
avg_F1_cross_corp_maptask = [0.8098, 0.7017, 0.8237, 0.7211, 0.734, 0.7654, 0.7732, 0.7105, 0.7954]
avg_F1_cross_corp_switchboard = [0.7005, 0.6004, 0.7162, 0.8515, 0.8123, 0.8747, 0.754, 0.6754, 0.7723]
avg_F1_cross_corp_combined = [0.8021, 0.584, 0.6272, 0.8528, 0.8243, 0.8266, 0.8219, 0.7474, 0.8368]

print(spearmanr(avg_F1_cross_corp_maptask, avg_F1_cross_corp_switchboard))
print(pearsonr(avg_F1_cross_corp_maptask, avg_F1_cross_corp_switchboard))

print(spearmanr(avg_F1_cross_corp_maptask, avg_F1_cross_corp_combined))
print(pearsonr(avg_F1_cross_corp_maptask, avg_F1_cross_corp_combined))

print(spearmanr(avg_F1_cross_corp_switchboard, avg_F1_cross_corp_combined))
print(pearsonr(avg_F1_cross_corp_switchboard, avg_F1_cross_corp_combined))

print("\n\n***** BCE Loss *****")
loss_cross_corp_maptask = [0.5358, 0.5794, 0.5291, 0.6179, 0.5938, 0.5842, 0.5657, 0.5836, 0.5493]
loss_cross_corp_switchboard = [0.6039, 0.635, 0.6001, 0.4717, 0.5032, 0.4653, 0.5547, 0.5874, 0.5495]
loss_cross_corp_combined = [0.5373, 0.668, 0.6167, 0.4724, 0.5118, 0.4706, 0.5132, 0.5517, 0.5077]

print(spearmanr(loss_cross_corp_maptask, loss_cross_corp_switchboard))
print(pearsonr(loss_cross_corp_maptask, loss_cross_corp_switchboard))

print(spearmanr(loss_cross_corp_maptask, loss_cross_corp_combined))
print(pearsonr(loss_cross_corp_maptask, loss_cross_corp_combined))

print(spearmanr(loss_cross_corp_switchboard, loss_cross_corp_combined))
print(pearsonr(loss_cross_corp_switchboard, loss_cross_corp_combined))
