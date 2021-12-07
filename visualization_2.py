import os
import data_utils as utils

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


""" 
########## Setting Variable Paths ########## 
"""
# Variable paths
dataset_base_path = '/home/li.zhiyuan/Desktop/Probabilistic Inference/Project/dataset/'
wesad_dataset = os.path.join(dataset_base_path, 'wesad', 'WESAD')


""""
########## Reading pickle files for all subjects ##########
"""
all_subj_pkl = utils.read_all_subj_pkl(wesad_dataset)  # a list of 15 dictionaries (one for each subject)


"""
########## Getting individual classes: baseline, stress, amusement, meditation ##########
"""
baseline_data = utils.get_class_from_data_removed_features(all_subj_pkl, 'baseline')
print("length of baseline_data : {} of type {}".format(len(baseline_data), type(baseline_data)))
stress_data = utils.get_class_from_data_removed_features(all_subj_pkl, 'stress')
print("length of stress_data : {}".format(len(stress_data)))
amusement_data = utils.get_class_from_data_removed_features(all_subj_pkl, 'amusement')
print("length of amusement_data : {}".format(len(amusement_data)))
meditation_data = utils.get_class_from_data_removed_features(all_subj_pkl, 'meditation')
print("length of meditation_data : {}".format(len(meditation_data)))


"""
########## Plotting bar graph for all 4 classes ##########
"""
# fig = plt.figure()
# x_labels = ["baseline", "stress", "amusement", "meditation"]
# y_ax = [len(baseline_data), len(stress_data), len(amusement_data), len(meditation_data)]
# plt.bar(x_labels, y_ax, color='maroon', width=0.45)
# plt.xlabel("Classes")
# plt.ylabel("Instances")
# plt.title("Number of instances per class")
# fig.savefig('../output/classes_distributions.png')
# plt.show()


"""
########## Plotting bar graph for each device ##########
"""
# fig = plt.figure()
# x_labels = ["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp", "w_ax", "w_ay", "w_az", "w_bvp", "w_eda", "w_temp"]
# y_ax = utils.get_chest_and_wrist(all_subj_pkl, leng=True)
# plt.bar(x_labels, y_ax, color='blue', width=0.45)
# plt.xlabel("Devices")
# plt.xticks(rotation=40)
# plt.ylabel("Instances")
# plt.title("Number of instances per device")
# fig.savefig('../output/devices_distribution.png')
# plt.show()


"""
########## Correlation matrix for the devices (heatmap) ##########
"""
# ddf = utils.get_chest_and_wrist(all_subj_pkl, leng=False)
# print("df shape {}, and type {}".format(ddf.shape, type(ddf)))
#
# col = ["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# df = pd.DataFrame(data=ddf, columns=col)
#
# dev_corr = df.corr()
# plt.figure(figsize=(10, 10))
# dev_heatmap = sb.heatmap(dev_corr, annot=True, cmap=plt.cm.Reds)
# plt.title("Correlation heatmap for the devices")
# fig = dev_heatmap.get_figure()
# fig.savefig("../output/device_heatmap.png")
# plt.show()


"""
########## Correlation matrix for the baseline and stress (heatmap) ##########
"""
# new_baseline = baseline_data[:len(stress_data)]
# new_stress = stress_data[:, len(stress_data)//2]
# numpy_data = np.array([new_baseline, stress_data])
# numpy_data = numpy_data.T

# col = ["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# df = pd.DataFrame(data=numpy_data, columns=col)

# class_corr = np.corrcoef(new_baseline, new_stress)
# plt.figure(figsize=(20, 20))
# dev_heatmap = sb.heatmap(class_corr, annot=True, cmap=plt.cm.Reds)
# plt.title("Correlation heatmap between baseline and stress")
# fig = dev_heatmap.get_figure()
# fig.savefig("../output/baseline_stress_heatmap.png")

# import scipy
from scipy.stats import entropy

#
# kl_div = entropy(new_baseline, stress_data)
# kl_div = scipy.special.kl_div(new_baseline, stress_data)
# print(kl_div)


"""
########## Classes amplitude plot ##########
"""

"""
########## Cleaning the data - removing outliers ##########
"""
df_baseline = pd.DataFrame(data=baseline_data,
                           columns=["c_ax", "c_ay", "c_az", "c_eda", "c_temp"])
print("Baseline train df shape = {}".format(df_baseline.shape))

df_stress = pd.DataFrame(data=stress_data,
                                 columns=["c_ax", "c_ay", "c_az", "c_eda", "c_temp"])
print("Stress train df shape = {}".format(df_stress.shape))

# Calculate the baseline IQR
baseline_q1 = df_baseline.quantile(0.25)
baseline_q3 = df_baseline.quantile(0.75)
baseline_iqr = baseline_q3 - baseline_q1
print("Baseline train IQR is {}".format(baseline_iqr))

stress_q1 = df_stress.quantile(0.25)
stress_q3 = df_stress.quantile(0.75)
stress_iqr = stress_q3 - stress_q1
print("Stress train IQR is {}".format(stress_iqr))

# Removing baseline outliers
df_baseline_clean = df_baseline[
    ~((df_baseline < (baseline_q1 - 1.5 * baseline_iqr)) | (df_baseline > (baseline_q3 + 1.5 * baseline_iqr))).any(
        axis=1)]
print("Baseline df train cleaned shape = {}".format(df_baseline_clean.shape))

df_stress_clean = df_stress[
    ~((df_stress < (stress_q1 - 1.5 * stress_iqr)) | (
                df_stress > (stress_q3 + 1.5 * stress_iqr))).any(
        axis=1)]
print("Stress df train cleaned shape = {}".format(df_stress_clean.shape))

# Convert the dataframe to list
print("Converting df_baseline_train_clean to list")
baseline_data_train_clean = utils.df_to_list_removed_features(df_baseline_clean)
print("Converting df_stress_train_clean to list")
stress_data_train_clean = utils.df_to_list_removed_features(df_stress_clean)


# fig1 = plt.figure(figsize=(30, 15))

# y_base, y_stress, y_amus, y_medi, x_leng = utils.get_classes_per_device(baseline_data_train_clean, stress_data_train_clean, amusement_data, meditation_data, "c_ax")

# plt.plot(x_leng[:50000], y_base[:50000], color="black", label="acc X base", linewidth=1)
# plt.plot(x_leng[:50000], y_stress[:50000], color="red", label="acc X stress", linewidth=1)
# # plt.plot(x_leng, y_amus, color="blue", label="acc X amusement", linewidth=1)
# # plt.plot(x_leng, y_medi, color="green", label="acc X meditation", linewidth=1)
# plt.xlabel("Number of samples", fontsize="xx-large")
# plt.ylabel("Sensor amplitude", fontsize="xx-large")
# plt.title("AX base vs stress", fontsize="xx-large")
# plt.grid(b=True)
# plt.legend(loc="upper left")

# fig1.savefig("../output/clean_data_visualization/ax_class_distribution.png")
# plt.show()

# fig2 = plt.figure(figsize=(30, 15))

# y_base, y_stress, y_amus, y_medi, x_leng = utils.get_classes_per_device(baseline_data_train_clean, stress_data_train_clean, amusement_data, meditation_data, "c_ay")

# plt.plot(x_leng, y_base, color="black", label="acc Y base", linewidth=1)
# plt.plot(x_leng, y_stress, color="red", label="acc Y stress", linewidth=1)
# # plt.plot(x_leng, y_amus, color="blue", label="acc Y amusement", linewidth=1)
# # plt.plot(x_leng, y_medi, color="green", label="acc Y meditation", linewidth=1)
# plt.xlabel("Number of samples", fontsize="xx-large")
# plt.ylabel("Sensor amplitude", fontsize="xx-large")
# plt.title("AY base vs stress", fontsize="xx-large")
# plt.grid(b=True)
# plt.legend(loc="upper left")

# fig2.savefig("../output/clean_data_visualization/ay_class_distribution.png")
# plt.show()

# fig3 = plt.figure(figsize=(30, 15))

# y_base, y_stress, y_amus, y_medi, x_leng = utils.get_classes_per_device(baseline_data_train_clean, stress_data_train_clean, amusement_data, meditation_data, "c_az")

# plt.plot(x_leng, y_base, color="black", label="acc Z base", linewidth=1)
# plt.plot(x_leng, y_stress, color="red", label="acc Z stress", linewidth=1)
# # plt.plot(x_leng, y_amus, color="blue", label="acc Z amusement", linewidth=1)
# # plt.plot(x_leng, y_medi, color="green", label="acc Z meditation", linewidth=1)
# plt.xlabel("Number of samples", fontsize="xx-large")
# plt.ylabel("Sensor amplitude", fontsize="xx-large")
# plt.title("AZ base vs stress", fontsize="xx-large")
# plt.grid(b=True)
# plt.legend(loc="upper left")

# fig3.savefig("../output/clean_data_visualization/az_class_distribution.png")
# plt.show()

# fig4 = plt.figure(figsize=(30, 15))

# y_base, y_stress, y_amus, y_medi, x_leng = utils.get_classes_per_device(baseline_data_train_clean, stress_data_train_clean, amusement_data, meditation_data, "c_ecg")

# plt.plot(x_leng[:5000], y_base[:5000], color="black", label="ecg base", linewidth=0.5)
# plt.plot(x_leng[:5000], y_stress[:5000], color="red", label="ecg stress", linewidth=0.5)
# # plt.plot(x_leng, y_amus, color="blue", label="ecg amusement", linewidth=1)
# # plt.plot(x_leng, y_medi, color="green", label="ecg meditation", linewidth=1)
# plt.xlabel("Number of samples", fontsize="xx-large")
# plt.ylabel("Sensor amplitude", fontsize="xx-large")
# plt.title("ECG base vs stress", fontsize="xx-large")
# plt.grid(b=True)
# plt.legend(loc="upper left")

# fig4.savefig("../output/clean_data_visualization/ecg_class_distribution.png")
# plt.show()

# fig5 = plt.figure(figsize=(30, 15))

# y_base, y_stress, y_amus, y_medi, x_leng = utils.get_classes_per_device(baseline_data_train_clean, stress_data_train_clean, amusement_data, meditation_data, "c_emg")

# plt.plot(x_leng[:50000], y_base[:50000], color="black", label="emg base", linewidth=0.1)
# plt.plot(x_leng[:50000], y_stress[:50000], color="red", label="emg stress", linewidth=0.1)
# # plt.plot(x_leng, y_amus, color="blue", label="emg amusement", linewidth=1)
# # plt.plot(x_leng, y_medi, color="green", label="emg meditation", linewidth=1)
# plt.xlabel("Number of samples", fontsize="xx-large")
# plt.ylabel("Sensor amplitude", fontsize="xx-large")
# plt.title("EMG base vs stress", fontsize="xx-large")
# plt.grid(b=True)
# plt.legend(loc="upper left")

# fig5.savefig("../output/clean_data_visualization/emg_class_distribution.png")
# plt.show()

# fig6 = plt.figure(figsize=(30, 15))

# y_base, y_stress, y_amus, y_medi, x_leng = utils.get_classes_per_device(baseline_data_train_clean, stress_data_train_clean, amusement_data, meditation_data, "c_eda")

# plt.plot(x_leng[:50000], y_base[:50000], color="black", label="eda base", linewidth=1)
# plt.plot(x_leng[:50000], y_stress[:50000], color="red", label="eda stress", linewidth=1)
# # plt.plot(x_leng, y_amus, color="blue", label="eda amusement", linewidth=1)
# # plt.plot(x_leng, y_medi, color="green", label="eda meditation", linewidth=1)
# plt.xlabel("Number of samples", fontsize="xx-large")
# plt.ylabel("Sensor amplitude", fontsize="xx-large")
# plt.title("EDA base vs stress", fontsize="xx-large")
# plt.grid(b=True)
# plt.legend(loc="upper left")

# fig6.savefig("../output/clean_data_visualization/eda_class_distribution.png")
# plt.show()

# fig7 = plt.figure(figsize=(30, 15))

# y_base, y_stress, y_amus, y_medi, x_leng = utils.get_classes_per_device(baseline_data_train_clean, stress_data_train_clean, amusement_data, meditation_data, "c_temp")

# plt.plot(x_leng[:50000], y_base[:50000], color="black", label="temp base", linewidth=1)
# plt.plot(x_leng[:50000], y_stress[:50000], color="red", label="temp stress", linewidth=1)
# # plt.plot(x_leng, y_amus, color="blue", label="temp amusement", linewidth=1)
# # plt.plot(x_leng, y_medi, color="green", label="temp meditation", linewidth=1)
# plt.xlabel("Number of samples", fontsize="xx-large")
# plt.ylabel("Sensor amplitude", fontsize="xx-large")
# plt.title("TEMP base vs stress", fontsize="xx-large")
# plt.grid(b=True)
# plt.legend(loc="upper left")

# fig7.savefig("../output//clean_data_visualization/temp_class_distribution.png")
# plt.show()

# fig8 = plt.figure(figsize=(30, 15))

# y_base, y_stress, y_amus, y_medi, x_leng = utils.get_classes_per_device(baseline_data_train_clean, stress_data_train_clean, amusement_data, meditation_data, "c_resp")

# plt.plot(x_leng[:50000], y_base[:50000], color="black", label="resp base", linewidth=1)
# plt.plot(x_leng[:50000], y_stress[:50000], color="red", label="resp stress", linewidth=1)
# # plt.plot(x_leng, y_amus, color="blue", label="resp amusement", linewidth=1)
# # plt.plot(x_leng, y_medi, color="green", label="resp meditation", linewidth=1)
# plt.xlabel("Number of samples", fontsize="xx-large")
# plt.ylabel("Sensor amplitude", fontsize="xx-large")
# plt.title("RESP base vs stress", fontsize="xx-large")
# plt.grid(b=True)
# plt.legend(loc="upper left")

# fig8.savefig("../output/clean_data_visualization/resp_class_distribution.png")
# plt.show()

# ########## Cleaning the data - removing outliers ##########
#     df_baseline = pd.DataFrame(data=baseline_data,
#                                columns=["c_ax", "c_ay", "c_az", "c_eda", "c_temp"])
#     print("Baseline train df shape = {}".format(df_baseline.shape))

#     df_stress = pd.DataFrame(data=stress_data,
#                                      columns=["c_ax", "c_ay", "c_az", "c_eda", "c_temp"])
#     print("Stress train df shape = {}".format(df_stress.shape))

#     # Calculate the baseline IQR
#     baseline_q1 = df_baseline.quantile(0.25)
#     baseline_q3 = df_baseline.quantile(0.75)
#     baseline_iqr = baseline_q3 - baseline_q1
#     print("Baseline train IQR is {}".format(baseline_iqr))

#     stress_q1 = df_stress.quantile(0.25)
#     stress_q3 = df_stress.quantile(0.75)
#     stress_iqr = stress_q3 - stress_q1
#     print("Stress train IQR is {}".format(stress_iqr))

#     # Removing baseline outliers
#     df_baseline_clean = df_baseline[
#         ~((df_baseline < (baseline_q1 - 1.5 * baseline_iqr)) | (df_baseline > (baseline_q3 + 1.5 * baseline_iqr))).any(
#             axis=1)]
#     print("Baseline df train cleaned shape = {}".format(df_baseline_clean.shape))

#     df_stress_clean = df_stress[
#         ~((df_stress < (stress_q1 - 1.5 * stress_iqr)) | (
#                     df_stress > (stress_q3 + 1.5 * stress_iqr))).any(
#             axis=1)]
#     print("Stress df train cleaned shape = {}".format(df_stress_clean.shape))

#     # Convert the dataframe to list
#     print("Converting df_baseline_train_clean to list")
#     baseline_data_train_clean = utils.df_to_list_removed_features(df_baseline_clean)
#     print("Converting df_stress_train_clean to list")
#     stress_data_train_clean = utils.df_to_list_removed_features(df_stress_clean)



"""
########## Performing PCA ##########
"""
# windows_b for baseline data train
windows_b = utils.extract_windows(baseline_data_train_clean, 0.25)
print("baseline data windows : {}".format(len(windows_b)))

# windows_s for stress data train
windows_s = utils.extract_windows(stress_data_train_clean, 0.25)
print("stress data windows : {}".format(len(windows_s)))

temp_train_all = []
for each in windows_b:
    temp_train_all.append(each)
print("baseline train length : {}".format(len(temp_train_all)))

for each in windows_s:
    temp_train_all.append(each)
print("stress + baseline train length : {}".format(len(temp_train_all)))

# Extracting features for training data
all_features_train = []
for each in temp_train_all:
    features = utils.feature_extraction(each)
    all_features_train.append(features)
print("temp_train length : {} -- with shape {}".format(len(all_features_train), np.array(all_features_train).shape))

X = StandardScaler().fit_transform(all_features_train)
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(X)

Y = [0] * len(windows_b)
Y.extend([1] * len(windows_s))
Y = np.array(Y).T
Y = pd.DataFrame(data=Y, columns=['target'])

# pDF = pd.DataFrame(data=pca_fit, columns=['pc1', 'pc2', 'pc3'])
# finalDF = pd.concat([pDF, Y], axis=1)
# pc1 = pca_fit[:, 0]
# pc2 = pca_fit[:, 1]
# pc3 = pca_fit[:, 2]

# colors = ['r', 'g']
# targets = [0, 1]

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_zlabel('Principal Component 3', fontsize=15)
# ax.set_title('3 component PCA', fontsize=20)

# # for target, color in zip(targets, colors):
# #     indicesToKeep = finalDF['target'] == target
# #     print("indicesToKeep : {}".format(indicesToKeep))
# #     # pc1df = finalDF[finalDF['pc1'] < 25]
# #     # pc2df = finalDF[finalDF['pc2'] < 100]
# #     # pc3df = finalDF[finalDF['pc3'] < 100]
# #     pc1df = finalDF['pc1']
# #     pc2df = finalDF['pc2']
# #     pc3df = finalDF['pc3']
# #     ax.scatter(pc1df.loc[indicesToKeep, True],
# #                pc2df.loc[indicesToKeep, True],
# #                pc3df.loc[indicesToKeep, True],
# #                edgecolors=color,
# #                facecolors='none',
# #                s=50)

# for i in np.unique(Y):
#     x = np.where(Y==i)
#     ax.scatter(pc1[x], pc2[x], pc3[x],
#                edgecolors=colors[i],
#                facecolors='none',
#                s=50)

# ax.legend(targets)
# ax.grid()
# plt.savefig("../images/3-component-pca.png")
# plt.show()

Xax = pca_fit[:,0]
print("this is Xax : {}".format(Xax))
Yax = pca_fit[:,1]
Zax = pca_fit[:,2]

cdict = {0:'red',1:'green'}
labl = {0:'Baseline',1:'Stress'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
for l in np.unique(Y):
    ix=np.where(Y==l)
    print("this is ix : {}".format(ix))
    a = Xax[ix]
    a_1 = Yax[ix[0]]
    a_2 = Zax[ix[0]]
    b = np.array([i for i in Xax[ix[0]] if i < 25])
    c = np.array([i for i in Yax[ix[0]] if i < 200])
    d = np.array([i for i in Zax[ix[0]] if i < 10])
    ax.scatter(b, c, d,
            c=cdict[l], s=40,
           label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
ax.set_xlabel("Principal Component 1", fontsize=14)
ax.set_ylabel("Principal Component 2", fontsize=14)
ax.set_zlabel("Principal Component 3", fontsize=14)

ax.legend()
plt.show()


"""
########## Boxplot analysis ##########
"""
# df_baseline = pd.DataFrame(data=baseline_data, columns=["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"])
# print("Baseline df shape = {}".format(df_baseline.shape))
#
# # baseline boxplot for c_ax, c_ay, c_az
# print("Plotting for c_ax, c_ay, c_az")
# df_baseline.boxplot(column=["c_ax", "c_ay", "c_az"])
# plt.savefig("../images/c_axyz_baseline_boxplot.png")
# plt.show()
#
# boxplot_list = ["c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# for each in boxplot_list:
#     print("Plotting for {}".format(each))
#     df_baseline.boxplot(column=[each])
#     plt.savefig("../images/{}_baseline_boxplot.png".format(each))
#     plt.show()
#
#
# hist_list = ["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# for each in hist_list:
#     print("Plotting {} histogram".format(each))
#     df_baseline[each].hist()
#     plt.savefig("../images/{}_baseline_hist.png".format(each))
#     plt.show()
#
#
# skewness_file = "../images/all_skewness_baseline.txt"
# values = []
# for each in hist_list:
#     skewness = df_baseline[each].skew()
#     values.append([each, skewness])
#     print("Baseline {} skewness = {}".format(each, skewness))
#
# with open(skewness_file, 'w') as f:
#     for item in values:
#         f.write("{}".format(item))
# print("Skewness values saved to file")
#
# # Calculate the baseline IQR
# baseline_q1 = df_baseline.quantile(0.25)
# baseline_q3 = df_baseline.quantile(0.75)
# baseline_iqr = baseline_q3 - baseline_q1
# print("Baseline IQR is {}".format(baseline_iqr))
# baseline_iqr_file = "../images/baseline_iqr.txt"
# with open(baseline_iqr_file, 'w') as f:
#     f.write("{}".format(baseline_iqr))
# print("Baseline IQR saved to file")
#
# # Removing baseline outliers
# df_baseline_clean = df_baseline[~((df_baseline < (baseline_q1 - 1.5 * baseline_iqr)) | (df_baseline > (baseline_q3 + 1.5 * baseline_iqr))).any(axis=1)]
# print("Baseline df cleaned shape = {}".format(df_baseline_clean.shape))
#
# # baseline boxplot for c_ax, c_ay, c_az for cleaned data
# print("Plotting cleaned data for c_ax, c_ay, c_az")
# df_baseline_clean.boxplot(column=["c_ax", "c_ay", "c_az"])
# plt.savefig("../images/c_axyz_baseline_cleaned_boxplot.png")
# plt.show()
#
# boxplot_list = ["c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# for each in boxplot_list:
#     print("Plotting cleaned data for {}".format(each))
#     df_baseline_clean.boxplot(column=[each])
#     plt.savefig("../images/{}_baseline_cleaned_boxplot.png".format(each))
#     plt.show()
#
# hist_list = ["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# for each in hist_list:
#     print("Plotting cleaned data {} histogram".format(each))
#     df_baseline_clean[each].hist()
#     plt.savefig("../images/{}_baseline_cleaned_hist.png".format(each))
#     plt.show()
#
#
# # Repeat for stress
#
# df_stress = pd.DataFrame(data=stress_data, columns=["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"])
# print("Stress df shape = {}".format(df_stress.shape))
#
# # Stress boxplot for c_ax, c_ay, c_az
# print("Plotting for c_ax, c_ay, c_az")
# df_stress.boxplot(column=["c_ax", "c_ay", "c_az"])
# plt.savefig("../images/c_axyz_stress_boxplot.png")
# plt.show()
#
# boxplot_list = ["c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# for each in boxplot_list:
#     print("Plotting for {}".format(each))
#     df_stress.boxplot(column=[each])
#     plt.savefig("../images/{}_stress_boxplot.png".format(each))
#     plt.show()
#
#
# hist_list = ["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# for each in hist_list:
#     print("Plotting {} histogram".format(each))
#     df_stress[each].hist()
#     plt.savefig("../images/{}_stress_hist.png".format(each))
#     plt.show()
#
#
# skewness_file = "../images/all_skewness_stress.txt"
# values = []
# for each in hist_list:
#     skewness = df_stress[each].skew()
#     values.append([each, skewness])
#     print("Stress {} skewness = {}".format(each, skewness))
#
# with open(skewness_file, 'w') as f:
#     for item in values:
#         f.write("{}".format(item))
# print("Skewness values saved to file")
#
# # Calculate the stress IQR
# stress_q1 = df_stress.quantile(0.25)
# stress_q3 = df_stress.quantile(0.75)
# stress_iqr = stress_q3 - stress_q1
# print("Stress IQR is {}".format(stress_iqr))
# stress_iqr_file = "../images/stress_iqr.txt"
# with open(stress_iqr_file, 'w') as f:
#     f.write("{}".format(stress_iqr))
# print("Stress IQR saved to file")
#
# # Removing stress outliers
# df_stress_clean = df_stress[~((df_stress < (stress_q1 - 1.5 * stress_iqr)) | (df_stress > (stress_q3 + 1.5 * stress_iqr))).any(axis=1)]
# print("Stress df cleaned shape = {}".format(df_stress_clean.shape))
#
# # Stress boxplot for c_ax, c_ay, c_az for cleaned data
# print("Plotting cleaned data for c_ax, c_ay, c_az")
# df_stress_clean.boxplot(column=["c_ax", "c_ay", "c_az"])
# plt.savefig("../images/c_axyz_stress_cleaned_boxplot.png")
# plt.show()
#
# boxplot_list = ["c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# for each in boxplot_list:
#     print("Plotting cleaned data for {}".format(each))
#     df_stress_clean.boxplot(column=[each])
#     plt.savefig("../images/{}_stress_cleaned_boxplot.png".format(each))
#     plt.show()
#
# hist_list = ["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"]
# for each in hist_list:
#     print("Plotting cleaned data {} histogram".format(each))
#     df_stress_clean[each].hist()
#     plt.savefig("../images/{}_stress_cleaned_hist.png".format(each))
#     plt.show()

a=2