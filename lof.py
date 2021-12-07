import os
import numpy as np
import data_utils as utils

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score


""" 
########## Setting Variable Paths ########## 
"""
pkl_base = "/home/li.zhiyuan/Desktop/Probabilistic Inference/Project/dataset/wesad pkl/"

pkl_baseline_train = os.path.join(pkl_base, "baseline_train.pkl")
pkl_stress_train = os.path.join(pkl_base, "stress_train.pkl")
pkl_baseline_test = os.path.join(pkl_base, "baseline_test.pkl")
pkl_stress_test = os.path.join(pkl_base, "stress_test.pkl")


"""
########## Read csv files for training and testing ##########
"""
print("reading baseline_data_train pickle file")
baseline_data_train = utils.read_pkl(pkl_baseline_train)
print("reading stress_data_train pickle file")
stress_data_train = utils.read_pkl(pkl_stress_train)
print("reading baseline_data_test pickle file")
baseline_data_test = utils.read_pkl(pkl_baseline_test)
print("reading stress_data_test pickle file")
stress_data_test = utils.read_pkl(pkl_stress_test)


"""
########## Sliding window of 0.25 seconds ##########
"""
# Readings taken at 700Hz --> 175 samples (at 0.25s window)

# windows_b for baseline data train
windows_b = utils.extract_windows(baseline_data_train, 0.25)
print("baseline data windows : {}".format(len(windows_b)))

# windows_s for stress data train
windows_s = utils.extract_windows(stress_data_train, 0.25)
print("stress data windows : {}".format(len(windows_s)))

# windows_b_t for baseline data test
windows_b_t = utils.extract_windows(baseline_data_test, 0.25)
print("baseline test data windows : {}".format(len(windows_b_t)))

# windows_s_t for stress data test
windows_s_t = utils.extract_windows(stress_data_test, 0.25)
print("stress test data windows : {}".format(len(windows_s_t)))


"""
########## Preparing training data ##########
"""
temp_train_all = []
for each in windows_b:
    temp_train_all.append(each)
print("baseline train length : {}".format(len(temp_train_all)))

for s in range(5600): # 11000, 8400, 5600, 2800, 1400
    temp_train_all.append(windows_s[s])
print("stress + baseline train length : {}".format(len(temp_train_all)))


"""
########## Preparing testing data ##########
"""
test_range = 1000
temp_test_all = []
for i in range(test_range):
    temp_test_all.append(windows_b_t[i])
print("baseline test length : {}".format(len(temp_test_all)))

for j in range(test_range):
    temp_test_all.append(windows_s_t[j])
print("stress + baseline test length : {}".format(len(temp_test_all)))


"""
########## Perform feature engineering on each of the windows ##########
"""
# Extracting features for training data
all_features_train = []
for each in temp_train_all:
    features = utils.feature_extraction(each)
    all_features_train.append(features)
print("temp_train length : {} -- with shape {}".format(len(all_features_train), np.array(all_features_train).shape))

# Extracting features for testing data
all_features_test = []
for each in temp_test_all:
    features = utils.feature_extraction(each)
    all_features_test.append(features)
print("temp_test length : {} -- with shape {}".format(len(all_features_test), np.array(all_features_test).shape))


"""
########## Fitting the model ##########
"""
print("going to fit model")

b_avg_acc = 0
s_avg_acc = 0
loops = 1
for k in range(loops):
    print("iteration number {}".format(k))

    lof = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='mahalanobis', p=2, metric_params=None, contamination=0.1)
    y_pred = lof.fit_predict(all_features_train)
    print("model fitted")

    pred = lof.negative_outlier_factor_
    print(pred)

    Y = [1]*len(windows_b)
    Y.extend([-1]*5600)
    print("Accuracy Score :")
    print(accuracy_score(Y, y_pred))
    print("Classification Report :")
    print(classification_report(Y, y_pred))

#     count = 0
#     for i in range(test_range):
#         if pred[i] == 1:
#             count += 1
#     b_acc = count / test_range * 100
#     b_avg_acc += b_acc
#     print("baseline count is {} and acc is {}%".format(count, b_acc))
#
#     s_count = 0
#     for j in range(test_range + 1, test_range * 2):
#         if pred[j] == -1:
#             s_count += 1
#     s_acc = s_count / test_range * 100
#     s_avg_acc += s_acc
#     print("stress count is {} and acc is {}%".format(s_count, s_acc))
#
# print("average baseline acc is {}%".format(b_avg_acc / loops))
# print("average stress acc is {}%".format(s_avg_acc / loops))


"""
########## Plotting ##########
"""