import os
import data_utils as utils
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
########## Splitting the dataset into train and test at 80:20 ratio ##########
"""
train, test = train_test_split(all_subj_pkl, train_size=0.8)
print("train, test : {}, {}".format(len(train), len(test)))


"""
########## Getting individual classes: baseline, stress, amusement, meditation ##########
"""
baseline_data_train = utils.get_class_from_data(train, 'baseline')
stress_data_train = utils.get_class_from_data(train, 'stress')
amusement_data_train = utils.get_class_from_data(train, 'amusement')
meditation_data_train = utils.get_class_from_data(train, 'meditation')

# used for testing
baseline_data_test = utils.get_class_from_data(test, 'baseline')
stress_data_test = utils.get_class_from_data(test, 'stress')


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

for each in windows_s:
    temp_train_all.append(each)
print("stress + baseline train length : {}".format(len(temp_train_all)))

temp_train_label = []
for i in range(len(windows_b)):
    temp_train_label.append(0)

for j in range(len(windows_s)):
    temp_train_label.append(1)

print("training label with baseline count = {}, and stress count = {}, in the format {}...{}".format(len(windows_b), len(windows_s), temp_train_label[0:10], temp_train_label[-10:]))


"""
########## Preparing testing data ##########
"""
test_range = 1000
temp_test_all = []
temp_test_label = []
for i in range(test_range):
    temp_test_all.append(windows_b_t[i])
    temp_test_label.append(0)
print("baseline test length : {}".format(len(temp_test_all)))

for j in range(test_range):
    temp_test_all.append(windows_s_t[j])
    temp_test_label.append(1)
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
b_avg_acc = 0
s_avg_acc = 0
pred_avg_score = 0
loop_num = 2

temp_temp_train = baseline_data_train[0:44000]
temp_temp_train.extend(stress_data_train[0:44000])
print("temp temp train length = {} with shape {}".format(len(temp_temp_train), np.array(temp_temp_train).shape))
temp_temp_train_label = [0] * 44000
temp_temp_train_label.extend([1] * 44000)

temp_temp_test = baseline_data_test[0:1000]
temp_temp_test.extend(stress_data_test[0:1000])
print("temp temp test length = {} with shape {}".format(len(temp_temp_test), np.array(temp_temp_test).shape))
temp_temp_test_label = [0] * 1000
temp_temp_test_label.extend([1] * 1000)


for t in range(loop_num):
    print("this is the {} iteration".format(t))
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    fit_obj = svm.fit(temp_temp_train, temp_temp_train_label) #all_features_train, temp_train_label

    print("model fitted")

    pred = svm.predict(temp_temp_test) #all_features_test
    print(pred)

    count = 0
    for i in range(test_range):
        if pred[i] == 0:
            count += 1
    b_acc = count / test_range * 100
    b_avg_acc += b_acc
    print("baseline count is {} and acc is {}%".format(count, b_acc))

    s_count = 0
    for j in range(test_range + 1, test_range * 2):
        if pred[j] == 1:
            s_count += 1
    s_acc = s_count / test_range * 100
    s_avg_acc += s_acc
    print("stress count is {} and acc is {}%".format(s_count, s_acc))

    pred_score = svm.score(temp_temp_test, temp_temp_test_label) #all_features_test, temp_test_label
    pred_avg_score += pred_score
    print("this is the prediction score : {}".format(pred_score))

b_avg = b_avg_acc / loop_num
s_avg = s_avg_acc / loop_num
pred_avg = pred_avg_score / loop_num

print("baseline average acc is {}%".format(b_avg))
print("stress average acc is {}%".format(s_avg))
print("score average acc is {}%".format(pred_avg))

