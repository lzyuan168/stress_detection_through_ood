import os
import data_utils as utils

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
#print("all subjects pkl readings : ", all_subj_pkl)


"""
########## Splitting the dataset into train and test at 80:20 ratio ##########
"""
train, test = train_test_split(all_subj_pkl, train_size=0.8)
print("train, test : {}, {}".format(len(train), len(test)))

# train, val = train_test_split(train_val, train_size=0.8)
# print("train, val : {}, {}".format(len(train), len(val)))


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
########## Saving train, test to csv files ##########
"""
savedir_base = "/home/li.zhiyuan/Desktop/Probabilistic Inference/Project/dataset/wesad pkl/"
savedir_train = os.path.join(savedir_base, "train.pkl")
savedir_test = os.path.join(savedir_base, "test.pkl")
savedir_baseline_train = os.path.join(savedir_base, "baseline_train.pkl")
savedir_stress_train = os.path.join(savedir_base, "stress_train.pkl")
savedir_baseline_test = os.path.join(savedir_base, "baseline_test.pkl")
savedir_stress_test = os.path.join(savedir_base, "stress_test.pkl")

utils.save_to_pickle(train, 'train', savedir_train)
utils.save_to_pickle(test, 'test', savedir_test)

print("writing baseline_data_train to file")
utils.save_to_pickle(baseline_data_train, 'baseline_data_train', savedir_baseline_train)
print("writing stress_data_train to file")
utils.save_to_pickle(stress_data_train, 'stress_data_train', savedir_stress_train)
print("writing baseline_data_test to file")
utils.save_to_pickle(baseline_data_test, 'baseline_data_test', savedir_baseline_test)
print("writing stress_data_test to file")
utils.save_to_pickle(stress_data_test, 'stress_data_test', savedir_stress_test)
