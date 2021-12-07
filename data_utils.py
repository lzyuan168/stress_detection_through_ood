import os
import pandas as pd
import numpy as np
import pickle


def read_pkl(filepath):
    """
    SX.pkl file is the manual synchronisation of the two devices’ raw data.
    The result is provided in the files SX.pkl, one file per subject. This file is a dictionary, with the following
    keys:

    ‘subject’: SX, the subject ID
    ‘signal’: includes all the raw data, in two fields:
        ‘chest’: RespiBAN data (all the modalities: ACC, ECG, EDA, EMG, RESP, TEMP)
        ‘wrist’: Empatica E4 data (all the modalities: ACC, BVP, EDA, TEMP)
    ‘label’: ID of the respective study protocol condition, sampled at 700 Hz. The following IDs
    are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
    4 = meditation, 5/6/7 = should be ignored in this dataset
    """
    data = pd.read_pickle(filepath)
    return data


def read_all_subj_pkl(dirpath):
    """
    Parameters
    ----------
    dirpath: path to the base dataset folder

    Returns:
    -------
    all_subj_pkl (a list of all the subjects pkl readings)
    """
    count = 1
    all_subj_pkl = []
    dir_list = os.listdir(dirpath)
    for each in dir_list:
        if os.path.isdir(os.path.join(dirpath, each)):
            print("Folder {} name is : {}".format(count, each))
            count += 1
            subj_path = os.path.join(dirpath, each)
            file_list = os.listdir(subj_path)
            for file in file_list:
                if file == "{}.pkl".format(each):
                    filepath = os.path.join(subj_path, file)
                    subj_pkl = read_pkl(filepath)
                    all_subj_pkl.append(subj_pkl)
    return all_subj_pkl


def get_class_from_data(all_subj_list, class_name):
    """
    Parameters
    ----------
    all_subj_list : pkl reading from all 15 subjects
    class_name : the class which should be extracted from the subj_list

    Returns
    class_list : a list of the intended class
    -------

    """
    class_list = []

    for each in all_subj_list:
        c_ax = each['signal']['chest']['ACC'][0:, 0]
        c_ay = each['signal']['chest']['ACC'][0:, 1]
        c_az = each['signal']['chest']['ACC'][0:, 2]
        c_ecg = each['signal']['chest']['ECG'][:, 0]
        c_emg = each['signal']['chest']['EMG'][:, 0]
        c_eda = each['signal']['chest']['EDA'][:, 0]
        c_temp = each['signal']['chest']['Temp'][:, 0]
        c_resp = each['signal']['chest']['Resp'][:, 0]
        label = each['label']

        numpy_data = np.array([c_ax, c_ay, c_az, c_ecg, c_emg, c_eda, c_temp, c_resp, label]) #c_ecg, c_emg, c_resp,
        numpy_data = numpy_data.T
        df = pd.DataFrame(data=numpy_data,
                          columns=["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp", "label"]) #"c_ecg", "c_emg", "c_resp",
        if class_name == 'baseline':
            class_df = df.loc[(df['label'] == 1)]
            class_list.extend(np.array(class_df.values[:, :8]))
        elif class_name == 'stress':
            class_df = df.loc[(df['label'] == 2)]
            class_list.extend(np.array(class_df.values[:, :8]))
        elif class_name == 'amusement':
            class_df = df.loc[(df['label'] == 3)]
            class_list.extend(np.array(class_df.values[:, :8]))
        elif class_name == 'meditation':
            class_df = df.loc[(df['label'] == 4)]
            class_list.extend(np.array(class_df.values[:, :8]))

    return class_list


def get_class_from_data_removed_features(all_subj_list, class_name):
    """
    Parameters
    ----------
    all_subj_list : pkl reading from all 15 subjects
    class_name : the class which should be extracted from the subj_list

    Returns
    class_list : a list of the intended class
    -------

    """
    class_list = []

    for each in all_subj_list:
        c_ax = each['signal']['chest']['ACC'][0:, 0]
        c_ay = each['signal']['chest']['ACC'][0:, 1]
        c_az = each['signal']['chest']['ACC'][0:, 2]
        # c_ecg = each['signal']['chest']['ECG'][:, 0]
        # c_emg = each['signal']['chest']['EMG'][:, 0]
        c_eda = each['signal']['chest']['EDA'][:, 0]
        c_temp = each['signal']['chest']['Temp'][:, 0]
        # c_resp = each['signal']['chest']['Resp'][:, 0]
        label = each['label']

        numpy_data = np.array([c_ax, c_ay, c_az, c_eda, c_temp, label]) #c_ecg, c_emg, c_resp,
        numpy_data = numpy_data.T
        df = pd.DataFrame(data=numpy_data,
                          columns=["c_ax", "c_ay", "c_az", "c_eda", "c_temp", "label"]) #"c_ecg", "c_emg", "c_resp",
        if class_name == 'baseline':
            class_df = df.loc[(df['label'] == 1)]
            class_list.extend(np.array(class_df.values[:, :5]))
        elif class_name == 'stress':
            class_df = df.loc[(df['label'] == 2)]
            class_list.extend(np.array(class_df.values[:, :5]))
        elif class_name == 'amusement':
            class_df = df.loc[(df['label'] == 3)]
            class_list.extend(np.array(class_df.values[:, :5]))
        elif class_name == 'meditation':
            class_df = df.loc[(df['label'] == 4)]
            class_list.extend(np.array(class_df.values[:, :5]))

    return class_list


# def truncate_data(data_list):
#
#     smallest = len(data_list[0])
#     new_data_list = []
#
#     for each in data_list:
#         leng = len(each)
#         if leng < smallest:
#             smallest = leng
#
#     for each in data_list:
#         new_data_list.append(each[:smallest, :])
#
#     return new_data_list


def extract_windows(data, window_size):
    """
    Parameters
    ----------
    data : list of lists
    window_size : in terms of seconds

    Returns : list of windows determined by the window size
    -------

    """
    windows = []
    sub_window_size = int(700 * window_size)
    print("sub_window_size : {}".format(sub_window_size))
    max_time = int(len(data) // sub_window_size)
    print("max_time: {}".format(max_time))

    start = 0
    for i in range(max_time):
        example = data[start:start+sub_window_size]
        windows.extend([example])
        start += sub_window_size

    return windows


def feature_extraction(window):
    """
    Parameters
    ----------
    window : a list of data in the appropriate window size

    Returns : a list of features for the window
    -------

    """
    features = []
    mean = np.mean(window, axis=0)
    features.extend(mean)
    std = np.std(window, axis=0)
    features.extend(std)
    max_lst = np.amax(window, axis=0)
    features.extend(max_lst)
    min_lst = np.amin(window, axis=0)
    features.extend(min_lst)
    return features


def save_to_pickle(mylist, lst_name, savedir):
    with open(savedir, 'wb') as myfile:
        pickle.dump(mylist, myfile)
        print("{} is saved successfully to {}".format(lst_name, savedir))


def get_chest_and_wrist(all_sub_list, leng=True):
    """
    Parameters
    ----------
    all_sub_list : pkl read from all 15 subjects
    leng : bool parameters to indicate if length should be returned

    Returns
    if leng==True : len of all chest and wrist devices
    if leng==False : return the df of chest devices
    -------

    """
    c_ax_all = []
    c_ay_all = []
    c_az_all = []
    c_ecg_all = []
    c_emg_all = []
    c_eda_all = []
    c_temp_all = []
    c_resp_all = []
    w_ax_all = []
    w_ay_all = []
    w_az_all = []
    w_bvp_all = []
    w_eda_all = []
    w_temp_all = []

    for each in all_sub_list:
        c_ax = each['signal']['chest']['ACC'][0:, 0]
        c_ay = each['signal']['chest']['ACC'][0:, 1]
        c_az = each['signal']['chest']['ACC'][0:, 2]
        c_ecg = each['signal']['chest']['ECG'][:, 0]
        c_emg = each['signal']['chest']['EMG'][:, 0]
        c_eda = each['signal']['chest']['EDA'][:, 0]
        c_temp = each['signal']['chest']['Temp'][:, 0]
        c_resp = each['signal']['chest']['Resp'][:, 0]
        w_ax = each['signal']['wrist']['ACC'][0:, 0]
        w_ay = each['signal']['wrist']['ACC'][0:, 1]
        w_az = each['signal']['wrist']['ACC'][0:, 2]
        w_bvp = each['signal']['wrist']['BVP'][:, 0]
        w_eda = each['signal']['wrist']['EDA'][:, 0]
        w_temp = each['signal']['wrist']['TEMP'][:, 0]

        c_ax_all.extend(c_ax)
        c_ay_all.extend(c_ay)
        c_az_all.extend(c_az)
        c_ecg_all.extend(c_ecg)
        c_emg_all.extend(c_emg)
        c_eda_all.extend(c_eda)
        c_temp_all.extend(c_temp)
        c_resp_all.extend(c_resp)
        w_ax_all.extend(w_ax)
        w_ay_all.extend(w_ay)
        w_az_all.extend(w_az)
        w_bvp_all.extend(w_bvp)
        w_eda_all.extend(w_eda)
        w_temp_all.extend(w_temp)

    if leng:
        return [len(c_ax_all), len(c_ay_all), len(c_az_all), len(c_ecg_all), len(c_emg_all), len(c_eda_all),
                len(c_temp_all), len(c_resp_all),
                len(w_ax_all), len(w_ay_all), len(w_az_all), len(w_bvp_all), len(w_eda_all), len(w_temp_all)]



    else:
        numpy_data = np.array([c_ax_all, c_ay_all, c_az_all, c_ecg_all, c_emg_all, c_eda_all, c_temp_all, c_resp_all])
        numpy_data = numpy_data.T
        df = pd.DataFrame(data=numpy_data,
                          columns=["c_ax", "c_ay", "c_az", "c_ecg", "c_emg", "c_eda", "c_temp", "c_resp"])

        return df


def get_classes_per_device(baseline_df, stress_df, amusement_df, meditation_df, dev_name):
    """
    Parameters
    ----------
    baseline_df : all 15 subj baseline_data list
    stress_df : all 15 subj baseline_data list
    amusement_df : all 15 subj baseline_data list
    meditation_df : all 15 subj baseline_data list
    dev_name : name of the device [eg. "c_ax" or "c_ecg" ... ]

    Returns
    y_base : the baseline readings for the device
    y_stress : the stress readings for the device
    y_amus : the amusement readings for the device
    y-medi : the meditation readings for the device
    x_leng : the min length of readings among the classes
    -------

    """
    leng = len(amusement_df)

    if dev_name == "c_ax":
        y_base = np.array(baseline_df)[:leng, 0]
        y_stress = np.array(stress_df)[:leng, 0]
        y_amus = np.array(amusement_df)[:leng, 0]
        y_medi = np.array(meditation_df)[:leng, 0]
        x_leng = np.arange(leng)

        return y_base, y_stress, y_amus, y_medi, x_leng

    elif dev_name == "c_ay":
        y_base = np.array(baseline_df)[:leng, 1]
        y_stress = np.array(stress_df)[:leng, 1]
        y_amus = np.array(amusement_df)[:leng, 1]
        y_medi = np.array(meditation_df)[:leng, 1]
        x_leng = np.arange(leng)

        return y_base, y_stress, y_amus, y_medi, x_leng

    elif dev_name == "c_az":
        y_base = np.array(baseline_df)[:leng, 2]
        y_stress = np.array(stress_df)[:leng, 2]
        y_amus = np.array(amusement_df)[:leng, 2]
        y_medi = np.array(meditation_df)[:leng, 2]
        x_leng = np.arange(leng)

        return y_base, y_stress, y_amus, y_medi, x_leng

    elif dev_name == "c_ecg":
        y_base = np.array(baseline_df)[:leng, 3]
        y_stress = np.array(stress_df)[:leng, 3]
        y_amus = np.array(amusement_df)[:leng, 3]
        y_medi = np.array(meditation_df)[:leng, 3]
        x_leng = np.arange(leng)

        return y_base, y_stress, y_amus, y_medi, x_leng

    elif dev_name == "c_emg":
        y_base = np.array(baseline_df)[:leng, 4]
        y_stress = np.array(stress_df)[:leng, 4]
        y_amus = np.array(amusement_df)[:leng, 4]
        y_medi = np.array(meditation_df)[:leng, 4]
        x_leng = np.arange(leng)

        return y_base, y_stress, y_amus, y_medi, x_leng

    elif dev_name == "c_eda":
        y_base = np.array(baseline_df)[:leng, 5]
        y_stress = np.array(stress_df)[:leng, 5]
        y_amus = np.array(amusement_df)[:leng, 5]
        y_medi = np.array(meditation_df)[:leng, 5]
        x_leng = np.arange(leng)

        return y_base, y_stress, y_amus, y_medi, x_leng

    elif dev_name == "c_temp":
        y_base = np.array(baseline_df)[:leng, 6]
        y_stress = np.array(stress_df)[:leng, 6]
        y_amus = np.array(amusement_df)[:leng, 6]
        y_medi = np.array(meditation_df)[:leng, 6]
        x_leng = np.arange(leng)

        return y_base, y_stress, y_amus, y_medi, x_leng

    elif dev_name == "c_resp":
        y_base = np.array(baseline_df)[:leng, 7]
        y_stress = np.array(stress_df)[:leng, 7]
        y_amus = np.array(amusement_df)[:leng, 7]
        y_medi = np.array(meditation_df)[:leng, 7]
        x_leng = np.arange(leng)

        return y_base, y_stress, y_amus, y_medi, x_leng


def df_to_list(dataframe):
    """
    Parameters
    ----------
    dataframe : variable of type dataframe

    Returns
    all_list : a list of lists for the dataframe
    -------

    """
    # all_list = []
    # leng = dataframe.shape[0]
    # for i in range(leng):
    #     one_line = dataframe.iloc[i, [0, 1, 2, 3, 4, 5, 6, 7]]
    #     one_list = one_line.tolist()
    #     all_list.append(one_list)

    c_ax = dataframe["c_ax"].tolist()
    c_ay = dataframe["c_ay"].tolist()
    c_az = dataframe["c_az"].tolist()
    c_ecg = dataframe["c_ecg"].tolist()
    c_emg = dataframe["c_emg"].tolist()
    c_eda = dataframe["c_eda"].tolist()
    c_temp = dataframe["c_temp"].tolist()
    c_resp = dataframe["c_resp"].tolist()
    combined_list = list(zip(c_ax, c_ay, c_az, c_ecg, c_emg, c_eda, c_temp, c_resp))
    return combined_list


def df_to_list_removed_features(dataframe):
    """
    Parameters
    ----------
    dataframe : variable of type dataframe

    Returns
    all_list : a list of lists for the dataframe
    -------

    """
    # all_list = []
    # leng = dataframe.shape[0]
    # for i in range(leng):
    #     one_line = dataframe.iloc[i, [0, 1, 2, 3, 4, 5, 6, 7]]
    #     one_list = one_line.tolist()
    #     all_list.append(one_list)

    c_ax = dataframe["c_ax"].tolist()
    c_ay = dataframe["c_ay"].tolist()
    c_az = dataframe["c_az"].tolist()
    #c_ecg = dataframe["c_ecg"].tolist()
    #c_emg = dataframe["c_emg"].tolist()
    c_eda = dataframe["c_eda"].tolist()
    c_temp = dataframe["c_temp"].tolist()
    #c_resp = dataframe["c_resp"].tolist()
    combined_list = list(zip(c_ax, c_ay, c_az, c_eda, c_temp))
    return combined_list

