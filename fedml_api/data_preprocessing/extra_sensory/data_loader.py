DEFAULT_TRAIN_CLINETS_NUM = 60

import numpy as np
import gzip
import io
import os
import torch
import warnings

def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index(str.encode('\n'))]
    columns = headline.split(str.encode(','))

    # The first column should be timestamp:
    assert columns[0] == str.encode('timestamp')
    # The last column should be label_source:
    assert columns[-1] == str.encode('label_source')
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith(str.encode('label:')):
            first_label_ind = ci
            break
        pass

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith(str.encode('label:'))
        label_names[li] = label.replace(str.encode('label:'),str.encode(''))
        pass
    
    return (feature_names,label_names)

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(io.BytesIO(csv_str),delimiter=str.encode(','),skiprows=1)
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int)
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)]
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]; # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat); # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix
    
    return (X,Y,M,timestamps)

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
def read_user_data(uuid, data_path='../../../data/extra_sensory/ExtraSensory.per_uuid_features_labels/'):
    user_data_file = '%s' % uuid
    # Read the entire csv file of the user:
    with gzip.open(data_path + user_data_file,'rb') as fid:
        csv_str = fid.read()
        pass

    (feature_names,label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features)

    return (X,Y,M,timestamps,feature_names,label_names)

def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names])
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith(b'raw_acc'):
            feat_sensor_names[fi] = 'Acc'
            pass
        elif feat.startswith(b'proc_gyro'):
            feat_sensor_names[fi] = 'Gyro'
            pass
        elif feat.startswith(b'raw_magnet'):
            feat_sensor_names[fi] = 'Magnet'
            pass
        elif feat.startswith(b'watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc'
            pass
        elif feat.startswith(b'watch_heading'):
            feat_sensor_names[fi] = 'Compass'
            pass
        elif feat.startswith(b'location'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith(b'location_quick_features'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith(b'audio_naive'):
            feat_sensor_names[fi] = 'Aud'
            pass
        elif feat.startswith(b'audio_properties'):
            feat_sensor_names[fi] = 'AP'
            pass
        elif feat.startswith(b'discrete'):
            feat_sensor_names[fi] = 'PS'
            pass
        elif feat.startswith(b'lf_measurements'):
            feat_sensor_names[fi] = 'LF'
            pass
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat)
        pass
    return feat_sensor_names;    

def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names),dtype=bool)
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor)
        use_feature = np.logical_or(use_feature,is_from_sensor)
        pass
    X = X[:,use_feature]
    return X

def estimate_standardization_params(X_train):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_vec = np.nanmean(X_train,axis=0)
        std_vec = np.nanstd(X_train,axis=0)
    return (mean_vec,std_vec)

def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1,-1))
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1))
    X_standard = X_centralized / normalizers
    return X_standard

def preprocess_extra_sensory(X, Y, M, timestamps, feature_names, label_names): 
    X_all = X.tolist()
    Y_all = Y.tolist()
    M_all = M.tolist()
    T_all = timestamps.tolist()
    for i in range(len(X)-1, -1, -1):
        labels_to_display = [b'LYING_DOWN',b'SITTING',b'OR_standing',b'FIX_walking',b'FIX_running']
        Y_all[i] = [Y[i][label_names.index(label)] for label in labels_to_display]
        if np.sum(Y_all[i]) != 1:
            Y_all.pop(i)
            T_all.pop(i)
            X_all.pop(i)
            M_all.pop(i)
        else:
            T_all[i] = T_all[i] - T_all[0]
            Y_all[i] = Y_all[i].index(1)
    sensors_to_use = ['Acc','WAcc']
    feat_sensor_names = get_sensor_names_from_features(feature_names)
    X_all = project_features_to_selected_sensors(np.array(X_all),feat_sensor_names,sensors_to_use)
    (mean_vec,std_vec) = estimate_standardization_params(X_all)
    X_all = standardize_features(X_all, mean_vec, std_vec)
    X_all[np.isnan(X_all)] = 0.

    return X_all, Y_all, M_all, T_all

def batch_data(data_x, data_y, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data

def load_partition_data_extra_sensory(data_path='../../../data/extra_sensory/ExtraSensory.per_uuid_features_labels/'):
    files = os.listdir(data_path)
    client_idx = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    batch_size_dict = dict()
    test_data_global = list()
    for user_data_file in files:
        if user_data_file.endswith('.csv.gz'):
            (X,Y,M,timestamps,feature_names,label_names) = read_user_data(user_data_file)
            X_all, Y_all, M_all, T_all = preprocess_extra_sensory(X, Y, M, timestamps, feature_names, label_names)
            #train_data_local_num_dict[client_idx] = user_train_data_num
            #train_data_local_num_dict[client_idx] = user_train_data_num
            #train_data_num += len(X)
            #test_data_num = train_data_num
            # TODO: train_data_num and test_data_num are for what?
            time_horizon = int(len(X_all)*0.7)
            train_X = X_all[:time_horizon]
            test_X = X_all[time_horizon+1:]
            train_Y = Y_all[:time_horizon]
            test_Y = Y_all[time_horizon+1:]
            #batch_size_dict[client_idx] = get_batch_size(client_idx)
            batch_size_dict[client_idx] = 4
            train_batch = batch_data(train_X, train_Y, batch_size_dict[client_idx])
            test_batch = batch_data(test_X, test_Y, batch_size_dict[client_idx])
            train_data_local_dict[client_idx] = train_batch
            test_data_local_dict[client_idx] = test_batch
            test_data_global += test_batch
            client_idx += 1
        else:
            continue
    client_num = client_idx
    class_num = 5
    return client_num, test_data_global, train_data_local_dict, test_data_local_dict, class_num
