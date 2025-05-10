import os
from scipy.io import loadmat
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
import random
import matplotlib.pyplot as plt

def normalization_processing(data):
    data_mean = data.mean()
    data_var = data.var()

    data = data - data_mean
    data = data / data_var

    return data

def wgn(x, snr):

    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr

    return np.random.randn(len(x)) * np.sqrt(npower)


def add_noise(data,snr_num):

    rand_data = wgn(data, snr_num)
    data = data + rand_data

    return data

def newprepro(d_path, length=1024, number=400,strides=1024, normal=True, rate=[0.5, 0.125, 0.375],valid=False,small=0,addnoise = False,snr = 0):
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    faulttype = ['B','OR','IR']
    faultlevel = ['07','14','21']
    def capture(original_path):
        files = {}
        for i in filenames:
            
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()

            for j in faulttype:
                if j in i:
                    for k in faultlevel:
                        if k in i:
                            for key in file_keys:
                                if 'DE' in key:
                                    label = j + k
                                    if label not in files:
                                        files[label] = []
                                    if addnoise == True:
                                        files[label].extend(add_noise(file[key].ravel(),snr))
                                    else:
                                        files[label].extend(file[key].ravel())
                elif 'norm' in i:
                    for key in file_keys:
                                if 'DE' in key:
                                    if 'norm' not in files:
                                        files['norm'] = []
                                    if addnoise == True:
                                        files['norm'].extend(add_noise(file[key].ravel(),snr))
                                    else:
                                        files['norm'].extend(file[key].ravel())
        return files

    
    def random_slice_enc(data,slice_rate=rate[1] + rate[2],slicerandom=True):
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_length = len(slice_data)
            data_sample = []
            Train_Sample = []
            Test_Sample = []
            start_index = 0
            while start_index + length <= all_length:
                sample = slice_data[start_index:start_index + length]
                data_sample.append(sample)
                start_index += strides
            testnum = int(number * slice_rate)
            trainnum = int(number * (1 - slice_rate))
            if slicerandom:
                data_sample_tuples = [tuple(x) for x in data_sample]
                Train_Sample_tuples = random.sample(data_sample_tuples, trainnum)
                remaining_samples = list(set(data_sample_tuples) - set(Train_Sample_tuples))
                Test_Sample = random.sample(remaining_samples,testnum)
                Train_Sample = [list(x) for x in Train_Sample_tuples]
                Test_Sample = [list(x) for x in Test_Sample]
            else:
                Train_Sample = data_sample[:trainnum]
                Test_Sample = data_sample[trainnum+1:]
            Train_Samples[i] = Train_Sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples
    
    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        labelkeys = train_test.keys()
        for i in labelkeys:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    def add_small_sample_labels(train_test,number):
        X = []
        Y = []
        label = 0
        labelkeys = train_test.keys()
        for i in labelkeys:
            x = random.sample(train_test[i], number)
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y
    
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test


    data = capture(original_path=d_path)
    train, test = random_slice_enc(data)


    if small == 0:
        Train_X, Train_Y = add_labels(train)
    else :
        Train_X, Train_Y = add_small_sample_labels(train,small)

    Test_X, Test_Y = add_labels(test)

    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)

    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:

        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)

    if valid:
        Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
        return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    return Train_X, Train_Y, Test_X, Test_Y


def data2prepro(d_path, length=1024, number=400,strides=1024, normal=True, rate=[0.5, 0.125, 0.375],valid=False,small=0,addnoise = False,snr = 0):
    filenames = os.listdir(d_path)
    def capture(original_path):
        files = {}
        for i in filenames:
            datafilenames = d_path + i + "\\"
            label = i
            Sourcedatafilenames =  os.listdir(datafilenames)
            for j in Sourcedatafilenames:
                    file_path = os.path.join(datafilenames, j)
                    file = loadmat(file_path)
                    file_keys = list(file.keys())
                    c = file[file_keys[3]]['Y'][0][0][0][6][2][0]
                    c = c[:245760]
                    if label not in files:
                        files[label] = []
                    if addnoise == True:
                        files[label].extend(add_noise(c,snr))
                    else:
                        files[label].extend(c)
        return files

    def random_slice_enc(data,slice_rate=rate[1] + rate[2],slicerandom=True):
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_length = len(slice_data)
            data_sample = []
            Train_Sample = []
            Test_Sample = []
            start_index = 0
            while start_index + length <= all_length:
                sample = slice_data[start_index:start_index + length]
                data_sample.append(sample)
                start_index += strides
            testnum = int(number * slice_rate)
            trainnum = int(number * (1 - slice_rate))
            if slicerandom:
                data_sample_tuples = [tuple(x) for x in data_sample]
                Train_Sample_tuples = random.sample(data_sample_tuples, trainnum)
                remaining_samples = list(set(data_sample_tuples) - set(Train_Sample_tuples))
                Test_Sample = random.sample(remaining_samples,testnum)
                Train_Sample = [list(x) for x in Train_Sample_tuples]
                Test_Sample = [list(x) for x in Test_Sample]
            else:
                Train_Sample = data_sample[:trainnum]
                Test_Sample = data_sample[trainnum+1:]
            Train_Samples[i] = Train_Sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples
    
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        labelkeys = train_test.keys()
        for i in labelkeys:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    def add_small_sample_labels(train_test,number):
        X = []
        Y = []
        label = 0
        labelkeys = train_test.keys()
        for i in labelkeys:
            x = random.sample(train_test[i], number)
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1

    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test


    data = capture(original_path=d_path)

    train, test = random_slice_enc(data)

    if small == 0:
        Train_X, Train_Y = add_labels(train)
    else :
        Train_X, Train_Y = add_small_sample_labels(train,small)

    Test_X, Test_Y = add_labels(test)

    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)

    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:

        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)

    if valid:
        Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
        return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    return Train_X, Train_Y, Test_X, Test_Y
