import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from scipy import signal
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

warnings.filterwarnings("ignore", message="Numerical issues were encountered ")


def sos_filter(data, f_1=1, f_2=45, fs=1024):
    # data: (N, T, C)
    # wn = [f_1 * 2 / fs, f_2 * 2 / fs]
    sos = signal.butter(3, Wn=[f_1, f_2], btype='bandpass', output='sos', fs=fs) 
    for trial in range(data.shape[0]):
        data[trial, ...] = signal.sosfilt(sos, data[trial, ...], axis=0)
    return data


def generate_posi(posi):
    _posi = list(posi)
    _posi.extend([5, 45])
    return _posi


def _load_data(data_dir, save_dir, block_num, trial_num=5):
    data = []
    label = []
    for block in range(1, block_num + 1):
        data_path = os.path.join(data_dir, f'data_{block}.npy')
        label_path = os.path.join(data_dir, f'label_{block}.npy')
        data_tmp = np.load(data_path)
        # data_tmp = filt(data_tmp)
        # data_tmp = filt_data(data_tmp)
        data_tmp = sos_filter(data_tmp)
        label_tmp = np.load(label_path).reshape(trial_num, -1)
        for trial in range(trial_num):
            
            posi = generate_posi(label_tmp[trial, :])
            # posi = label_tmp[trial, :]
    
            for idx, i in enumerate(posi):
                # posi_idx = (i + 1) * 200 + 200
                posi_idx = (i + 1) * 200
                
                # remove baseline
                _data_tmp = data_tmp[trial, posi_idx:posi_idx+1024, :64]
                baseline = np.mean(data_tmp[trial, posi_idx-400:posi_idx, :64])
                # _data_tmp -= baseline
                data.append(_data_tmp)
                
                # # two-class (0: person, 1: desk)
                # if idx < 2:
                #     label.append(0)
                # elif idx < 4:
                #     label.append(1)
                
                # three-class (0: bg, 1: person 2: desk)
                if idx < 2:
                    label.append(1)
                elif idx < 4:
                    label.append(2)
                else:
                    label.append(0)

                # # two-class (0: bg, 1: person & desk)
                # if idx < 4:
                #     label.append(1)
                # else:
                #     label.append(0)
                    
    np.save(f'{save_dir}/x', np.array(data))
    np.save(f'{save_dir}/y', np.array(label))          


def filt_data(data, f_1=1, f_2=40, fs=1024): 
    # data: (N, T, C)
    wn = [f_1 * 2 / fs, f_2 * 2 / fs]
    b, a = signal.butter(3, wn, 'bandpass') 
    for trial in range(data.shape[0]):
        data[trial, ...] = signal.filtfilt(b, a, data[trial, ...], axis=0)
        

def filt(data, fs=1024):
    fs = fs / 2
    wp, ws = [1/fs, 40/fs], [0.1/fs, 48/fs]
    rp, rs = 3, 40
    n, wn = signal.cheb1ord(wp, ws, rp, rs)
    # b, a = signal.cheby1(round(n/6), rp, wn, btype='bandpass')
    b, a = signal.cheby1(n//3, rp, wn, btype='bandpass')
    for trial in range(data.shape[0]):
        data[trial, ...] = signal.filtfilt(b, a, data[trial, ...], axis=0)
    

def scale_data(data):
    for i in range(data.shape[0]):
        data[i, ...] = preprocessing.scale(data[i, ...])
        # data[i, ...] = preprocessing.minmax_scale(data[i, ...])
        # data[i, ...] = preprocessing.maxabs_scale(data[i, ...])


def load_data(subject_id=1):
    data_dir = f'data/s{subject_id:>02d}'
    # shape of x:(N, T, C)
    # x = np.load(f'data/s{subject_id:>02d}/x.npy').astype(np.float32)
    # y = np.load(f'data/s{subject_id:>02d}/y.npy').astype(np.int64)
    
    x = np.load(f'{data_dir}/x_2.npy').astype(np.float32)
    y = np.load(f'{data_dir}/y_2.npy').astype(np.int64)
    
    # x = np.load(f'data/s{subject_id:>02d}/x_desk_person.npy').astype(np.float32)
    # y = np.load(f'data/s{subject_id:>02d}/y_desk_person.npy').astype(np.int64)
    print(f'Original data shape: {x.shape}')
    
    # bandpass filt
    filt_data(x)
    # filt(x)
    
    # (N, T, C) --> (N, T/4, C)
    x = x[:, range(0, x.shape[1], 4), :]

    # train-test shuffle split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=2)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    # preprocess
    scale_data(x_train)
    scale_data(x_test)
    
    # # save to mat
    # x_train = np.transpose(x_train, (2, 1, 0))
    # x_test = np.transpose(x_test, (2, 1, 0))
    # print(x_train.shape)
    # print(x_test.shape)
    # y_train = y_train.reshape(1, -1)
    # y_test = y_test.reshape(1, -1)
    # scio.savemat(f'mat_data/s{subject_id}_x_train.mat', {'x_train':x_train})
    # scio.savemat(f'mat_data/s{subject_id}_x_test.mat', {'x_test':x_test})
    # scio.savemat(f'mat_data/s{subject_id}_y_train.mat', {'y_train':y_train})
    # scio.savemat(f'mat_data/s{subject_id}_y_test.mat', {'y_test':y_test})        
        
    # (N, T, C) --> (N, 1, C, T)
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
    
    print(f'Processed train data shape: {x_train.shape}')
    print(f'Processed test  data shape: {x_test.shape}')
    
    return (x_train, x_test, y_train, y_test)
  

if __name__ == '__main__':
    _data_dir = '/home/chongwar/code/eeg/rsvp_multiclass/data'
    _subject_id = [i for i in range(6, 14)]
    
    for subject_id in tqdm(_subject_id):
        # data_dir = os.path.join(_data_dir, f's{subject_id:>02d}', 'original')
        # block_num = len(os.listdir(data_dir)) // 2
        # save_dir = os.path.join(_data_dir, f's{subject_id:>02d}')
        # _load_data(data_dir, save_dir, block_num)
        
        load_data(subject_id)
    
    # subject_id = 6
        # x_train, x_test, y_train, y_test = load_data(subject_id)
