import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from tqdm import trange
from load_data import load_data
from make_dataset import MyDataset
from model import PPNN
from torch.utils.data import DataLoader
from train_test import train, test

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ', DEVICE)


def generate_data_info(subject_id, batch_size):
    print(f'Current directory: s{subject_id}')
    
    # load data from '*.npy' file
    x_train, x_test, y_train, y_test = load_data(subject_id)
    # num_classes = 3
    num_classes = 2
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    
    # make dataset
    train_data = MyDataset(x_train, x_test, y_train, y_test)
    test_data = MyDataset(x_train, x_test, y_train, y_test, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    data_info = defaultdict()
    data_info['subject_id'] = subject_id
    data_info['batch_size'] = batch_size
    data_info['num_classes'] = num_classes
    data_info['times'] = x_train.shape[-1]
    data_info['channels'] = x_train.shape[-2]
    data_info['train_num'] = train_num
    data_info['test_num'] = test_num
    data_info['train_loader'] = train_loader
    data_info['test_loader'] = test_loader

    return data_info


def main(date_info, net, epochs, iters_num, logged=False):
    num_classes = date_info['num_classes']
    
    for i in trange(iters_num):
        # model initiation
        if net == 'PPNN':
            print('Using PPNN')
            model = PPNN(num_classes=num_classes, 
                             T=data_info['times'], C=data_info['channels'])
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        subject_id = data_info['subject_id']
        batch_size = data_info['batch_size']
        log = []
        if logged:
            log.append(f's{subject_id}\t{net:<10s}\t{batch_size:<4d}\t{epochs:<4d}\t')
        
        train(model, criterion, optimizer, data_info['train_loader'], 
              DEVICE, data_info['train_num'], epochs, logged)
        
        test(model, criterion, data_info['test_loader'], 
             DEVICE, data_info['test_num'], log, logged)


if __name__ == '__main__':
    """
    Hyperparameter search
    """
    _subject_id = [13]
    _batch_size = [16]

    _net = ['PPNN']
    _epochs = [15]
    iters_num = 2
    logged = True
    
    for subject_id in _subject_id:
        for batch_size in _batch_size:
            data_info = generate_data_info(subject_id, batch_size)
            for net in _net:
                for epochs in _epochs:
                    main(data_info, net, epochs, iters_num, logged)
