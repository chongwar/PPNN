import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

subject = 's03'


class LayerActivations:
    features = None

    def __init__(self, model, lay_num):
        self.hook = model[lay_num].register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features = output.cpu().detach()
        
    def remove(self):
        self.hook.remove()


def train(model, criterion, optimizer, data_loader, device, train_num, epochs, logged=False):
    loss_all = []
    acc_all = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_num = 0
        batch_size = None
        for index, data in enumerate(data_loader):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            _, pred = torch.max(y_pred, 1)
            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += float(loss.item())
            
        batch_num = train_num // batch_size
        _loss = running_loss / (batch_num + 1)
        acc = correct_num / train_num * 100
        loss_all.append(_loss)
        acc_all.append(acc)
        if logged:
            print(f'Epoch {epoch+1}/{epochs}\tTrain loss: {_loss:.4f}\tTrain acc: {acc:.2f}%')

    # path = f'checkpoint/{model.__class__.__name__}_{epochs}_{subject}.pth'
    # torch.save(model.state_dict(), path)
    print('Finish Training!')
         
         
def test(model, criterion, data_loader, device, test_num, log, logged=False):
    # model.load_state_dict(torch.load(f'checkpoint/{model.__class__.__name__}_30_{subject}.pth'))
    # conv_out = LayerActivations(model.block_2, 1)
    
    running_loss = 0.0
    correct_num = 0
    model.eval()
    batch_size = None
    start = time.perf_counter()
    for index, data in enumerate(data_loader):
        x, y = data
        batch_size = x.shape[0] if index == 0 else batch_size
        x, y = x.to(device), y.to(device)
        
        y_pred = model(x)
        _, pred = torch.max(y_pred, 1)
        correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        
        # get feature maps
        # if index == 0:
        #     conv_out.remove()
        #     ori_x = x.cpu().numpy()
        #     act_label = y.cpu().numpy()
        #     act_pred = pred.cpu().numpy()
        #     act = conv_out.features.numpy()
            
        loss = criterion(y_pred, y)
        running_loss += float(loss.item())
    
    end = time.perf_counter()
    print(f'Time cost: {end-start:.2f}s')
    
    batch_num = test_num // batch_size
    _loss = running_loss / (batch_num + 1)
    acc = correct_num / test_num * 100
    print(f'Test loss: {_loss:.4f}\tTest acc: {acc:.2f}%')   
    
    if logged:
        log.append(f'{acc:.2f}\t\n')
        with open('result/202209.txt', 'a') as f:
            f.writelines(log)
    
    # print(act.shape, act_label, act_pred, sep='\n')
    # np.save(f'features/s-t/{subject}/ori_x', ori_x)
    # np.save(f'features/s-t/{subject}/feature', act)
    # np.save(f'features/s-t/{subject}/label', act_label)
    # np.save(f'features/s-t/{subject}/pred', act_pred)
    
            