# -*- coding: utf-8 -*-

import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_provider
# 导入自己的模型
import seq2pointmodel as model

# our schme multiple states without CRF layer
params_appliance = {
    'kettle': {
        'window_len': 599,
        'uk_on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        'uk_state_num': 2,
        'uk_state': [2000, 4500],
        'uk_state_average': [1.15, 2280.79],  # 1.2230124 2796.673
        's2s_length': 128,
        'lmda': 0.6
    },
    'microwave': {
        'window_len': 599,
        'redd_on_power_threshold': 300,
        'uk_on_power_threshold': 300,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        'redd_state_num': 2,
        'redd_state': [300, 3000],
        'redd_state_average': [4.2, 1557.501],
        'uk_state_num': 2,
        'uk_state': [300, 3000],
        'uk_state_average': [1.4, 1551.3],
        's2s_length': 128,
        'lmda': 1.7,
        'lmda_redd': 0.5
    },
    'fridge': {
        'window_len': 599,
        'redd_on_power_threshold': 50,
        'uk_on_power_threshold': 20,
        'max_on_power': 3323,
        # 'mean': 200,
        # 'std': 400,

        'mean': 50,
        'std': 100,
        'redd_state_num': 3,
        'redd_state': [50, 300, 500],
        'redd_state_average': [3.2, 143.3, 397.3],
        'redd_house1_state_num': 3,
        'redd_house1_state': [50, 300, 500],
        'redd_house1_state_average': [6.49, 192.57, 443],
        'redd_house2_state_num': 3,
        'redd_house2_state': [50, 300, 500],
        'redd_house2_state_average': [6.34, 162.87, 418.36],
        'redd_house3_state_num': 3,
        'redd_house3_state': [50, 300, 500],
        'redd_house3_state_average': [0.54, 118.85, 409.75],
        # Old
        # 'uk_state_num': 3,
        # 'uk_state': [20, 200, 2500],
        'uk_state_num': 2,
        'uk_state': [20, 200],
        'uk_state_average': [0.13, 87.26, 246.5],
        's2s_length': 512,
        'lmda': 6,
        'lmda_redd': 16
    },
    'dishwasher': {
        'window_len': 599,
        'redd_on_power_threshold': 150,
        'uk_on_power_threshold': 50,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        'redd_state_num': 4,
        'redd_state': [150, 300, 1000, 3000],
        'redd_state_average': [0.57, 232.91, 733.89, 1198.31],
        'redd_house1_state_num': 4,
        'redd_house1_state': [150, 300, 1000, 3000],
        'redd_house1_state_average': [0.21, 216.75, 438.51, 1105.08],
        'redd_house2_state_num': 3,
        'redd_house2_state': [150, 1000, 3000],
        'redd_house2_state_average': [0.16, 250.26, 1197.93],
        'redd_house3_state_num': 3,
        'redd_house3_state': [50, 400, 1000],
        'redd_house3_state_average': [0.97, 195.6, 743.42],
        'uk_state_num': 3,
        'uk_state': [50, 1000, 4500],
        'uk_state_average': [0.89, 122.56, 2324.9],
        's2s_length': 1536,
        'lmda': 1.8,
        'lmda_redd': 30
    },
    'washingmachine': {
        'window_len': 599,
        'redd_on_power_threshold': 500,
        'uk_on_power_threshold': 50,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        'redd_state_num': 2,
        'redd_state': [500, 5000],
        'redd_state_average': [0, 2627.3],
        'uk_state_num': 4,
        # 'uk_state': [50, 800, 3500],
        # 'uk_state_average': [0.13, 204.64, 1892.85],
        'uk_house2_state_num': 4,
        'uk_house2_state': [50, 200, 1000, 4000],
        'uk_house2_state_average': [2.83, 114.34, 330.25, 2100.14],
        's2s_length': 2000,
        'lmda': 3,
        'lmda_redd': 1 / 170
    },
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance_name',
                        type=str,
                        default='dishwasher',
                        help='the name of target appliance')
    parser.add_argument('--data_dir',
                        type=str,
                        default='/redd/',
                        help='this is the directory of the training samples')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='The batch size of training examples')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=20,
                        help='The number of epoches.')
    parser.add_argument('--patience',
                        type=int,
                        default=1)
    parser.add_argument('--seed',
                        type=int,
                        default=819)
    return parser.parse_args()


args = get_arguments()
# save_path='/result/redd_fa_132_'+str(args.seed)+'_'
save_path = './result/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_dataset():   

    import pandas as pd
    path = f'./REDD/{args.appliance_name}'    
    
    train = pd.read_csv(os.path.join(path, f'{args.appliance_name}_training_.csv'), header=None).to_numpy()
    val = pd.read_csv(os.path.join(path, f'{args.appliance_name}_validation_.csv'), header=None).to_numpy()
    test = pd.read_csv(os.path.join(path, f'{args.appliance_name}_test_.csv'), header = None).to_numpy()

    tra_set_x, tra_set_y, tra_set_s = train[:, 0], train[:, 1], train[:, 2]
    val_set_x, val_set_y, val_set_s = val[:, 0],  val[:, 1], val[:, 2]
    test_set_x, test_set_y, test_set_s = test[:, 0], test[:, 1], test[:, 2]
    print('tra_set_x.shape: ', tra_set_s.shape)
    print('val_set_x.shape: ', val_set_s.shape)
    print('test_set_x.shape: ', test_set_s.shape)
    
    return tra_set_x, tra_set_y, tra_set_s, val_set_x, val_set_y, val_set_s, test_set_x, test_set_y, test_set_s


# load the data set
tra_set_x, tra_set_y, tra_set_s, val_set_x, val_set_y, val_set_s, test_set_x, test_set_y, test_set_s = load_dataset()

# hyper parameters according to appliance
window_len = 599
state_num =  params_appliance[args.appliance_name]['redd_state_num']
print(state_num)

offset = int(0.5 * (window_len - 1.0))

tra_kwag = {
    'inputs': tra_set_x,
    'targets': tra_set_y,
    'targets_s': tra_set_s,
}
val_kwag = {
    'inputs': val_set_x,
    'targets': val_set_y,
    'targets_s': val_set_s,
}
test_kwag = {
    'inputs': test_set_x,
    'targets': test_set_y,
    'targets_s': test_set_s,
}
mean = params_appliance[args.appliance_name]['mean']
std = params_appliance[args.appliance_name]['std']
# threshold = (params_appliance[args.appliance_name]['redd_on_power_threshold'] - mean) / std
tra_provider = data_provider.S2P_State_Slider(batch_size=args.batch_size,
                                              shuffle=True, offset=offset, length=window_len)  # , threshold=threshold
val_provider = data_provider.S2P_State_Slider(batch_size=5000,
                                              shuffle=False, offset=offset, length=window_len)
test_provider = data_provider.S2P_State_Slider(batch_size=5000,
                                               shuffle=False, offset=offset, length=window_len)

m = model.TFFusionStateModel(state_num).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 假设你的模型是 m
num_params = count_parameters(m)
print(f"Total number of parameters: {num_params}")
exit()

_params = filter(lambda p: p.requires_grad, m.parameters())
optimizer = torch.optim.Adam(_params, lr=1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
mean = params_appliance[args.appliance_name]['mean']
std = params_appliance[args.appliance_name]['std']

# train & val
best_state_dict_path = 'state_dict/{}'.format(args.appliance_name) 
best_val_loss = float('inf')
best_val_epoch = -1

lmda = params_appliance[args.appliance_name]['lmda_redd']


for epoch in range(args.n_epoch):
    train_loss, n_batch_train = 0, 0
    for idx, batch in enumerate(tra_provider.feed(**tra_kwag)):
        m.train()
        optimizer.zero_grad()
        x_train, y_train, s_train = batch
        x_train = torch.tensor(x_train, dtype=torch.float, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float, device=device)
        s_train = torch.tensor(s_train, dtype=torch.long, device=device)
        op_train, os_train = m(x_train)

        op_train = torch.reshape(op_train, (op_train.shape[0], state_num))
        os_train = torch.reshape(os_train, (os_train.shape[0], state_num))
        # op_train.shape = [batch_size, out_len, state_num]
        # os_train.shape = [batch_size, out_len, state_num]
        oss_train = F.softmax(os_train, dim=-1)
        # oss_train.shape = [batch_size, out_len, state_num]
        o_train = torch.sum(oss_train * op_train, dim=-1, keepdim=False)
        # o_train.shape = [batch_size*out_len, state_num]
        s_train = s_train.flatten()
       
        loss = F.mse_loss(o_train.flatten(), y_train.flatten()) * lmda + F.cross_entropy(os_train, s_train)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batch_train += 1
    train_loss = train_loss / n_batch_train

    val_loss, n_batch_val = 0, 0
    
    mean_mse_loss, mean_cross_entropy_loss = 0, 0

    with torch.no_grad():
        for batch in val_provider.feed(**val_kwag):
            m.eval()
            x_val, y_val, s_val = batch
            x_val = torch.tensor(x_val, dtype=torch.float, device=device)
            y_val = torch.tensor(y_val, dtype=torch.float, device=device)
            s_val = torch.tensor(s_val, dtype=torch.long, device=device)
            op_val, os_val = m(x_val)
            op_val = torch.reshape(op_val, (op_val.shape[0],  state_num))
            os_val = torch.reshape(os_val, (os_val.shape[0],  state_num))
            oss_val = F.softmax(os_val, dim=-1)
            o_val = torch.sum(oss_val * op_val, dim=-1, keepdim=False)
            # val_loss += F.mse_loss(o_val.flatten(), y_val.flatten()).item() + F.cross_entropy(os_val, s_val.flatten()).item()
            mse_loss = F.mse_loss(o_val.flatten(), y_val.flatten()).item()
            cross_entropy_loss = F.cross_entropy(os_val, s_val.flatten()).item()
            val_loss = mse_loss * lmda + cross_entropy_loss
            mean_mse_loss += mse_loss * lmda
            mean_cross_entropy_loss += cross_entropy_loss
            n_batch_val += 1

    val_loss = val_loss / n_batch_val

    mean_mse_loss = mean_mse_loss / n_batch_val
    mean_cross_entropy_loss = mean_cross_entropy_loss / n_batch_val

    print('>>> Epoch {}: train loss {:.6f}, val loss {:.10f}, mean_mse_loss {:6f}, mean_cross_entropy_loss {:6f}'.format(epoch, train_loss, val_loss, mean_mse_loss, mean_cross_entropy_loss), flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch

        if not os.path.exists('state_dict/'):
            os.mkdir('state_dict/')
        torch.save(m.state_dict(), best_state_dict_path + '.pkl')

    elif best_val_epoch + args.patience < epoch:
        print('>>> Early stopping')
        break

    print('>>> Best model is at epoch {}'.format(best_val_epoch))
    lr_scheduler.step(best_val_loss)

# test
gt = test_set_y[offset : -offset]
gt_s = test_set_s[offset : -offset]

m.load_state_dict(torch.load(best_state_dict_path + '.pkl'))
m.eval()
pred = []
pred_p = []
pred_s = []


with torch.no_grad():
    for batch in test_provider.feed(**test_kwag):
        x_test, y_test, s_test = batch
        x_test = torch.tensor(x_test, dtype=torch.float, device=device)
        y_test = torch.tensor(y_test, dtype=torch.float, device=device)
        s_test = torch.tensor(s_test, dtype=torch.int, device=device)
        op_test, os_test = m(x_test)
        op_test = torch.reshape(op_test, (op_test.shape[0], state_num))
        os_test = torch.reshape(os_test, (os_test.shape[0], state_num))
        os_test = F.softmax(os_test, dim=-1)
        o_test = torch.sum(os_test * op_test, dim=-1, keepdim=False)

        o_test = o_test.cpu().numpy().reshape(-1)
        pred.append(o_test)
        pred_p.append(op_test.cpu().numpy())
        pred_s.append(os_test.cpu().numpy())

pred = np.concatenate(pred, axis=0)
pred_p = np.concatenate(pred_p, axis = 0).reshape(-1, state_num)
pred_sh = np.concatenate(pred_s, axis = 0).reshape(-1, state_num)


pred = pred * std + mean
pred[pred <= 0.0] = 0.0

gt = gt * std + mean
gt[gt <= 0.0] = 0.0

import metric

sample_second = 6.0  # sample time is 6 seconds
print('MAE:{0}'.format(metric.get_abs_error(gt.flatten(), pred.flatten())))
print('SAE:{0}'.format(metric.get_sae(gt.flatten(), pred.flatten(), sample_second)))
print('SAE_Delta:{}'.format(metric.get_sae_delta(gt.flatten(), pred.flatten(), 1200)))
print(metric.get_sae_delta(gt.flatten(), pred.flatten(), 600))
print('F1: {}'.format(metric.get_F1(gt.reshape(1, -1), pred.reshape(1, -1), args.appliance_name)))

savegt = gt.flatten()
savepred = pred.flatten()

savegt_s = gt_s.flatten()
savepred_s = pred_sh.flatten()

np.savetxt(save_path + args.appliance_name + '_pred.txt', savepred, fmt='%f', newline='\n')
np.savetxt(save_path + args.appliance_name + '_gt.txt', savegt, fmt='%f', newline='\n')
# np.savetxt(save_path+args.appliance_name+'_mains.txt',savemains,fmt='%f',newline='\n')
np.savetxt(save_path + args.appliance_name + '_pred_p.txt', pred_p, fmt='%f', newline='\n')
np.savetxt(save_path + args.appliance_name + '_gt_s.txt', savegt_s, fmt='%d', newline='\n')
np.savetxt(save_path + args.appliance_name + '_pred_s.txt', savepred_s, fmt='%d', newline='\n')
# np.savetxt(save_path + args.appliance_name + '_pred_sp.txt', pred_s, fmt='%f', newline='\n'