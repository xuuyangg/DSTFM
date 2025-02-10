# -*- coding: utf-8 -*-

import os
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import data_provider
import s2pmodel as model
from torch.utils.tensorboard import SummaryWriter
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
        's2s_length': 128
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
        's2s_length': 128
    },
    'fridge': {
        'window_len': 599,
        'redd_on_power_threshold': 50,
        'uk_on_power_threshold': 20,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
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
        'uk_state_num': 3,
        'uk_state': [20, 200, 2500],
        'uk_state_average': [0.13, 87.26, 246.5],
        's2s_length': 512
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
        's2s_length': 1536
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
        'uk_state_num': 3,
        'uk_state': [50, 800, 3500],
        'uk_state_average': [0.13, 204.64, 1892.85],
        'uk_house2_state_num': 4,
        'uk_house2_state': [50, 200, 1000, 4000],
        'uk_house2_state_average': [2.83, 114.34, 330.25, 2100.14],
        's2s_length': 2000
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
    path = f'./UK_DALE'    
    
    train = pd.read_csv(os.path.join(path, f'{args.appliance_name}_training_.csv'), header=None).to_numpy()
    val = pd.read_csv(os.path.join(path, f'{args.appliance_name}_validation_.csv'), header=None).to_numpy()
    test = pd.read_csv(os.path.join(path, f'{args.appliance_name}_test_.csv'), header = None).to_numpy()
    tra_set_x, tra_set_y = train[:, 0], train[:, 1]
    val_set_x, val_set_y = val[:, 0],  val[:, 1]
    test_set_x, test_set_y = test[:, 0], test[:, 1]
    
    print('training set:', tra_set_x.shape, tra_set_y.shape)
    print('validation set:', val_set_x.shape, val_set_y.shape)
    print('testing set:', test_set_x.shape, test_set_y.shape)
    
    return tra_set_x, tra_set_y, val_set_x,  val_set_y,  test_set_x, test_set_y


# load the data set
tra_set_x, tra_set_y, val_set_x, val_set_y, test_set_x, test_set_y = load_dataset()

# hyper parameters according to appliance
window_len = 599
offset = int(0.5 * (window_len - 1.0))

tra_kwag = {
    'inputs': tra_set_x,
    'targets': tra_set_y,
}
val_kwag = {
    'inputs': val_set_x,
    'targets': val_set_y,
}
test_kwag = {
    'inputs': test_set_x,
    'targets': test_set_y,
}
mean = params_appliance[args.appliance_name]['mean']
std = params_appliance[args.appliance_name]['std']
# threshold = (params_appliance[args.appliance_name]['redd_on_power_threshold'] - mean) / std
tra_provider = data_provider.S2P_Slider(batch_size=args.batch_size,
                                              shuffle=True, offset=offset, length=window_len)  # , threshold=threshold
val_provider = data_provider.S2P_Slider(batch_size=5000,
                                              shuffle=False, offset=offset, length=window_len)
test_provider = data_provider.S2P_Slider(batch_size=5000,
                                               shuffle=False, offset=offset, length=window_len)

m = model.FreqBaseModel().to(device)
_params = filter(lambda p: p.requires_grad, m.parameters())
optimizer = torch.optim.Adam(_params, lr=1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
mean = params_appliance[args.appliance_name]['mean']
std = params_appliance[args.appliance_name]['std']

# train & val
best_state_dict_path = 'state_dict/{}'.format(args.appliance_name)

best_val_loss = float('inf')
best_val_epoch = -1
writer = SummaryWriter(f'runs/{args.appliance_name}_no_position_training_logs')

scaler = GradScaler()  # 用于梯度缩放
best_val_loss = float('inf')
best_val_epoch = 0
for epoch in range(args.n_epoch):
    train_loss, n_batch_train = 0, 0
    for idx, batch in enumerate(tra_provider.feed(**tra_kwag)):
        m.train()
        optimizer.zero_grad()
        x_train, y_train = batch
        x_train = torch.tensor(x_train, dtype=torch.float, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float, device=device)
        # 使用 autocast 上下文管理器开启混合精度训练
        with autocast():
            p_train = m(x_train)
            loss = F.mse_loss(p_train, y_train)
        # 使用 scaler 进行梯度缩放和反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        n_batch_train += 1
    train_loss = train_loss / n_batch_train
    val_loss, n_batch_val = 0, 0
    with torch.no_grad():
        for batch in val_provider.feed(**val_kwag):
            m.eval()
            x_val, y_val = batch
            x_val = torch.tensor(x_val, dtype=torch.float, device=device)
            y_val = torch.tensor(y_val, dtype=torch.float, device=device)
            # 使用 autocast 上下文管理器开启混合精度推理
            with autocast():
                p_val = m(x_val)
                val_loss += F.mse_loss(p_val, y_val).item()
            n_batch_val += 1
    val_loss = val_loss / n_batch_val
    print('>>> Epoch {}: train mse loss {:.6f}, val mse loss {:.6f}'.format(epoch, train_loss, val_loss), flush=True)
    # 将训练和验证损失写入 TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
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

writer.close()
# test
gt = test_set_y[offset : -offset]
m.load_state_dict(torch.load(best_state_dict_path + '.pkl'))
m.eval()
datanum = 0
pred = []
with torch.no_grad():
    for batch in test_provider.feed(**test_kwag):
        x_test, y_test = batch
        x_test = torch.tensor(x_test, dtype=torch.float, device=device)
        y_test = torch.tensor(y_test, dtype=torch.float, device=device)
        p_test = m(x_test)
        p_test = p_test.cpu().numpy().reshape(-1)
        pred.append(p_test)

pred = np.concatenate(pred, axis=0)

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

# save the pred to files
# savemains = test_set_x[offset:-offset].flatten()*814+522
savegt = gt.flatten()
savepred = pred.flatten()

np.savetxt(save_path + args.appliance_name + '_pred.txt', savepred, fmt='%f', newline='\n')
np.savetxt(save_path + args.appliance_name + '_gt.txt', savegt, fmt='%f', newline='\n')
# np.savetxt(save_path+args.appliance_name+'_mains.txt',savemains,fmt='%f',newline='\n')
