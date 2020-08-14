
# token = '5ca06ca5b5512817186315dd0063337d0c381006f6837285ed55abf5'
# ts.set_token(token)
# pro = ts.pro_api()
# df = pro.daily(ts_code='000001.SZ', start_date='20100101', end_date='')
# df = df.dropna()
# df.to_csv('SZ000001.csv')

# df.describe()
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import datetime
import numpy as np
import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torchvision
import torchvision.transforms as trans
import tushare as ts
# cons = ts.get_apis()
# df = ts.bar('000300', conn=cons, asset='INDEX', start_date='2010-01-01', end_date='')
# df = df.dropna()
# df.to_csv('sh300.csv')

N = 30
LR = 0.1
EPOCH = 200
batch_size = 32

device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')


class MyTrainSet(Dataset):
    def __init__(self, df_numpy):
        sampleNum = len(df_numpy) - N
        self.data = np.ndarray((N, sampleNum))
        self.label = np.zeros((sampleNum,))

        idx = 0
        while True:
            self.data[:, idx] = df_numpy[idx: idx + N]
            self.label[idx] = df_numpy[idx + N]
            idx += 1
            if idx == sampleNum:
                break
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1], -1))
        self.label = self.label.reshape((self.label.shape[0], -1))
        self.data = tc.tensor(self.data, dtype=tc.float32)
        self.label = tc.tensor(self.label, dtype=tc.float32)

    def __getitem__(self, index):
        return self.data[:, index, :], self.label[index, :]

    def __len__(self):
        return self.data.shape[1]


def make_dataset(filename):
    df = pd.read_csv(filename)
    df_numpy = np.array(df)
    df_numpy = df_numpy[:, 4]
    df_numpy = df_numpy[::-1]  # 倒序

    return MyTrainSet(df_numpy)


trainset = make_dataset('sh300.csv')
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)


class StockPredRNN(tc.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StockPredRNN, self).__init__()

        self.layer1 = nn.BatchNorm1d(N)
        self.layer2 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.layer3 = nn.BatchNorm1d(hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        BNx = self.layer1(x)
        rnn_out, _ = self.layer2(BNx)
        BN2 = self.layer3(rnn_out[:, -1, :])
        out = self.layer4(BN2)
        return out


rnn = StockPredRNN(1, 64)
optimizer = tc.optim.Adam(rnn.parameters(), lr=LR)
loss_fn = nn.MSELoss()

lossList = list()
for _ in range(EPOCH):
    epochLoss = 0
    for i, (data, label) in enumerate(trainloader):
        output = rnn(data)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochLoss += loss.item()

    mean_loss = epochLoss / len(trainloader) / batch_size
    print(mean_loss)

    lossList.append(mean_loss)




