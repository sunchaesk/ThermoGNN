
import tqdm

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing

import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split

from data_collect import Buffer
import pickle

#load ./data/training_data.pt
buf = Buffer(500000000000, [1,1],[1,1], '','')
# all_a_values = []
# all_b_values = []

# for sublist in data:
#     for a, b in sublist:
#         all_a_values.append(a)
#         all_b_values.append(b)
load_preprocessed_data = True
if not load_preprocessed_data:
    p_f = open('./data/training_data.pt', 'rb')
    buf = pickle.load(p_f)
    p_f.close()
    print('training_data.pt opened')
    X = []
    y = []
    for episode_data in buf.buffer:
        for X_data, y_data in episode_data:
            X.append(X_data)
            y.append(y_data)

    print('PROCESSED DATA')
else:
    p_f = open('./data/processed_training_data2249568.pt', 'rb')
    processed_training_data = pickle.load(p_f)
    p_f.close()
    X = processed_training_data[0]
    y = processed_training_data[1]
    print("LOADED PROCESSED DATA")
print('data_processed')


save = False
if load_preprocessed_data:
    save = False
if save:
    # save preprocessed testing format
    save_data = [X, y]
    p_f = open('./data/processed_training_data' + str(len(X)) + '.pt', 'wb')
    pickle.dump(save_data, p_f)
    p_f.close()
print('processed training data saved')

# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('scaled data')

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


#input_dim = len(X[0][0])
class ThermoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self(ThermoMLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.b1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.b2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.lin1(x)
        out = self.b1(out)
        out = self.act1(out)
        out = self.lin2(out)
        out = self.b2(out)
        out = self.act2(out)
        out = self.out(out)
        return out

class ThermoRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ThermoRNN, self).__init__()

        self.rnn1 = nn.RNN(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1= nn.ReLU()
        self.rnn2 = nn.RNN(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn1(x)  # RNN returns output and hidden state, we only need the output
        #out = out.permute(1, 2, 0)  # Reshape to (batch_size, hidden_dim, seq_length)
        out = self.batch_norm1(out)
        out = self.act1(out)
        out, _ = self.rnn2(out)  # RNN returns output and hidden state, we only need the output
        #out = out.permute(1, 2, 0)  # Reshape to (batch_size, hidden_dim, seq_length)
        out = self.batch_norm2(out)
        out = self.act2(out)
        #out = out[:, -1, :]  # Select only the last time-step's output for each sample in the batch
        out = self.out(out)
        return out


input_dim = 35
hidden_dim = 30
model = ThermoRNN(input_dim, hidden_dim)
# model = nn.Sequential(
#     nn.RNN(input_dim, hidden_dim),
#     nn.BatchNorm1d(hidden_dim),
#     nn.ReLU(),
#     nn.RNN(hidden_dim, hidden_dim),
#     nn.BatchNorm1d(hidden_dim),
#     nn.ReLU(),
#     nn.Linear(hidden_dim, 1)
# )

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training parameters
n_epochs = 100000  # number of epochs to run
#batch_size = int(len(X) * 0.3)  # size of each batch
batch_size = 1000
print('...batch size:', batch_size)
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

# training loop
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            print('start:', start)
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss = torch.sqrt(loss)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    print('MSE epoch#{}: {}'.format(epoch, mse))
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(model, './data/training_model_temp')

# restore model and return best accuracy
model.load_state_dict(best_weights)
torch.save(model, './data/training_model_' + str(best_mse))
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()


# MSE epoch#0: 279.91357421875
# MSE epoch#1: 99.11660766601562
# MSE epoch#2: 2.1106603145599365
# MSE epoch#3: 0.7359910011291504
# MSE epoch#4: 0.6471648812294006
# MSE epoch#5: 0.582144558429718
# MSE epoch#6: 0.5447712540626526
# MSE epoch#7: 0.5321786999702454
# MSE epoch#8: 0.5337067246437073
# MSE epoch#9: 0.5050747394561768
# MSE epoch#10: 0.47606927156448364
# MSE epoch#11: 0.44648465514183044
# MSE epoch#12: 0.4330050051212311
# MSE epoch#13: 0.41482919454574585
# MSE epoch#14: 0.40628987550735474
# MSE epoch#15: 0.39626380801200867
# MSE epoch#16: 0.38430753350257874
# MSE epoch#17: 0.37527620792388916
# MSE epoch#18: 0.3677639365196228
# MSE epoch#19: 0.35860124230384827
# MSE epoch#20: 0.3500438332557678
# MSE epoch#21: 0.3401944041252136
# MSE epoch#22: 0.3325343430042267
# MSE epoch#23: 0.3236319124698639
# MSE epoch#24: 0.3174102008342743
# MSE epoch#25: 0.30783453583717346
# MSE epoch#26: 0.2997978925704956
# MSE epoch#27: 0.2928401231765747
# MSE epoch#28: 0.28581702709198
# MSE epoch#29: 0.2798464596271515
# MSE epoch#30: 0.27530694007873535
# MSE epoch#31: 0.2687636613845825
# MSE epoch#32: 0.26320746541023254
# MSE epoch#33: 0.25772085785865784
# MSE epoch#34: 0.2519882321357727
# MSE epoch#35: 0.2474362999200821
# MSE epoch#36: 0.2428051233291626
# MSE epoch#37: 0.2357606440782547
# MSE epoch#38: 0.2308056503534317
# MSE epoch#39: 0.22658979892730713
# MSE epoch#40: 0.22069793939590454
# MSE epoch#41: 0.21512576937675476
# MSE epoch#42: 0.21201205253601074
# MSE epoch#43: 0.20763860642910004
# MSE epoch#44: 0.20398665964603424
# MSE epoch#45: 0.20167547464370728
# MSE epoch#46: 0.19772227108478546
# MSE epoch#47: 0.1941882073879242
# MSE epoch#48: 0.19159291684627533
# MSE epoch#49: 0.18977026641368866
# MSE epoch#50: 0.18968965113162994
# MSE epoch#51: 0.186818465590477
# MSE epoch#52: 0.18253560364246368
# MSE epoch#53: 0.18273141980171204
# MSE epoch#54: 0.17992018163204193
# MSE epoch#55: 0.179151251912117
# MSE epoch#56: 0.17741085588932037
# MSE epoch#57: 0.17618899047374725
# MSE epoch#58: 0.17364738881587982
# MSE epoch#59: 0.17169316112995148
# MSE epoch#60: 0.17000272870063782
# MSE epoch#61: 0.16782443225383759
# MSE epoch#62: 0.166261225938797
# MSE epoch#63: 0.16518989205360413
# MSE epoch#64: 0.16337771713733673
# MSE epoch#65: 0.16240669786930084
# MSE epoch#66: 0.16060732305049896
# MSE epoch#67: 0.1586012840270996
# MSE epoch#68: 0.1577826887369156
# MSE epoch#69: 0.1561082899570465
# MSE epoch#70: 0.1548057496547699
# MSE epoch#71: 0.15264850854873657
# MSE epoch#72: 0.15174312889575958
# MSE epoch#73: 0.149786114692688
# MSE epoch#74: 0.14905056357383728
# MSE epoch#75: 0.14759723842144012
# MSE epoch#76: 0.14564263820648193
# MSE epoch#77: 0.14528663456439972
# MSE epoch#78: 0.14413289725780487
# MSE epoch#79: 0.1427122801542282
# MSE epoch#80: 0.14154109358787537
# MSE epoch#81: 0.1403668224811554
# MSE epoch#82: 0.1397613286972046
# MSE epoch#83: 0.1386665552854538
# MSE epoch#84: 0.13791652023792267
# MSE epoch#85: 0.13686537742614746
# MSE epoch#86: 0.1359003484249115
# MSE epoch#87: 0.1354893445968628
# MSE epoch#88: 0.13517525792121887
# MSE epoch#89: 0.13272690773010254
# MSE epoch#90: 0.13351015746593475
# MSE epoch#91: 0.13210254907608032
# MSE epoch#92: 0.13208910822868347
# MSE epoch#93: 0.13114215433597565
# MSE epoch#94: 0.1307377964258194
# MSE epoch#95: 0.12984313070774078
# MSE epoch#96: 0.1296548843383789
# MSE epoch#97: 0.12944450974464417
# MSE epoch#98: 0.12883687019348145
# MSE epoch#99: 0.12876014411449432
# MSE epoch#100: 0.12835627794265747
# MSE epoch#101: 0.1275915950536728
# MSE epoch#102: 0.1271650642156601
# MSE epoch#103: 0.12752586603164673
# MSE epoch#104: 0.1275760531425476
# MSE epoch#105: 0.12732616066932678
# MSE epoch#106: 0.1270572692155838
# MSE epoch#107: 0.12741956114768982
# MSE epoch#108: 0.1267528086900711
# MSE epoch#109: 0.1282249242067337
# MSE epoch#110: 0.12659890949726105
# MSE epoch#111: 0.12685780227184296
# MSE epoch#112: 0.126753032207489
# MSE epoch#113: 0.12545308470726013
# MSE epoch#114: 0.12558528780937195
# MSE epoch#115: 0.12549415230751038
# MSE epoch#116: 0.12627847492694855
# MSE epoch#117: 0.12490817159414291
