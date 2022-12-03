import math
import warnings
import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

warnings.simplefilter(action='ignore', category=RuntimeWarning)
torch.manual_seed(2022)
# np.random.seed(2022) # np的随机性。
# random.seed(2022) # python的随机性

#########
#本实验中的G-mean是我的评价指标
#########
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv1d(1, 16, 2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 16, 5])
            nn.Conv1d(16, 32, 2),
            nn.ReLU(),
            nn.MaxPool1d(4),  # torch.Size([128, 32, 1])
            nn.Flatten(),  # torch.Size([128, 32])    (假如上一步的结果为[128, 32, 2]， 那么铺平之后就是[128, 64])
        )
        self.model2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.reshape(-1,1,11)   #结果为[128,1,11]  目的是把二维变为三维数据
        x = self.model1(input)
        x = self.model2(x)
        return x

data = pd.read_csv("../dataset/train.csv")
dataset = data.values
X_train = dataset[:,1:12].astype(float)
Y_train = dataset[:,0:1]
encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train.ravel())

X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
train_dataset =  torch.utils.data.TensorDataset(X_train, Y_train)


data = pd.read_csv("../dataset/test.csv")
dataset = data.values
X_test = dataset[:,1:12].astype(float)
Y_test = dataset[:,0:1]
encoder = LabelEncoder()
Y_test = encoder.fit_transform(Y_test.ravel())
X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
test_dataset =  torch.utils.data.TensorDataset(X_test, Y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=4444, shuffle=True)

tudui = Tudui()
loss_function = nn.CrossEntropyLoss()

epochs = 1000
learning_rate = 0.001
optim = torch.optim.Adam(tudui.parameters(), lr=learning_rate)


for i in range(epochs):
    print("--------第{}轮训练开始---------".format(i+1))
    total_G_mean = 0

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    tudui.train()
    for data in train_loader:
        X_data, Y_data = data[0], data[1]
        output = tudui(X_data)
        loss = loss_function(output, Y_data)
        optim.zero_grad()
        loss.backward()
        optim.step()

        pred = output.argmax(axis=1)
        matrix = confusion_matrix(Y_data, pred)
        TN = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TP = matrix[1][1]

        FDR = TP / (TP + FN)
        FAR = FP / (FP + TN)
        P = TN / (TN + FP)
        G_mean = math.sqrt(FDR * P)
        if np.isnan(G_mean):
            G_mean = 0.0

        total_G_mean = total_G_mean + G_mean

    length = len(train_loader)
    total_G_mean = total_G_mean / length
    print("G-mean为{:.4f}".format(total_G_mean))


# 验证数据
    if (i + 1) % 10 == 0:
        G_mean_test = 0
        total_G_mean_test = 0
        total_FDR = 0
        total_FAR = 0
        tudui.eval()
        with torch.no_grad():
            for test in test_loader:
                X_test_data, Y_test_data = test[0], test[1]
                out = tudui(X_test_data)
                loss = loss_function(out, Y_test_data)

                pred_test = out.argmax(axis=1)
                matrix = confusion_matrix(Y_test_data, pred_test)
                TN = matrix[0][0]
                FP = matrix[0][1]
                FN = matrix[1][0]
                TP = matrix[1][1]
                FDR = TP / (TP + FN)
                FAR = FP / (FP + TN)
                P = TN / (TN + FP)
                G_mean_test = math.sqrt(FDR * P)
                if np.isnan(G_mean_test):
                    G_mean_test = 0.0
                total_G_mean_test = total_G_mean_test + G_mean_test

        total_G_mean_test = total_G_mean_test / len(test_loader)

        print("**********************验证数据***********************")
        print("G-mean在测试集上的表现为{:.4f}".format(total_G_mean_test))

# writer.close()
