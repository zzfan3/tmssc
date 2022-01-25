import math
#from tsk.classifier import TSK
from sklearn.decomposition import PCA
from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
from torch.optim import *
import math
import random
import numpy as np
#from clustering import CMeansClustering
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans
from torch.utils.data import random_split
import time
from sklearn.utils import check_random_state

import scipy.io

##import dataset and do some preprocessing
#handwritten
# data = scipy.io.loadmat('handwritten_6views.mat')
# test_label = data['gt_test']
# test_label = test_label - 1
# train_label = data['gt_train']
# train_label = train_label - 1
# test_data = data['x1_test']
# train_data = data['x1_train']
# dat_x = np.concatenate((train_data, test_data),axis=0)
# dat_y = np.concatenate((train_label, test_label),axis=0)
# dat_x = dat_x.astype(np.float64)
# dat_y = (np.array(dat_y.T)).reshape(-1)
# dat_y = dat_y.astype(np.int32)

#bbcsport
# data = scipy.io.loadmat('BBCSport.mat')
# label = data['bbcsport'][0][0]
# x1 = data['bbcsport'][1][0]
# x2 = data['bbcsport'][2][0]
# dat_x = x1
# dat_y = label
# dat_x = dat_x.astype(np.float64)
# dat_y = (np.array(dat_y.T)).reshape(-1)

#MSRC_v1
# data = scipy.io.loadmat('MSRC_v1.mat')
# label = data['gt'] - 1
# x1 = data['fea'][0][0]
# x2 = data['fea'][0][1]
# x3 = data['fea'][0][2]
# x4 = data['fea'][0][3]
# x5 = data['fea'][0][4]
# dat_x = x3
# dat_y = label
# dat_x = dat_x.astype(np.float64)
# dat_y = (np.array(dat_y.T)).reshape(-1)

#caltech101

data = scipy.io.loadmat('Caltech101-all.mat')
dat_x_1 = []
dat_y_1 = []
y = data['Y']
for i in range(len(y)):
    if y[i] == 33:
        dat_x_1.append(data['X'][0][5][i,:])
        dat_y_1.append(0)
    if y[i]==2:
        dat_x_1.append(data['X'][0][5][i,:])
        dat_y_1.append(1)
    if y[i]==45:
        dat_x_1.append(data['X'][0][5][i,:])
        dat_y_1.append(2)
    if y[i]==5:
        dat_x_1.append(data['X'][0][5][i,:])
        dat_y_1.append(3)
    if y[i]==85:
        dat_x_1.append(data['X'][0][5][i,:])
        dat_y_1.append(4)
    if y[i]==90:
        dat_x_1.append(data['X'][0][5][i,:])
        dat_y_1.append(5)
    if y[i]==100:
        dat_x_1.append(data['X'][0][5][i,:])
        dat_y_1.append(6)

dat_x = np.array(dat_x_1)
dat_y = np.array(dat_y_1)
# dat_x = x3
# dat_y = label
# dat_x = dat_x.astype(np.float64)
# dat_y = (np.array(dat_y.T)).reshape(-1)



##初始化参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
print(torch.cuda.is_available())
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)
#length = 9
num_class = 7 #2 #6 for reuter,7 for msrc, 7 for caltech101
batch_size = 40
num_epochs = 26000
p = 0.5
pm = 0.65 #0.65
learning_rate = 0.05
NumFuzz = 10
NumRule = 10 #8
NumEnhan = 20
random_state = check_random_state(0) #setup a random seed

permutation = random_state.permutation(dat_x.shape[0])
dat_x = dat_x[permutation]
dat_y = dat_y[permutation]
#dat_x = preprocessing.scale(dat_x)
#rng = np.random.default_rng()

Center=[]
for i in range(NumFuzz):
    kmeans = KMeans(n_clusters=NumRule,init='random',n_init=1).fit(dat_x)#d[0][:,1:])
    centers = kmeans.cluster_centers_
    Center.append(centers)
Center = np.array(Center)

input_dim = dat_x.shape[1]#d[0][:,1:].shape[1]
Alpha = np.random.rand(NumFuzz,input_dim*NumRule,NumRule)
WeightEnhan = np.random.rand(NumFuzz*NumRule+1,NumEnhan)


def compute_firing_level(data: np.ndarray, centers: int, delta: float) -> np.ndarray:
    """
    Compute firing strength using Gaussian model

    :param data: n_Samples * n_Features
    :param centers: data center，n_Clusters * n_Features
    :param delta: variance of each feature， n_Clusters * n_Features
    :return: firing strength
    """

    d = -(np.expand_dims(data, axis=2) - np.expand_dims(centers.T, axis=0)) ** 2 / (2 * delta)

    d = np.exp(np.sum(d, axis=1))

    d = np.fmax(d, np.finfo(np.float64).eps)

    return d / np.sum(d, axis=1, keepdims=True)


def apply_firing_level_o(x: np.ndarray, firing_levels: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Convert raw input to tsk input, based on the provided firing levels
    :param x: (np.ndarray) Raw input
    :param firing_levels: (np.ndarray) Firing level for each rule
    :param order: (int) TSK order. Valid values are 0 and 1
    :return:
    """
    if order == 0:
        return firing_levels
    else:
        n = x.shape[0]
        firing_levels = np.expand_dims(firing_levels, axis=1)
        x = np.expand_dims(x, axis=2)
        x = np.repeat(x, repeats=firing_levels.shape[2], axis=2)

        output = x * firing_levels

        output = output.reshape([n, -1])

        return output

def normalize(data):
    Dmax,Dmin = data.max(axis=0),data.min(axis=0)
    data = (data-Dmin)/(Dmax-Dmin)
    return data


def FBLS_pre(data, numfuzz, numrule, alpha, weightEnhan):
    y = np.zeros((data.shape[0], numfuzz * numrule))

    for i in range(numfuzz):
        a = Alpha[i]
        mu_a = compute_firing_level(data, Center[i], 1)
        d = apply_firing_level_o(data, mu_a, 1)
        T1 = d @ a
        T1 = normalize(T1)
        y[:, numrule * (i):numrule * (i + 1)] = T1
    H = np.concatenate((y, np.ones([y.shape[0], 1])), axis=1)
    T = H @ weightEnhan
    l = np.max(T)
    l1 = 0.8 / l
    T = np.tanh(T * l1)
    T2 = np.concatenate((y, T), axis=1)
    return T2

def update_ema_variables(model, ema_model,alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

class myDataset(Dataset):
    def __init__(self,datas,labels,root_dir,NumFuzz,NumRule,Alpha,WeightEnhan):
        #data_set = np.loadtxt(open(path, "rb"), delimiter = ",", skiprows = 0)
        #self.data = data_set[:,1:]
        self.data = datas

        self.data = FBLS_pre(self.data,NumFuzz,NumRule,Alpha,WeightEnhan)
        self.label = labels
        #self.label = data_set[:,0]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        data = (self.data[idx],self.label[idx])
        return data

data_set = myDataset(dat_x,dat_y,"D:/python35/2020RAtrain",NumFuzz,NumRule,Alpha,WeightEnhan)
train_dataset,test_dataset = random_split(data_set, [1189,300])
train_dataset,train_no_dataset = random_split(train_dataset, [len(train_dataset)-int(len(train_dataset)*p),int(len(train_dataset)*p)])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=len(train_dataset),
                                           shuffle=True)
train_no_loader = torch.utils.data.DataLoader(dataset=train_no_dataset,
                                          batch_size=len(train_no_dataset),#40,#
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=len(test_dataset),
                                          shuffle=False)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(NumRule * NumFuzz + NumEnhan, num_class)

        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)

        return x

def softmax_mse_loss(input_logits, target_logits):

    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


class TSLoss(torch.nn.Module):

    def __init__(self):
        super(TSLoss, self).__init__()

    def forward(self, output1, outputs, output2, label1):
        # loss1 = F.mse_loss(output1_w, label1,reduction='mean')
        loss1 = F.cross_entropy(output1, label1)
        loss2 = softmax_mse_loss(outputs, output2)  # F.mse_loss(output1, output2,reduction='mean')#

        return loss1 + loss2

model_student = NN()
model_teacher = NN()
#criterion = nn.MSELoss(reduction='mean')#nn.CrossEntropyLoss()#
#criterion =nn.CrossEntropyLoss()
criterion = TSLoss()
#criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model_student.parameters(), lr=learning_rate)

total_step = len(train_loader)

# below is the core code for MTFBLS，add calculation formular of the output matrix for teacher student model
for epoch in range(num_epochs):
    for (data1, labels1), (data2, labels2) in zip(train_loader, train_no_loader):
        # for i,(data1, labels1) in enumerate(train_loader):
        # labels = labels.reshape(len(train_dataset), 1)
        # print(labels1)
        labels1 = labels1.long()
        # print(labels1)
        # y = torch.zeros(len(train_dataset),num_class).scatter_(1,labels,1)
        outputs1 = model_student(data1)

        outputs = model_student(data2)
        outputs2 = model_teacher(data2)

        # loss_t = criterion(outputs1,labels)
        loss_t = criterion(outputs1, outputs, outputs2, labels1)
        alpha = min(1 - 1 / (epoch + 1), pm)
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        update_ema_variables(model_student, model_teacher, alpha)
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},alpha:{}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss_t.item(), alpha))

model_student.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in train_loader:

        labels = labels.long()
        outputs = model_student(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Train Accuracy of the student_model on the 1600 train images: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in train_loader:

        labels = labels.long()
        outputs = model_teacher(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Train Accuracy of the teacher_model on the 1600 train images: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        labels = labels.long()
        outputs = model_student(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the student model on the 400 test images: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        #data = data.reshape(input_dim,-1 , 1)
        #data = data.reshape(-1, 1500)
        labels = labels.long()
        outputs = model_teacher(data)
        #outputs = outputs.reshape(-1,num_class)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the teacher model on the 400 test images: {} %'.format(100 * correct / total))

