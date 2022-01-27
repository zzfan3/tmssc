
import torch
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
import scipy.io

#导入数据集并准备
# handwritten
from torch.utils.data import random_split

data = scipy.io.loadmat('handwritten_6views.mat')
test_label = data['gt_test']
test_label = test_label - 1
train_label = data['gt_train']
train_label = train_label - 1
test_data = data['x1_test']
train_data = data['x1_train']
dat_x = np.concatenate((train_data, test_data),axis=0)
dat_y = np.concatenate((train_label, test_label),axis=0)
dat_x = dat_x.astype(np.float64)
dat_y = (np.array(dat_y.T)).reshape(-1)
dat_y = dat_y.astype(np.int32)

##初始化参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
print(torch.cuda.is_available())
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)
#length = 9
num_class = 5 #2
batch_size = 40
num_epochs = 1000
p = 0.8
pm = 0.65
learning_rate = 0.001
NumFuzz = 20
NumRule = 10 #8
NumEnhan = 10
random_state = check_random_state(0)#设置随机种子

permutation = random_state.permutation(dat_x.shape[0])
dat_x = dat_x[permutation]
dat_y = dat_y[permutation]
dat_x = preprocessing.scale(dat_x)
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

model_student = Classifier()
model_teacher = Classifier()

data_set = myDataset(dat_x,dat_y,"D:/python35/2020RAtrain",NumFuzz,NumRule,Alpha,WeightEnhan)
train_dataset,test_dataset = random_split(data_set, [840,360])
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

optimizer = torch.optim.Adam(model_student.parameters(), lr=learning_rate)

total_step = len(train_loader)

# 以下部分为MTFBLS，增加teacher student 模型的输出矩阵计算部分
for epoch in range(num_epochs):
    for (data1, labels1), (data2, labels2) in zip(train_loader, train_no_loader):
        # for i,(data1, labels1) in enumerate(train_loader):
        # labels = labels.reshape(len(train_dataset), 1)
        print(labels1)
        labels1 = labels1.long().random_(1)
        print(labels1)
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

with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in train_no_loader:

        labels = labels.long()
        outputs = model_teacher(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(predicted)
    print('Train Accuracy of the student_model on the 1600 train images: {} %'.format(100 * correct / total))
