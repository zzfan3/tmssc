import scipy.io
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from torch.autograd import Variable
from torch.optim import optimizer
from torch.utils.data import random_split

from semi_model import semi_model, myDataset, NN
import torch

def normalize(x, min=0):
    if min == 0:
        scaler =  MinMaxScaler([0,1])
    else:
        scaler = MinMaxScaler([-1, 1])
    norm_x = scaler.fit_transform(x)
    return norm_x

data = scipy.io.loadmat('handwritten_6views.mat')
view_number = int((len(data)-5)/2)
dims = [[240], [76], [216], [47], [64], [6]] #change manualy according to the dataset you use
views = len(dims)
classes = 10 #change manualy according to the dataset you use

## Get train data
X = dict()
for v_num in range(view_number):
    X[v_num] = normalize(data['x'+str(v_num+1)+'_train'])
y = data['gt_train']
if np.min(y) == 1:
    y = y - 1
tmp = np.zeros(y.shape[0])
y = np.reshape(y, np.shape(tmp))

##get test data
Xt = dict()
for v_num in range(view_number):
    Xt[v_num] = normalize(data['x'+str(v_num+1)+'_test'])
yt = data['gt_test']
if np.min(yt) == 1:
    yt = yt - 1  # let the class label start from 0 ...
tmp1 = np.zeros(yt.shape[0])
yt = np.reshape(yt, np.shape(tmp1))


evidence = dict()
alpha = dict()
for v_num in range(view_number):
    evidence[v_num] = semi_model(X[v_num]) #calculate evidence matrix for each view by semi_model
    alpha[v_num] = evidence[v_num] + 1

alpha_a = semi_model.DS_Combin(alpha, classes)
evidence_a = alpha_a - 1
_, predicted = torch.max(evidence_a, 1)

# at the beginning:
start_time = time.time()
##初始化参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
print(torch.cuda.is_available())
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)
#length = 9
num_class = 2 #2 #6 for reuter,7 for msrc, 7 for caltech101, 10 for hw, 2 for bbcnews, 6 for reuters_new, 3 for sonar
batch_size = 40
num_epochs = 16000
p = 0.1
pm = 0.65 #0.65
learning_rate = 0.06
NumFuzz = 3
NumRule = 6 #8
NumEnhan = 10
random_state = check_random_state(0) #setup a random seed
permutation = random_state.permutation(Xt[v_num].shape[0])
dat_x = Xt[v_num][permutation]
dat_y = yt[permutation]
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


data_set = myDataset(Xt[v_num],yt,"",NumFuzz,NumRule,Alpha,WeightEnhan)
train_dataset,test_dataset = random_split(data_set, [158,50])  #[1189,300]for caltech 1600,400 for hw, [100,45] for bbcnews, [900,300] for reuters, [158,50] fro sonar
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

model_student = NN()
model_teacher = NN()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(epoch):
    semi_model.train()
    loss_meter = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        target = Variable(target.long().cuda())
        # refresh the optimizer
        optimizer.zero_grad()
        evidences, evidence_a, loss = semi_model(data, target)
        # print(evidences)
        # compute gradients and take step
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    print(evidences)


def test(epoch):
    semi_model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            evidences, evidence_a, loss = semi_model(data, target)
            # print(evidences)
            _, predicted = torch.max(evidence_a.data, 1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())

    print('====> acc: {:.4f}'.format(correct_num / data_num))
    return loss_meter.avg, correct_num / data_num
    print(evidences)

# at the end of the program:
print("%f seconds" % (time.time() - start_time))