import torch
import torch.nn as nn
import torch.nn.functional as F

# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def softmax_mse_loss(input_logits, target_logits):

    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    D = softmax_mse_loss(outputs, output2)  # F.mse_loss(output1, output2,reduction='mean')#
    return (A + B) + C + D

class MT_FBLS(nn.module):

    def __init__(self, classes, views, classifier_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(MT_FBLS, self).__init__()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

    def DS_Combin(alpha,classes):

        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2, classes):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = torch.Tensor(alpha1), torch.Tensor(alpha2)
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1], classes)
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1], classes)
        return alpha_a

    def trusted_net(evidence, classes):
        alpha = dict()
        for v_num in range(5): #how many views
            alpha[v_num] = evidence[v_num] + 1
        alpha_a = DS_Combin(alpha, classes) #how many classes
        evidence_a = alpha_a - 1


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

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(NumRule * NumFuzz + NumEnhan, num_class)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        return x


