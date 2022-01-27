import torch
import scipy.io

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
  
  def DS_Combin(alpha,classes):

    for v in range(len(alpha)-1):
        if v==0:
            alpha_a = DS_Combin_two(alpha[0], alpha[1], classes)
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v+1], classes)
    return alpha_a
  
  data1 = scipy.io.loadmat('evidence1_05.mat')
data2 = scipy.io.loadmat('evidence2_05.mat')
data3 = scipy.io.loadmat('evidence3_05.mat')
data4 = scipy.io.loadmat('evidence4_05.mat')
data5 = scipy.io.loadmat('evidence5_05.mat')
# data6 = scipy.io.loadmat('evidence6_01.mat')
evidence = dict()
evidence[0] = data1['NetoutTest']
evidence[1] = data2['NetoutTest']
evidence[2] = data3['NetoutTest']
evidence[3] = data4['NetoutTest']
evidence[4] = data5['NetoutTest']
# evidence[5] = data6['NetoutTest']

alpha = dict()
for v_num in range(5): #how many views
    alpha[v_num] = evidence[v_num] + 1
alpha_a = DS_Combin(alpha, 7) #how many classes
evidence_a = alpha_a - 1
_, predicted = torch.max(evidence_a, 1)
predicted.type()
y = scipy.io.loadmat('truth.mat')
target = y['test_yy']-1
target = torch.Tensor(target)
correct_num= 0
correct_num += (predicted == target.T).sum().item()
print('====> acc: {:.4f}'.format(correct_num/target.shape[0]))
