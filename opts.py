import torch
import torch.nn as nn
import numpy as np


class Parameters_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, encoder1, encoder2, eps=1):
        '''
        :param encoder1:  nn.Module
        :param encoder2:  nn.Module
        :param eps:
        :return: cosine loss
        '''
        EW1 = None
        EW2 = None
        for p1, p2 in zip(encoder1.parameters(), encoder2.parameters()):
            if EW1 is None and EW2 is None:
                EW1 = p1.view(-1)
                EW2 = p2.view(-1)
            else:
                EW1 = torch.cat([EW1, p1.view(-1)], dim=0)
                EW2 = torch.cat([EW2, p2.view(-1)], dim=0)
        # print(EW1.shape)
        # print(EW2.shape)
        para_loss = torch.dot(EW1, EW2) / (EW1.norm() * EW2.norm() + eps)

        return para_loss*0.01

class VecSim_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, vec1, vec2, eps=1):
        '''
        :param encoder1:  nn.Module
        :param encoder2:  nn.Module
        :param eps:
        :return: cosine loss
        '''

        para_loss = torch.dot(EW1, EW2) / (EW1.norm() * EW2.norm() + eps)

        return para_loss





def contrastive_loss(logits, dim):
    neg_ce = torch.diag(nn.functional.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


class clip_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_f = contrastive_loss
    def forward(self,similarity):
        image_loss = contrastive_loss(similarity, dim=0)
        text_loss = contrastive_loss(similarity, dim=1)
        return (text_loss + image_loss) / 2.0

def pdist(x1, x2):
    """
        compute euclidean distance between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise euclidean distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = torch.sum(x1*x1, 1).view(-1, 1)
    x2_square = torch.sum(x2*x2, 1).view(1, -1)
    return torch.sqrt(x1_square - 2 * torch.mm(x1, x2.transpose(0, 1)) + x2_square + 1e-4)


def pdist_cos(x1, x2):
    """
        compute cosine similarity between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise cosine distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_norm = x1 / x1.norm(dim=1)[:, None]
    x2_norm = x2 / x2.norm(dim=1)[:, None]
    res = torch.mm(x1_norm, x2_norm.transpose(0, 1))
    mask = torch.isnan(res)
    res[mask] = 0
    return res

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    inputs: shape = (N,N)
    output: shape =(N,)
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        print(diagonal.shape)
        d1 = diagonal.expand_as(scores)
        print(d1.shape)
        d2 = diagonal.t().expand_as(scores)
        print(d2.shape)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        print(mask.shape)
        # if torch.cuda.is_available():
        #     I = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


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


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss



 ####
'''
相似性可以使用 KL 散度进行计算 
import torch.nn.functional as F
kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
或者下面的CMD loss
'''
###
class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

if __name__ == '__main__':
    ## parameter loss test
    encoder1 = nn.Sequential(
        nn.Linear(512, 2048),
        nn.LayerNorm(2048),
        nn.Linear(2048, 512)
    )
    encoder2 = nn.Sequential(
        nn.Linear(512, 2048),
        nn.LayerNorm(2048),
        nn.Linear(2048, 512)
    )
    p_loss = Parameters_loss()
    loss = p_loss(encoder1, encoder2)
    loss.backward()

    inp1 = torch.randn(16,512)
    inp2 = torch.rand(16, 512)

    diff_loss = DiffLoss()
    dif_loss = diff_loss(inp2,inp1)
