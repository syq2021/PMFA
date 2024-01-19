import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



        
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CenterTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni,label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class CCenterTripletLoss(nn.Module):
    def __init__(self, k_size=8, margin=0):
        super(CCenterTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        centers = torch.stack(centers)

        dist_pc = (inputs - centers) ** 2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, centers, centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n, self.k_size):
            dist_an.append((self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean())
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = 0.1 * dist_pc.mean() + 0. * dist_an.mean()
        return loss, dist_pc.mean(), dist_an.mean()

class MMD_Loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_Loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) + 1e-9 for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val), L2_distance

    def forward(self, source, target):

        xx_batch, yy_batch, xy_batch, yx_batch = 0, 0, 0, 0

        batch_size = int(source.size()[0])
        kernels, L2dist = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                               fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        xx_batch = torch.mean(XX)
        yy_batch = torch.mean(YY)
        xy_batch = torch.mean(XY)
        yx_batch = torch.mean(YX)

        loss = torch.mean(XX + YY - XY - YX)
        return loss, torch.max(L2dist), [xx_batch, yy_batch, xy_batch, yx_batch]


class MarginMMD_Loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, P=8, K=4, margin=None):
        super(MarginMMD_Loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.P = P
        self.K = K
        self.margin = margin
        if self.margin:
            print(f'Using Margin : {self.margin}')
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) + 1e-9 for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        if torch.sum(torch.isnan(sum(kernel_val))):
            ## We encountered a Nan in Kernel
            print(f'Bandwidth List : {bandwidth_list}')
            print(f'L2 Distance : {L2_distance}')
            ## Check for Nan in L2 distance
            print(f'L2 Nan : {torch.sum(torch.isnan(L2_distance))}')
            for bandwidth_temp in bandwidth_list:
                print(f'Temp: {bandwidth_temp}')
                print(f'BW Nan : {torch.sum(torch.isnan(L2_distance / bandwidth_temp))}')
        return sum(kernel_val), L2_distance

    def forward(self, source, target, labels1=None, labels2=None):
        ## Source  - [P*K, 2048], Target - [P*K, 2048]
        ## Devide them in "P" groups of "K" images
        rgb_features_list, ir_features_list = list(torch.split(source, [self.K] * self.P, dim=0)), list(
            torch.split(target, [self.K] * self.P, dim=0))
        total_loss = torch.tensor([0.], requires_grad=True).to(torch.device('cuda'))
        if labels1 is not None and labels2 is not None:
            rgb_labels, ir_labels = torch.split(labels1, [self.K] * self.P, dim=0), torch.split(labels2,
                                                                                                [self.K] * self.P,
                                                                                                dim=0)
            print(f'RGB Labels : {rgb_labels}')
            print(f'IR Labels : {ir_labels}')

        xx_batch, yy_batch, xy_batch, yx_batch = 0, 0, 0, 0

        for rgb_feat, ir_feat in zip(rgb_features_list, ir_features_list):
            source, target = rgb_feat, ir_feat  ## 4, 2048 ; 4*2048 -> 4*2048
            ## (rgb, ir, mid) -> rgb - mid + ir- mid ->
            batch_size = int(source.size()[0])
            kernels, l2dist = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul,
                                                   kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]

            xx_batch += torch.mean(XX)
            yy_batch += torch.mean(YY)
            xy_batch += torch.mean(XY)
            yx_batch += torch.mean(YX)

            if self.margin:
                loss = torch.mean(XX + YY - XY - YX)
                if loss - self.margin > 0:
                    total_loss += loss
                else:
                    total_loss += torch.clamp(loss - self.margin, min=0)

            else:
                total_loss += torch.mean(XX + YY - XY - YX)

        total_loss /= self.P
        return total_loss, torch.max(l2dist), [xx_batch / self.P, yy_batch / self.P, xy_batch / self.P,
                                               yx_batch / self.P]

class Modal_Loss(nn.Module):
    def __init__(self):
        super(Modal_Loss,self).__init__()
        pass

    def forward(self, feats, modals=2):
        feat1, feat2 = feats.chunk(modals, 0)
        x = torch.mean(feat1, dim=0, keepdim=True)
        y = torch.mean(feat2, dim=0, keepdim=True)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

class reg_Loss(nn.Module):
    def __init__(self):
        super(reg_Loss,self).__init__()
        self.part_num = 6

    def forward(self, masks, b=64):
        loss_reg = torch.bmm(masks, masks.permute(0, 2, 1))
        loss_reg = torch.triu(loss_reg, diagonal=1).sum() / (b * self.part_num * (self.part_num - 1) / 2)

        return loss_reg


class OFPenalty(nn.Module):
    def __init__(self, of_beta):
        super(OFPenalty, self).__init__()

        self.beta = of_beta
        # self.softmax = Softmax(dim=-1)

    def dominant_eigenvalue(self, A):
        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        # x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

        return numerator / denominator

    def get_singular_values(self, A):
        AAT = torch.bmm(A, A.permute(0, 2, 1))  # C*S*S*C=C*C
        # AAT = self.softmax(AAT)
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def apply_penalty(self, x):
        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        singular_penalty = (largest - smallest) * self.beta

        singular_penalty *= 0.01

        return singular_penalty.sum() / (x.size(0) / 64.)  # Quirk: normalize to 32-batch case

    def forward(self, reg_feats):
        singular_penalty = sum([self.apply_penalty(x) for x in reg_feats])

        return singular_penalty