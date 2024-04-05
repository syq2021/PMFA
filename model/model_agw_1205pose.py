import torch
import torch.nn as nn
from torch.nn import init
from model.backbone.resnet import resnet50, resnet18
from model.backbone.resnet_ibn import resnet50_ibn_a
from model import ScoremapComputer, compute_local_features
from torch.nn import functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()

    # (N,C,H,W)=(N,2048,7,7)
    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,kernel_size= (3, inp_size[3]))#(kernel_size=(1,7))

#####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)


class MAM(nn.Module):
    def __init__(self, dim, r=16):
        super(MAM, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask = self.channel_attention(pooled)
        x = x * mask + x

        return x

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class base_resnet_ibn(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_ibn, self).__init__()

        model_base = resnet50_ibn_a(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local = 'on', gm_pool = 'on', pcb = 'on', aligned = False, arch ='resnet50', num_strips = 6, local_feat_dim = 256):
        super(embed_net, self).__init__()

        self.pcb = pcb
        self.num_stripes = num_strips
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        # self.base_resnet = base_resnet_ibn(arch=arch)

        pool_dim = 2048*2
        self.l2norm = Normalize(2)


        self.bottleneck = nn.BatchNorm1d(2*2048)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck1 = nn.BatchNorm1d(2048)
        self.bottleneck1.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_2 = nn.AdaptiveMaxPool2d((1, 1))
        self.gm_pool = gm_pool
        # self.MAM = MAM(2048)
        self._init_device()
        # keypoints model
        self.scoremap_computer = ScoremapComputer(10.0).to(self.device)
        # self.scoremap_computer = nn.DataParallel(self.scoremap_computer).to(self.device)
        self.scoremap_computer = self.scoremap_computer.eval()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def _init_device(self):
        self.device = torch.device('cuda')
    def forward(self, x1, x2, modal=0):

        if modal == 0:
            x_k = torch.cat((x1, x2), dim=0)
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x_k = x1
            x = self.visible_module(x1)
        elif modal == 2:
            x_k = x2
            x = self.thermal_module(x2)
        x = self.base_resnet(x)

        with torch.no_grad():
            score_maps, keypoints_confidence, _ = self.scoremap_computer(x_k)
        # x_m = self.MAM(x)
        feature_vector_list = compute_local_features(x, score_maps)#return local_feature
        feature_vector = torch.cat(feature_vector_list,dim=2)
        feature_vector = self.maxpool(feature_vector)
        feature_vector = torch.squeeze(feature_vector)


        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0 #10.0 3.0
            x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            #x_pool = self.maxpool_2(x)
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        x_pool1 = torch.cat((feature_vector,x_pool),dim=1)
        # x_pool1 = torch.cat((x_pool, x_pool), dim=1)
        # feat = self.bottleneck(x_pool)
        # feature_vector1 = self.bottleneck1(feature_vector)
        # feat = torch.cat((feat,feature_vector1),dim=1)
        feat = self.bottleneck(x_pool1)
        if self.training:
            return x_pool1, feat, self.classifier(feat)
        else:
            return self.l2norm(x_pool1), self.l2norm(feat)
