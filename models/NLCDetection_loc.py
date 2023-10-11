# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seg_hrnet_config import get_cfg_defaults
import time

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv  = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        ## GX: masking the input outside function.
        output = self.input_conv(input)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)        

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0

        ## in output_mask, fills the 0-value-position with 1.0
        ## without this step, math error occurs.
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        
        return output, new_mask

class NonLocalMask(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(NonLocalMask, self).__init__()

        self.r = reduce_scale

        # input channel number
        self.ic = in_channels * self.r * self.r

        # middle channel number
        self.mc = self.ic

        self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)
        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=18,
                                kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=18, out_channels=1, 
                                kernel_size=3, stride=1, padding=1)

        ## Pconv
        self.Pconv_1 = PartialConv(3, 3, kernel_size=3, stride=2)
        self.Pconv_2 = PartialConv(3, 3, kernel_size=3, stride=2)
        self.Pconv_3 = PartialConv(3, 1, kernel_size=3, stride=2)

    def forward(self, x, img):
        b, c, h, w = x.shape

        x1 = x.reshape(b, self.ic, h // self.r, w // self.r)

        # g x
        g_x = self.g(x1).view(b, self.ic, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta
        theta_x = self.theta(x1).view(b, self.mc, -1)
        theta_x_s = theta_x.permute(0, 2, 1)

        # phi x
        phi_x = self.phi(x1).view(b, self.mc, -1)
        phi_x_s = phi_x

        # non-local attention
        f_s = torch.matmul(theta_x_s, phi_x_s)
        f_s_div = F.softmax(f_s, dim=-1)

        # get y_s
        y_s = torch.matmul(f_s_div, g_x)
        y_s = y_s.permute(0, 2, 1).contiguous()
        y_s = y_s.view(b, c, h, w)

        # GX: (256,256,18), output mask for the deep metric loss.
        mask_feat = x + self.gamma_s * self.W_s(y_s)

        # get 1-dimensional mask_tmp
        # mask_binary = self.getmask(mask_feat)
        mask_feat = self.conv_1(mask_feat)
        mask_binary = mask_feat
        mask_binary = self.relu(mask_binary)
        # print("mask_feat: ", mask_feat.size())  # torch.Size([4, 18, 256, 256])
        mask_binary = self.conv_2(mask_binary)
        # print("mask_binary: ", mask_binary.size())  # torch.Size([4, 1, 256, 256])
        mask_binary = torch.sigmoid(mask_binary)
        mask_tmp = mask_binary.repeat(1, 3, 1, 1)
        mask_img = img * mask_tmp # mask_img is the overlaid image.

        ## conv output
        x, new_mask = self.Pconv_1(mask_img, mask_tmp)
        x, new_mask = self.Pconv_2(x, new_mask)
        x, _        = self.Pconv_3(x, new_mask)
        mask_binary = mask_binary.squeeze(dim=1)
        return x, torch.sigmoid(mask_feat), mask_binary

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):       
        return x.view(x.size(0), -1)

class Classifer(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(Classifer, self).__init__()
        self.pool = nn.Sequential(
                                  # nn.AdaptiveAvgPool2d((1,1)),
                                  nn.AdaptiveAvgPool2d(1),
                                  Flatten()
                                )
        self.fc = nn.Linear(in_channels, output_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.pool(x)
        feat = self.relu(feat)
        cls_res = self.fc(feat)
        return cls_res

class BranchCLS(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(BranchCLS, self).__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                  Flatten()
                                )
        self.fc = nn.Linear(18, output_channels, bias=True)
        self.bn = nn.BatchNorm1d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_cls = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, 
                                                  padding=1, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=32, out_channels=18,
                                                  padding=1, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True), 
                                        )
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        feat = self.branch_cls(x)
        x = self.pool(feat)
        x = self.bn(x)
        cls_res = self.fc(x)
        cls_pro = self.leakyrelu(cls_res)
        zero_vec = -9e15*torch.ones_like(cls_pro)
        cls_pro  = torch.where(cls_pro > 0, cls_pro, zero_vec)
        return cls_res, cls_pro, feat

class FPN_loc(nn.Module):
    '''self-implementation Feature Pyramid Networks '''
    def __init__(self, args, clip_dim=64, multi_feat=None):
        super(FPN_loc, self).__init__()
        ## obtain the dimensions. 
        feat1_num, feat2_num, feat3_num, feat4_num = multi_feat

        self.smooth_s4 = nn.Sequential(
                                    nn.Conv2d(feat4_num, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                    )
        self.smooth_s3 = nn.Sequential(
                                    nn.Conv2d(feat3_num, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                    )
        self.smooth_s2 = nn.Sequential(
                                    nn.Conv2d(feat2_num, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                    )
        self.smooth_s1 = nn.Sequential(
                                    nn.Conv2d(feat1_num, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                    )

        ## new branch.
        self.fpn1 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            # nn.Upsample(scale_factor=2)
        )

        self.fpn2 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.fpn3 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.fpn4 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

        smooth_ops = [self.smooth_s4, self.smooth_s3, self.smooth_s2, self.smooth_s1]
        fpn_ops = [self.fpn4, self.fpn3, self.fpn2, self.fpn1]

class NLCDetection(nn.Module):
    def __init__(self):
        super(NLCDetection, self).__init__()
        self.crop_size = (256, 256)
        self.split_tensor_1 = torch.tensor([1, 3]).cuda()
        self.split_tensor_2 = torch.tensor([1, 2, 1, 3]).cuda()
        self.softmax_m = nn.Softmax(dim=1)
        FENet_cfg = get_cfg_defaults()
        feat1_num, feat2_num, feat3_num, feat4_num = FENet_cfg['STAGE4']['NUM_CHANNELS']

        ## mask generation branch.
        feat_dim = 64 # large clip_dim will ruin the space of Multi-branch-feature-extractor
        self.getmask = NonLocalMask(feat_dim, 4)
        self.FPN_LOC = FPN_loc(feat_dim, multi_feat=FENet_cfg['STAGE4']['NUM_CHANNELS'])

        ## classification branch.
        self.branch_cls_level_1 = BranchCLS(317, 14)   # 252 + 64 = 316
        self.branch_cls_level_2 = BranchCLS(252, 7)    # 144+72+36 = 252
        self.branch_cls_level_3 = BranchCLS(216, 5)    # 144+72 = 216
        self.branch_cls_level_4 = BranchCLS(144, 3)    # 144

    def feature_resize(self, feat):
        '''first obtain the mask via the progressive scheme.'''
        s1, s2, s3, s4 = feat
        s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)
        return s1, s2, s3, s4

    def forward(self, feat, img):

        s1, s2, s3, s4 = self.feature_resize(feat)
        img = F.interpolate(img, size=self.crop_size, 
                            mode='bilinear', align_corners=True)

        feat_4 = self.FPN_LOC.smooth_s4(s4)
        feat_4 = self.FPN_LOC.fpn4(feat_4)   
        feat_3 = self.FPN_LOC.smooth_s3(s3)
        feat_3 = self.FPN_LOC.fpn3(feat_3+feat_4)   
        feat_2 = self.FPN_LOC.smooth_s2(s2)
        feat_2 = self.FPN_LOC.fpn2(feat_2+feat_3)   
        feat_1 = self.FPN_LOC.smooth_s1(s1)
        s1 = self.FPN_LOC.fpn1(feat_1+feat_2)   
        pconv_feat, mask, mask_binary = self.getmask(s1, img)
        pconv_feat = pconv_feat.clone().detach()

        pconv_1 = F.interpolate(pconv_feat, size=s1.size()[2:], mode='bilinear', align_corners=True)

        ## forth branch.
        cls_4, pro_4, _ = self.branch_cls_level_4(s4)
        cls_prob_4      = self.softmax_m(pro_4)
        cls_prob_40 = torch.unsqueeze(cls_prob_4[:,0],1)
        cls_prob_41 = torch.unsqueeze(cls_prob_4[:,1],1)
        cls_prob_42 = torch.unsqueeze(cls_prob_4[:,2],1)
        cls_prob_mask_3 = torch.cat([cls_prob_40, cls_prob_41, cls_prob_41, cls_prob_42, cls_prob_42],axis=1)

        ## third branch
        s4F = F.interpolate(s4, size=s3.size()[2:], mode='bilinear', align_corners=True)
        s3_input = torch.cat([s4F, s3], axis=1)
        cls_3, pro_3, _ = self.branch_cls_level_3(s3_input)
        cls_prob_3      = self.softmax_m(pro_3)
        cls_3 = cls_3 + cls_3 * cls_prob_mask_3
        cls_prob_30 = torch.unsqueeze(cls_prob_3[:,0],1)
        cls_prob_31 = torch.unsqueeze(cls_prob_3[:,1],1)
        cls_prob_32 = torch.unsqueeze(cls_prob_3[:,2],1)
        cls_prob_33 = torch.unsqueeze(cls_prob_3[:,3],1)
        cls_prob_34 = torch.unsqueeze(cls_prob_3[:,4],1)
        cls_prob_mask_2 = torch.cat([cls_prob_30, cls_prob_31, cls_prob_31, 
                                     cls_prob_32, cls_prob_32,
                                     cls_prob_33, cls_prob_34],axis=1)

        ## second branch
        s3F = F.interpolate(s3_input, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s2_input = torch.cat([s3F, s2], axis=1)
        cls_2, pro_2, _ = self.branch_cls_level_2(s2_input) 
        cls_prob_2      = self.softmax_m(pro_2)
        cls_2 = cls_2 + cls_2 * cls_prob_mask_2
        cls_prob_20 = torch.unsqueeze(cls_prob_2[:,0],1)
        cls_prob_21 = torch.unsqueeze(cls_prob_2[:,1],1)
        cls_prob_22 = torch.unsqueeze(cls_prob_2[:,2],1)
        cls_prob_23 = torch.unsqueeze(cls_prob_2[:,3],1)
        cls_prob_24 = torch.unsqueeze(cls_prob_2[:,4],1)
        cls_prob_25 = torch.unsqueeze(cls_prob_2[:,4],1)
        cls_prob_26 = torch.unsqueeze(cls_prob_2[:,4],1)
        cls_prob_mask_1 = torch.cat([cls_prob_20, 
                                     cls_prob_21, cls_prob_21, cls_prob_22, cls_prob_22,    # 4 diffusion
                                     cls_prob_23, cls_prob_23, cls_prob_24, cls_prob_24,    # 4 gan
                                     cls_prob_25, cls_prob_25,                              # faceshifter+stgan
                                     cls_prob_26, cls_prob_26, cls_prob_26], axis=1)        # 3 editing

        s2F = F.interpolate(s2_input, size=s1.size()[2:], mode='bilinear', align_corners=True)
        s1_input = torch.cat([s2F, s1, pconv_1], axis=1)
        cls_1, pro_1, _ = self.branch_cls_level_1(s1_input) 
        cls_1 = cls_1 + cls_1 * cls_prob_mask_1

        mask = mask.squeeze(dim=1)
        return mask, mask_binary, cls_4, cls_3, cls_2, cls_1
