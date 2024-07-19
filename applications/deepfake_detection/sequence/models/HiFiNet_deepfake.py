# coding: utf-8
# author: Hierarchical Fine-Grained Image Forgery Detection and Localization
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./sequence/models')
from hrnet.seg_hrnet_config import get_cfg_defaults
from hrnet.seg_hrnet import get_seg_model

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):		
        return x.view(x.size(0), -1)

class CatDepth(nn.Module):
    def __init__(self):
        super(CatDepth, self).__init__()

    def forward(self, x, y):
        return torch.cat([x,y],dim=1)

class HiFiNet_deepfake(nn.Module):
    def __init__(self, use_laplacian=False, drop_rate=0.5, use_magic_loss=True,
                 feat_dim = 1024, pretrained=True,
                 rnn_type='LSTM', rnn_hidden_size=10, num_rnn_layers=1, rnn_drop_rate=0.5,
                 bidir=False, merge_mode='concat',gate_type='sigmoid', device='cuda'):
        super(HiFiNet_deepfake, self).__init__()
        self.use_laplacian = use_laplacian
        self.feat_dim = feat_dim
        self.rnn_type = rnn_type
        self.rnn_input_size = feat_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.rnn_drop_rate = rnn_drop_rate
        self.bidir = bidir
        self.magic_loss = use_magic_loss
        
        self.device = device
        
        self.FENet = get_seg_model(get_cfg_defaults()).to(self.device)
        self.rnn = nn.LSTM(input_size=self.rnn_input_size, hidden_size=self.rnn_hidden_size,
                            num_layers=self.num_rnn_layers, batch_first=False, dropout=self.rnn_drop_rate,
                            bidirectional=self.bidir
                            )
        self.output_rnn = nn.Sequential(nn.ReLU(inplace=True),
                                        nn.Linear(256, 2))

        # Select the merger function
        if merge_mode == 'concat':
            self.merger_function = merge_concat
        elif merge_mode == 'sum':
            self.merger_function = merge_sum
        
    def forward(self,x):
        batch_size, window_size, _, H, W = x.size()
        x = x.view(batch_size * window_size, 3, H, W) # Input for RGB branch

        conv_feat = self.FENet(x)
        z = conv_feat.view(batch_size, window_size, -1).permute(1,0,2)
        out, (h,c) = self.rnn(z)
        out = self.merger_function(out[-1, :, :self.rnn_hidden_size], out[0, :, self.rnn_hidden_size:]) 
        out = self.output_rnn(out)

        return out

    def up (self,x, size):
        return F.interpolate(x,size=size,mode='bilinear',
                             align_corners=False)
    def up_pix(self,x,r):
        return F.pixel_shuffle(x,r)

## Functions to merger the bidirectional outputs
# Concatenation function
def merge_concat(out1, out2):
    return torch.cat((out1, out2), 1)
# Summation function
def merge_sum(out1, out2):
    return torch.add(out1, out2)

if __name__ == "__main__":
    import torch
    input = torch.randn((4, 1, 3, 224, 224)).cuda()  # [64, 10, 3, 224, 224]
    model = HiFiNet_deepfake(use_laplacian=True, drop_rate=0.2, use_magic_loss=False,
                            pretrained=True, rnn_drop_rate=0.2, feat_dim=1000,
                            rnn_hidden_size=128, num_rnn_layers=2,
                            bidir=True).cuda()
    model = torch.nn.DataParallel(model)

    print(f"...comes to this place...")
    output = model(input)
    print(f"the model output: ", output.size())
    print("...over...")