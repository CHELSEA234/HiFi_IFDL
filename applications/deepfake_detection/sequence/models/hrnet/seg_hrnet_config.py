# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The script is adopted from Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# high_resoluton_net related params for segmentation
HRNET = CN()
HRNET.PRETRAINED_LAYERS = ['*']
HRNET.STEM_INPLANES = 64
HRNET.FINAL_CONV_KERNEL = 1
HRNET.PRETRAINED = '/research/cvl-guoxia11/semi_supervise_local/IJCV_deepfake/deepfake_HiFiNet++_v4_pretrained/hrnet_w18_small_model_v2.pth'

HRNET.STAGE1 = CN()
HRNET.STAGE1.NUM_MODULES = 1
HRNET.STAGE1.NUM_BRANCHES = 1
HRNET.STAGE1.NUM_BLOCKS = [2]
HRNET.STAGE1.NUM_CHANNELS = [64]
HRNET.STAGE1.BLOCK = 'BOTTLENECK'
HRNET.STAGE1.FUSE_METHOD = 'SUM'

HRNET.STAGE2 = CN()
HRNET.STAGE2.NUM_MODULES = 1
HRNET.STAGE2.NUM_BRANCHES = 4
HRNET.STAGE2.NUM_BLOCKS = [2, 2, 2, 2]
HRNET.STAGE2.NUM_CHANNELS = [18, 36, 72, 144]
HRNET.STAGE2.BLOCK = 'BASIC'
HRNET.STAGE2.FUSE_METHOD = 'SUM'

HRNET.STAGE3 = CN()
HRNET.STAGE3.NUM_MODULES = 1
HRNET.STAGE3.NUM_BRANCHES = 4
HRNET.STAGE3.NUM_BLOCKS = [2, 2, 2, 2]
HRNET.STAGE3.NUM_CHANNELS = [18, 36, 72, 144]
HRNET.STAGE3.BLOCK = 'BASIC'
HRNET.STAGE3.FUSE_METHOD = 'SUM'

HRNET.STAGE4 = CN()
HRNET.STAGE4.NUM_MODULES = 1
HRNET.STAGE4.NUM_BRANCHES = 4
HRNET.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
HRNET.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
HRNET.STAGE4.BLOCK = 'BASIC'
HRNET.STAGE4.FUSE_METHOD = 'SUM'


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return HRNET.clone()

# def one_hot_label_new(vector, Softmax_m=nn.Softmax(dim=1)):
#     '''
#         compute the probability for being as the synthesized image (TODO: double check).
#     '''
#     x = Softmax_m(vector)
#     indices = torch.argmax(x, dim=1)
#     prob = 1 - x[:,0]
#     indices = list(indices.cpu().numpy())
#     prob = list(prob.cpu().numpy())
#     return indices, prob

# def level_1_convert(input_lst):
#     res_lst = []
#     for _ in input_lst:
#         if _ == 0:
#             res_lst.append(0)
#         else:
#             res_lst.append(1)
#     return res_lst

if __name__ == "__main__":
  print("Hello World!")