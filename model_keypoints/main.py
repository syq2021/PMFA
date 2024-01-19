# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
#
# from .config import cfg as pose_config
# from .pose_hrnet import get_pose_net
# from .pose_processor import HeatmapProcessor2
#
#
# if __name__ == '__main__':
#     device = "cuda"
#     img =
#     scoremap_computer = ScoremapComputer(10.0).to(device)
#     with torch.no_grad():
#         score_maps, keypoints_confidence, keypoints_location = scoremap_computer(img)