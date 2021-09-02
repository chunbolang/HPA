import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class Feature_Reweighting(nn.Module):

    # Ref: https://github.com/yiskw713/DualAttention_for_Segmentation/blob/master/model/attention.py

    def __init__(self):
        super(Feature_Reweighting, self).__init__()

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)   # Learnable Param.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sup_fea, que_fea):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, H, W = sup_fea.shape

        query_sup = sup_fea.reshape(N, C, -1)                  # (B, C, H*W)
        query_que = que_fea.reshape(N, C, -1)                  # (B, C, H*W)

        key_que   = que_fea.reshape(N, C, -1).permute(0, 2, 1) # (B, H*W, C)
        value_que = que_fea.reshape(N, C, -1)                  # (B, C, H*W)

        cross_atten = torch.bmm(query_sup, key_que)
        self_atten  = torch.bmm(query_que, key_que)

        self_atten = torch.max(self_atten, -1, keepdim=True)[0].expand_as(self_atten) - self_atten  # prevent loss divergence https://github.com/junfu1115/DANet/issues/9
        cross_atten = torch.max(cross_atten, -1, keepdim=True)[0].expand_as(cross_atten) - cross_atten

        cross_atten_soft = self.softmax(cross_atten)  # (C, C)
        self_atten_soft = self.softmax(self_atten)    # (C, C)

        total_atten = (self_atten_soft + self.alpha*cross_atten_soft)/(1+self.alpha)

        out = torch.bmm(total_atten, value_que)
        out = out.view(N, C, H, W)
        out = out + que_fea

        return out, self.alpha