"""
Model definition of DeepAttnMISL

If this work is useful for your research, please consider to cite our papers:

[1] "Whole Slide Images based Cancer Survival Prediction using Attention Guided Deep Multiple Instance Learning Networks"
Jiawen Yao, XinliangZhu, Jitendra Jonnagaddala, NicholasHawkins, Junzhou Huang,
Medical Image Analysis, Available online 19 July 2020, 101789

[2] "Deep Multi-instance Learning for Survival Prediction from Whole Slide Images", In MICCAI 2019

"""

import torch.nn as nn
import torch

class DeepAttnMIL_Surv(nn.Module):
    """
    Deep AttnMISL Model definition
    """

    def __init__(self, cluster_num):
        super(DeepAttnMIL_Surv, self).__init__()
        self.embedding_net = nn.Sequential(nn.Conv2d(2048, 64, 1),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1,1))
                                     )


        self.attention = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )

        self.fc6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.cluster_num = cluster_num


    def masked_softmax(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / (mask+1e-5))
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)


    def forward(self, x, mask):

        " x is a tensor list"
        res = []
        for i in range(self.cluster_num):
            hh = x[i].type(torch.FloatTensor).to("cuda")
            output = self.embedding_net(hh)
            output = output.view(output.size()[0], -1)
            res.append(output)


        h = torch.cat(res)

        b = h.size(0)
        c = h.size(1)

        h = h.view(b, c)

        A = self.attention(h)
        A = torch.transpose(A, 1, 0)  # KxN

        A = self.masked_softmax(A, mask)


        M = torch.mm(A, h)  # KxL

        Y_pred = self.fc6(M)

        return Y_pred

