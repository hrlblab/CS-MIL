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

class Classifier(nn.Module):
    """
    Deep AttnMISL Model definition
    """

    def __init__(self):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(5, 2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):

        out = self.classifier(x)

        return out

