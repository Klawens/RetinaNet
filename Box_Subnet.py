import torch.nn as nn

'''
Box Regression Subnet is a small FCN attached to each FPN level.
Purpose: Regressing the offset from each anchor box to a nearby ground-truth object.
'''


class Box_Subnet(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(Box_Subnet, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.output(out)

        # out is B x C x W x H, C = num_anchors * 4
        out = out.permute(0, 2, 3, 1)
        return out
