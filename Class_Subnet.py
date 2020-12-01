import torch.nn as nn

'''
Classification Subnet is a small FCN attached to each FPN level.
Taking an input feature map with C channels from a given pyramid level, the subnet applies four 3x3 conv layers,
each with C filters and each followed by ReLU activations, followed by a 3x3 conv layer with KA filters.
Finally sigmoid activations are attached to output the KA binary predictions per spatial location.
'''


class Class_Subnet(nn.Module):
    # COCO dataset has 80 classes
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(Class_Subnet, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_classes * num_anchors, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

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
        out = self.sigmoid(out)

        # B x C x W x H, C = num_classes x num_anchors
        out = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out.shape

        out = out.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out.contiguous().view(x.shape[0], -1, self.num_classes)
