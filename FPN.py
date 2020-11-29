import torch.nn as nn

'''
    Higher feature up sampling to mix up with lower feature,
    each layer predicts separately.
    This differs slightly from vanilla-FPN but improves speed while maintaining accuracy.
'''


class FeaturePyramid(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(FeaturePyramid, self).__init__()

        # P3 to P5 are computed from the output of the corresponding ResNet residual stage.
        # C5 -> P5
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1)  # stride=1, padding=0 by default
        self.P5_upsampling = nn.Upsample(scale_factor=2)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # C4 -> P5
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1)
        self.P4_upsampling = nn.Upsample(scale_factor=2)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # C3 -> P3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6.
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # P6 is obtained via a 3x3 stride-2 conv on C5.
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_up = self.P5_upsampling(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_up + P4_x
        P4_up = self.P4_upsampling(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P4_up + P3_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
