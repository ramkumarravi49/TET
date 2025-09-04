import random
from models.layers import *



class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2,64,3,1,1),
            Layer(64,128,3,1,1),
            pool,
            Layer(128,256,3,1,1),
            Layer(256,256,3,1,1),
            pool,
            Layer(256,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
            Layer(512,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

class VGGSNNwoAP(nn.Module):
    def __init__(self):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(2,64,3,1,1),
            Layer(64,128,3,2,1),
            Layer(128,256,3,1,1),
            Layer(256,256,3,2,1),
            Layer(256,512,3,1,1),
            Layer(512,512,3,2,1),
            Layer(512,512,3,1,1),
            Layer(512,512,3,2,1),
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            conv_layer = nn.Conv2d(in_channels if i == 0 else out_channels,
                                     out_channels, kernel_size=3, padding=1, bias=False)
            bn_layer = tdBatchNorm(out_channels)
            layers.append(tdLayer(conv_layer, bn_layer))
            layers.append(LIFSpike())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class vgg11(nn.Module):
    def __init__(self, num_classes=10):
        super(vgg11, self).__init__()
        self.T = 1
        
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = tdBatchNorm(128)
        self.spike = LIFSpike()
        self.features = nn.Sequential(
            tdLayer(self.conv1, self.bn1),
            LIFSpike(),

            VGGBlock(128, 128, 1),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

            VGGBlock(128, 256, 3),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

            VGGBlock(256, 512, 3),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

        )


        self.classifier = nn.Sequential(
            tdLayer(nn.Linear(512 * 4 * 4, 2048)),
            LIFSpike(),
            tdLayer(nn.Linear(2048, 2048)),
            LIFSpike(),
            tdLayer(nn.Linear(2048, num_classes))
        )

    def _forward_impl(self, x):
        x = self.features(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = add_dimention(x, self.T)  # Add time dimension: [T, B, C, H, W]
        return self._forward_impl(x)

if __name__ == '__main__':
    model = VGGSNNwoAP()
    