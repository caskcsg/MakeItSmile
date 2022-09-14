import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 下采样
        self.DownSample = nn.Sequential(nn.Conv2d(3, 32, 7, padding=3),
                                        nn.BatchNorm2d(32),
                                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.BatchNorm2d(64),
                                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                        nn.LeakyReLU(negative_slope=0.1),)
        # 中间过程
        self.Bottleneck = nn.Sequential(nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, 3, padding=1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        # nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, 3, padding=1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, 3, padding=1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        # nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, 3, padding=1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, 3, padding=1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        # nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, 3, padding=1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        )
        # 上采样
        self.UpSample = nn.Sequential(nn.BatchNorm2d(256),
                                      nn.Conv2d(256, 128, 3, padding=1),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Upsample(scale_factor=2),
                                      nn.BatchNorm2d(128),
                                      nn.Conv2d(128, 64, 3, padding=1),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Upsample(scale_factor=2),
                                      nn.BatchNorm2d(64),
                                      nn.Conv2d(64, 3, 3, padding=1),
                                      nn.Tanh())
        self.FeatureExtractor = FeatureExtractor()

    def forward(self, source, driving):
        logits, segmentation_map, driving_feature = self.FeatureExtractor(driving)
        source_feature = self.DownSample(source)
        feature = torch.cat([source_feature, driving_feature], dim=1)
        feature = self.Bottleneck(feature)
        generated = self.UpSample(feature)
        return logits, segmentation_map, generated


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 分类任务
        self.Classification = nn.Sequential(nn.Conv2d(3, 64, 7, padding=3),
                                            nn.AvgPool2d(2, stride=2),
                                            nn.BatchNorm2d(64),
                                            nn.Conv2d(64, 128, 3, padding=1),
                                            nn.LeakyReLU(negative_slope=0.1),
                                            nn.AvgPool2d(2, stride=2),
                                            )
        self.down = nn.Sequential(nn.BatchNorm2d(128),
                                  nn.Conv2d(128, 1, 3, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1),)
        self.Linear = nn.Linear(64*64, 75)
        # 分割任务
        self.Segmentation = nn.Sequential(nn.Conv2d(3, 64, 7, padding=3),
                                          nn.AvgPool2d(2, stride=2),
                                          nn.BatchNorm2d(64),
                                          nn.Conv2d(64, 128, 3, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1),
                                          nn.BatchNorm2d(128),
                                          nn.Conv2d(128, 128, 3, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1),
                                          nn.AvgPool2d(2, stride=2),
                                          nn.BatchNorm2d(128),
                                          nn.Conv2d(128, 256, 3, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1),
                                          nn.BatchNorm2d(256),
                                          nn.Conv2d(256, 256, 3, padding=1),
                                          nn.LeakyReLU(negative_slope=0.1),
                                          nn.BatchNorm2d(256),
                                          nn.Conv2d(256, 1, 3, padding=1),
                                          )

    def forward(self, x):
        B = x.shape[0]
        F = self.Classification(x)
        feature = self.down(F)
        logit = nn.Softmax()(self.Linear(feature.view(B, -1)))
        M = self.Segmentation(x)
        F_M = F * M
        return logit, M, F_M

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, x):
        return self.net(x)

# if __name__ == '__main__':
#     # 测试
#     source = torch.randn(5, 3, 256, 256)
#     driving = torch.randn(5, 3, 256, 256)
#     model = Generator()
#     dis = Discriminator()
#     output = model(source, driving)
#     pred_fake = dis(output)
#     # x = torch.randn(5, 3, 256, 256)
#     # model = FeatureExtractor()
#     # output = model(x)
#     pass
