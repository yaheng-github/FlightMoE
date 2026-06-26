"""
GANomaly 2D-CNN 网络（适配 RflyMAD STFT 图像 [batch, 3, 64, 64]）
直接复用原论文 DCGAN 结构
"""
import torch
import torch.nn as nn


def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


class Encoder(nn.Module):
    """DCGAN Encoder"""
    def __init__(self, isize, nz, nc, ndf, add_final_conv=True):
        super().__init__()
        assert isize % 16 == 0, "isize must be multiple of 16"

        layers = []
        in_ch = nc
        out_ch = ndf
        size = isize

        while size > 4:
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
            out_ch *= 2
            size //= 2

        if add_final_conv:
            layers += [nn.Conv2d(in_ch, nz, 4, 1, 0, bias=False)]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    """DCGAN Decoder"""
    def __init__(self, isize, nz, nc, ngf):
        super().__init__()
        assert isize % 16 == 0, "isize must be multiple of 16"

        cngf = ngf
        tsize = 4
        while tsize < isize:
            cngf *= 2
            tsize *= 2
        cngf //= 2

        layers = [
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(cngf),
            nn.ReLU(True),
        ]
        size = 4
        while size < isize // 2:
            layers += [
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(cngf // 2),
                nn.ReLU(True),
            ]
            cngf //= 2
            size *= 2

        layers += [
            nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class NetG(nn.Module):
    """Generator: Encoder1 -> Decoder -> Encoder2"""
    def __init__(self, opt=None, isize=64, nz=64, nc=3, ndf=64):
        super().__init__()
        if opt is not None:
            isize, nz, nc, ndf = opt.isize, opt.nz, opt.nc, opt.ndf
        self.encoder1 = Encoder(isize, nz, nc, ndf, add_final_conv=True)
        self.decoder = Decoder(isize, nz, nc, ndf)
        self.encoder2 = Encoder(isize, nz, nc, ndf, add_final_conv=True)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen = self.decoder(latent_i)
        latent_o = self.encoder2(gen)
        return gen, latent_i, latent_o


class NetD(nn.Module):
    """Discriminator: features + classifier"""
    def __init__(self, opt=None, isize=64, nz=64, nc=3, ndf=64):
        super().__init__()
        if opt is not None:
            isize, nz, nc, ndf = opt.isize, opt.nz, opt.nc, opt.ndf
        # Discriminator 的 Encoder 输出 1 通道用于二分类
        encoder = Encoder(isize, 1, nc, ndf, add_final_conv=True)
        layers = list(encoder.main.children())
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1], nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1)
        return classifier, features
