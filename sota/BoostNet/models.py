import torch
import torch.nn as nn
from torch.autograd import Variable


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=True, act=nn.ReLU(True), res_scale=None):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class ResBlock_(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=True, act=nn.ReLU(True),
            res_scale=None, drop=0.0):

        super(ResBlock_, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
                m.append(nn.Dropout(p=drop))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class EstRGBRealNoise(nn.Module):

    def __init__(self, test_mode=False, \
                 num_input_channels=3, num_layer_sigma=12, output_features=96, \
                 num_mid_blocks=12, num_feature_maps=96):
        super(EstRGBRealNoise, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_mid_blocks = num_mid_blocks
        self.num_layer_sigma = num_layer_sigma
        self.output_features = output_features

        kernel_size = 3
        act = nn.ReLU(inplace=True)

        # define down module (after concat)
        self.down = nn.Sequential( \
            nn.Conv2d(self.num_input_channels, self.output_features, \
                      kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(self.output_features),
            act,
        )
        # define head module
        self.head = nn.Sequential( \
            nn.Conv2d((self.output_features + self.num_layer_sigma), \
                      self.num_feature_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_feature_maps),
            act,
        )
        # define body module
        m_body = [
            ResBlock(
                conv, self.num_feature_maps, kernel_size, act=act, res_scale=1, bn=True
            ) for _ in range(self.num_mid_blocks)
        ]
        self.body = nn.Sequential(*m_body)
        # define pre_up
        self.pre_up = nn.Sequential( \
            nn.Conv2d(self.num_feature_maps, self.output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_features),
            act,
        )
        # define tail module
        self.up = nn.Sequential( \
            nn.ConvTranspose2d(self.output_features, self.num_input_channels, kernel_size=2, \
                               stride=2, padding=0, output_padding=0),
        )

    def forward(self, x, noise_sigma):
        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = x.shape
        if sh_im[2] % 2 == 1 or sh_im[3] % 2 == 1:
            x_ = x
            if sh_im[2] % 2 == 1:
                expanded_h = True
                x_ = torch.cat((x_, x_[:, :, -2:-1, :]), dim=2)
            if sh_im[3] % 2 == 1:
                expanded_w = True
                x_ = torch.cat((x_, x_[:, :, :, -2:-1]), dim=3)
            down = self.down(x_)
        else:
            down = self.down(x)
        # concat noise level map
        N, C, H, W = down.size()
        sigma_map = noise_sigma.view(N, 1, 1, 1).repeat(1, self.num_layer_sigma, H, W)
        if self.test_mode:
            sigma_map = Variable(sigma_map, volatile=True)
        else:
            sigma_map = Variable(sigma_map)

        concat_noise_x = torch.cat((down, sigma_map), dim=1)
        # head
        head = self.head(concat_noise_x)
        # body
        body = self.body(head)
        body = body + head
        # pre_up
        pre_up = self.pre_up(body)
        pre_up = pre_up + down
        # up
        up = self.up(pre_up)

        if expanded_h:
            up = up[:, :, :-1, :]
        if expanded_w:
            up = up[:, :, :, :-1]

        up = up + x

        return up

class GENetGRAY(nn.Module):
    def __init__(self, test_mode=False, \
                 num_input_channels=1, num_layer_sigma=4, output_features=16, \
                 num_mid_blocks=22, num_feature_maps=64):
        super(GENetGRAY, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_mid_blocks = num_mid_blocks
        self.num_layer_sigma = num_layer_sigma
        self.output_features = output_features

        kernel_size = 3
        act = nn.ReLU(inplace=True)
        # define down module (after concat)
        self.down_sigma = nn.Sequential( \
            nn.Conv2d(self.num_input_channels, self.num_layer_sigma, \
                      kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(self.output_features),
            act,
        )
        # define down module (after concat)
        self.down = nn.Sequential( \
            nn.Conv2d(self.num_input_channels, self.output_features, \
                      kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(self.output_features),
            act,
        )
        # define head module
        self.head = nn.Sequential( \
            nn.Conv2d((self.output_features + self.num_layer_sigma), \
                      self.num_feature_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_feature_maps),
            act,
        )
        # define body module
        m_body = [
            ResBlock(
                conv, self.num_feature_maps, kernel_size, act=act, res_scale=1, bn=True,
            ) for _ in range(self.num_mid_blocks)
        ]
        self.body = nn.Sequential(*m_body)
        # define pre_up
        self.pre_up = nn.Sequential( \
            nn.Conv2d(self.num_feature_maps, self.output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_features),
            act,
        )
        # define tail module
        self.up = nn.Sequential( \
            nn.ConvTranspose2d(self.output_features, self.num_input_channels, kernel_size=2, \
                               stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(self.num_input_channels),
            # act,
        )

    def forward(self, x, noise_sigma):
        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = x.shape
        if sh_im[2] % 2 == 1 or sh_im[3] % 2 == 1:
            x_ = x
            if sh_im[2] % 2 == 1:
                expanded_h = True
                x_ = torch.cat((x_, x_[:, :, -2:-1, :]), dim=2)
            if sh_im[3] % 2 == 1:
                expanded_w = True
                x_ = torch.cat((x_, x_[:, :, :, -2:-1]), dim=3)
            N, C, H, W = x_.shape
            down = self.down(x_)
        else:
            N, C, H, W = x.shape
            down = self.down(x)
        # Noivelevel map
        sigma_map = noise_sigma.view(N, 1, 1, 1).repeat(1, C, H, W)
        if self.test_mode:
            sigma_map = Variable(sigma_map, volatile=True)
        else:
            sigma_map = Variable(sigma_map)
        down_sigma_map = self.down_sigma(sigma_map)

        concat_noise_x = torch.cat((down, down_sigma_map), dim=1)
        # head
        head = self.head(concat_noise_x)
        # body
        body = self.body(head)
        body = body + head
        # pre_up
        pre_up = self.pre_up(body)
        pre_up = pre_up + down
        # up
        up = self.up(pre_up)

        if expanded_h:
            up = up[:, :, :-1, :]
        if expanded_w:
            up = up[:, :, :, :-1]

        up = up + x

        return up

class GNet(nn.Module):

    def __init__(self, test_mode=False, \
                 num_input_channels=3, num_layer_sigma=12, output_features=128, \
                 num_mid_blocks=18, num_feature_maps=128):
        super(GNet, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_mid_blocks = num_mid_blocks
        self.num_layer_sigma = num_layer_sigma
        self.output_features = output_features

        kernel_size = 3
        act = nn.ReLU(inplace=True)

        # define down module (before concat)
        self.down_sigma = nn.Conv2d(self.num_input_channels, self.num_layer_sigma, \
                                    kernel_size=2, stride=2, padding=0)
        self.down = nn.Sequential( \
            nn.Conv2d(self.num_input_channels * 2, self.output_features, \
                      kernel_size=2, stride=2, padding=0),
            act,
        )
        # define head module
        self.head = nn.Sequential( \
            nn.Conv2d((self.output_features + self.num_layer_sigma), \
                      self.num_feature_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_feature_maps),
            act,
        )
        # define body module
        m_body = [
            ResBlock(
                conv, self.num_feature_maps, kernel_size, act=act, res_scale=1, bn=True,
            ) for _ in range(self.num_mid_blocks)
        ]
        self.body = nn.Sequential(*m_body)
        # define pre_up
        self.pre_up = nn.Sequential( \
            nn.Conv2d(self.num_feature_maps, self.output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_features),
            act,
        )
        # define tail module
        self.up = nn.Sequential( \
            nn.ConvTranspose2d(self.output_features, self.num_input_channels, kernel_size=2, \
                               stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(self.num_input_channels),
            # act,
        )

    def forward(self, x, est_noi):
        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = x.shape
        si, noi = list(est_noi)
        if sh_im[2] % 2 == 1 or sh_im[3] % 2 == 1:
            x_ = x
            if sh_im[2] % 2 == 1:
                expanded_h = True
                x_ = torch.cat((x_, x_[:, :, -2:-1, :]), dim=2)

                noi = torch.cat((noi, noi[:, :, -2:-1, :]), dim=2)
                si = torch.cat((si, si[:, :, -2:-1, :]), dim=2)
            if sh_im[3] % 2 == 1:
                expanded_w = True
                x_ = torch.cat((x_, x_[:, :, :, -2:-1]), dim=3)
                noi = torch.cat((noi, noi[:, :, :, -2:-1]), dim=3)
                si = torch.cat((si, si[:, :, :, -2:-1]), dim=3)
            down = self.down(torch.cat((x_, noi), dim=1))
        else:
            down = self.down(torch.cat((x, noi), dim=1))

        sigma_map = self.down_sigma(si)

        concat_noise_x = torch.cat((down, sigma_map), dim=1)
        # head
        head = self.head(concat_noise_x)
        # body
        body = self.body(head)
        body = body + head
        # pre_up
        pre_up = self.pre_up(body)
        pre_up = pre_up + down
        # up
        up = self.up(pre_up)

        if expanded_h:
            up = up[:, :, :-1, :]
        if expanded_w:
            up = up[:, :, :, :-1]

        up = up + x

        return up

class EstNetRGB(nn.Module):

    def __init__(self, test_mode=False, \
                 num_input_channels=3, num_layer_sigma=128, output_features=128, \
                 num_mid_blocks=10, num_feature_maps=128):
        super(EstNetRGB, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_mid_blocks = num_mid_blocks
        self.num_layer_sigma = num_layer_sigma
        self.output_features = output_features

        kernel_size = 3
        act = nn.ReLU(inplace=True)

        # define down module (after concat)
        self.down = nn.Sequential( \
            nn.Conv2d(self.num_input_channels, self.num_feature_maps, \
                      kernel_size=2, stride=2, padding=0),
            act,
        )
        # define body module
        m_body = [
            ResBlock(
                conv, self.num_feature_maps, kernel_size, act=act, res_scale=1, bn=True,
            ) for _ in range(self.num_mid_blocks)
        ]
        m_body.append(conv(self.num_feature_maps, self.num_feature_maps, kernel_size))
        self.body = nn.Sequential(*m_body)
        # define noise map module
        self.noisemap = nn.Sequential( \
            nn.ConvTranspose2d(self.num_layer_sigma, self.num_layer_sigma, kernel_size=2, \
                               stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(self.num_layer_sigma),
            act,
            nn.Conv2d(self.num_layer_sigma, self.num_input_channels, kernel_size=3, stride=1, padding=1),
        )
        # define denoise module
        self.denoise = nn.Sequential( \
            nn.ConvTranspose2d(self.num_feature_maps, self.output_features, kernel_size=2, \
                               stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(self.output_features),
            act,
            nn.Conv2d(self.output_features, self.num_input_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # Handle odd sizes and down
        expanded_h = False
        expanded_w = False
        sh_im = x.shape
        if sh_im[2] % 2 == 1 or sh_im[3] % 2 == 1:
            x_ = x
            if sh_im[2] % 2 == 1:
                expanded_h = True
                x_ = torch.cat((x_, x_[:, :, -2:-1, :]), dim=2)
            if sh_im[3] % 2 == 1:
                expanded_w = True
                x_ = torch.cat((x_, x_[:, :, :, -2:-1]), dim=3)
            down = self.down(x_)
        else:
            down = self.down(x)
        # body
        body = self.body(down)
        body = body + down
        # noisemap
        est_noise_ = self.noisemap(body[:, 0:self.num_layer_sigma, :, :])
        # denoise
        est_noise = self.denoise(body)

        if expanded_h:
            est_noise_ = est_noise_[:, :, :-1, :]
            est_noise = est_noise[:, :, :-1, :]
        if expanded_w:
            est_noise_ = est_noise_[:, :, :, :-1]
            est_noise = est_noise[:, :, :, :-1]


        est_noise_ = est_noise_ + x
        est_noise = est_noise + x

        return est_noise_, est_noise

class BoostNet(nn.Module):

    def __init__(self, test_mode=False, num_input_channels=6, num_mid_blocks=6, num_feature_maps=96):
        super(BoostNet, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_mid_blocks = num_mid_blocks

        kernel_size = 3
        act = nn.ReLU(inplace=True)
        # define head module
        self.head = nn.Sequential( \
            nn.Conv2d(self.num_input_channels, self.num_feature_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_feature_maps),
            act,
        )
        # define body module
        m_body = [
            ResBlock_(
                conv, self.num_feature_maps, kernel_size, act=act, res_scale=1, bn=True,
            ) for _ in range(self.num_mid_blocks)
        ]
        self.body = nn.Sequential(*m_body)
        # define tail module
        self.out = nn.Sequential( \
            nn.Conv2d(self.num_feature_maps, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x_noGan, x_gan):

        concat_noise_x = torch.cat((x_noGan, x_gan), dim=1)
        # head
        head = self.head(concat_noise_x)
        # body
        body = self.body(head)
        body = body + head
        # pre_up
        out = self.out(body)

        out = out + 0.0 * x_noGan + 0.0 * x_gan

        return out

# FMD
class EstNetFMD(nn.Module):

    def __init__(self, test_mode=False, \
                 num_input_channels=1, num_layer_sigma=12, output_features=96, \
                 num_mid_blocks=12, num_feature_maps=96):
        super(EstNetFMD, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_mid_blocks = num_mid_blocks
        self.num_layer_sigma = num_layer_sigma
        self.output_features = output_features

        kernel_size = 3
        act = nn.ReLU(inplace=True)

        # define down module (after concat)
        self.down = nn.Sequential( \
            nn.Conv2d(self.num_input_channels, self.output_features, \
                      kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(self.output_features),
            act,
        )
        # define head module
        self.head = nn.Sequential( \
            nn.Conv2d((self.output_features + self.num_layer_sigma), \
                      self.num_feature_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_feature_maps),
            act,
        )
        # define body module
        m_body = [
            ResBlock(
                conv, self.num_feature_maps, kernel_size, act=act, res_scale=1, bn=True
            ) for _ in range(self.num_mid_blocks)
        ]
        self.body = nn.Sequential(*m_body)
        # define pre_up
        self.pre_up = nn.Sequential( \
            nn.Conv2d(self.num_feature_maps, self.output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_features),
            act,
        )
        # define tail module
        self.up = nn.Sequential( \
            nn.ConvTranspose2d(self.output_features, self.num_input_channels, kernel_size=2, \
                               stride=2, padding=0, output_padding=0),
        )

    def forward(self, x, noise_sigma):
        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = x.shape
        if sh_im[2] % 2 == 1 or sh_im[3] % 2 == 1:
            x_ = x
            if sh_im[2] % 2 == 1:
                expanded_h = True
                x_ = torch.cat((x_, x_[:, :, -2:-1, :]), dim=2)
            if sh_im[3] % 2 == 1:
                expanded_w = True
                x_ = torch.cat((x_, x_[:, :, :, -2:-1]), dim=3)
            down = self.down(x_)
        else:
            down = self.down(x)
        # concat noise level map
        N, C, H, W = down.size()
        sigma_map = noise_sigma.view(N, 1, 1, 1).repeat(1, self.num_layer_sigma, H, W)
        if self.test_mode:
            sigma_map = Variable(sigma_map, volatile=True)
        else:
            sigma_map = Variable(sigma_map)

        concat_noise_x = torch.cat((down, sigma_map), dim=1)

        head = self.head(concat_noise_x)
        # body
        body = self.body(head)
        body = body + head
        # pre_up
        pre_up = self.pre_up(body)
        pre_up = pre_up + down
        # up
        up = self.up(pre_up)

        if expanded_h:
            up = up[:, :, :-1, :]
        if expanded_w:
            up = up[:, :, :, :-1]

        up = up + x

        return up


class GNetFMD(nn.Module):

    def __init__(self, test_mode=False, \
                 num_input_channels=1, num_layer_sigma=12, output_features=128, \
                 num_mid_blocks=18, num_feature_maps=128):
        super(GNetFMD, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_mid_blocks = num_mid_blocks
        self.num_layer_sigma = num_layer_sigma
        self.output_features = output_features

        kernel_size = 3
        act = nn.ReLU(inplace=True)

        # define down module (before concat)
        self.down_sigma = nn.Conv2d(self.num_input_channels, self.num_layer_sigma, \
                                    kernel_size=2, stride=2, padding=0)
        self.down = nn.Sequential( \
            nn.Conv2d(self.num_input_channels * 2, self.output_features, \
                      kernel_size=2, stride=2, padding=0),
            act,
        )
        # define head module
        self.head = nn.Sequential( \
            nn.Conv2d((self.output_features + self.num_layer_sigma), \
                      self.num_feature_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_feature_maps),
            act,
        )
        # define body module
        m_body = [
            ResBlock(
                conv, self.num_feature_maps, kernel_size, act=act, res_scale=1, bn=True,
            ) for _ in range(self.num_mid_blocks)
        ]
        self.body = nn.Sequential(*m_body)
        # define pre_up
        self.pre_up = nn.Sequential( \
            nn.Conv2d(self.num_feature_maps, self.output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_features),
            act,
        )
        # define tail module
        self.up = nn.Sequential( \
            nn.ConvTranspose2d(self.output_features, self.num_input_channels, kernel_size=2, \
                               stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(self.num_input_channels),
            # act,
        )

    def forward(self, x, est_noi):
        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = x.shape
        si, noi = list(est_noi)
        if sh_im[2] % 2 == 1 or sh_im[3] % 2 == 1:
            x_ = x
            if sh_im[2] % 2 == 1:
                expanded_h = True
                x_ = torch.cat((x_, x_[:, :, -2:-1, :]), dim=2)

                noi = torch.cat((noi, noi[:, :, -2:-1, :]), dim=2)
                si = torch.cat((si, si[:, :, -2:-1, :]), dim=2)
            if sh_im[3] % 2 == 1:
                expanded_w = True
                x_ = torch.cat((x_, x_[:, :, :, -2:-1]), dim=3)
                noi = torch.cat((noi, noi[:, :, :, -2:-1]), dim=3)
                si = torch.cat((si, si[:, :, :, -2:-1]), dim=3)
            down = self.down(torch.cat((x_, noi), dim=1))
        else:
            down = self.down(torch.cat((x, noi), dim=1))

        sigma_map = self.down_sigma(si)

        concat_noise_x = torch.cat((down, sigma_map), dim=1)
        # head
        head = self.head(concat_noise_x)
        # body
        body = self.body(head)
        body = body + head
        # pre_up
        pre_up = self.pre_up(body)
        pre_up = pre_up + down
        # up
        up = self.up(pre_up)

        if expanded_h:
            up = up[:, :, :-1, :]
        if expanded_w:
            up = up[:, :, :, :-1]

        up = up + x

        return up


class BoostNetFMD(nn.Module):

    def __init__(self, test_mode=False, num_input_channels=3, num_mid_blocks=16, num_feature_maps=128):
        super(BoostNetFMD, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_mid_blocks = num_mid_blocks

        kernel_size = 3
        act = nn.ReLU(inplace=True)
        # define head module
        self.head = nn.Sequential( \
            nn.Conv2d(self.num_input_channels, self.num_feature_maps, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.num_feature_maps),
            act,
        )
        # define body module
        m_body = [
            ResBlock_(
                conv, self.num_feature_maps, kernel_size, act=act, res_scale=1, bn=False,
            ) for _ in range(self.num_mid_blocks)
        ]
        self.body = nn.Sequential(*m_body)
        # define tail module
        self.out = nn.Sequential( \
            nn.Conv2d(self.num_feature_maps, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, x_noGan, x_gan):
        concat_noise_x = torch.cat((x, x_noGan, x_gan), dim=1)
        # head
        head = self.head(concat_noise_x)
        # body
        body = self.body(head)
        body = body + head
        # pre_up
        out = self.out(body)

        out = out + x

        return out


