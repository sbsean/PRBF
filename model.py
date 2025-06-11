#filname : model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0.2, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def clip_gradients_by_layers(
    model, encoder_clip_value=1.0, decoder_clip_value=1.0, final_clip_value=0.5
):
    """
    각 레이어 그룹별로 다른 gradient clipping 값을 적용합니다.

    Args:
        model: DCLUNet 모델 인스턴스
        encoder_clip_value: 인코더 레이어(Conv1-Conv5)에 적용할 클리핑 값
        decoder_clip_value: 디코더 레이어(Up5-Up_conv2)에 적용할 클리핑 값
        final_clip_value: 최종 출력 레이어(Conv_1x1)에 적용할 클리핑 값
    """
    # 인코더 레이어 클리핑
    encoder_layers = [model.Conv1, model.Conv2, model.Conv3, model.Conv4, model.Conv5]
    for layer in encoder_layers:
        for param in layer.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, encoder_clip_value)

    # 디코더 레이어 클리핑
    decoder_layers = [
        model.Up5,
        model.Up_conv5,
        model.Up4,
        model.Up_conv4,
        model.Up3,
        model.Up_conv3,
        model.Up2,
        model.Up_conv2,
    ]
    for layer in decoder_layers:
        for param in layer.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, decoder_clip_value)

    # 최종 출력 레이어 클리핑
    for param in model.Conv_1x1.parameters():
        if param.grad is not None:
            torch.nn.utils.clip_grad_norm_(param, final_clip_value)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class DCLUNet(nn.Module):
    def __init__(self, img_ch=2, output_ch=2):
        super(DCLUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        self.Conv5 = conv_block(ch_in=256, ch_out=512)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return F.tanh(d1)