import torch
import torch.nn as nn
import torch.nn.functional as F
from KANConv import KAN_Convolutional_Layer


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=C_in, out_channels=C_out, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=C_out),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=C_out,
                out_channels=C_out,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=C_out),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class KConv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=C_in, out_channels=C_out, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=C_out),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            KAN_Convolutional_Layer(
            kernel_size=(3, 3), n_convs=1, stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(num_features=C_out),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(
                in_channels=C, out_channels=C, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(
            in_channels=C, out_channels=C // 2, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, r):
        up = F.interpolate(input=x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.Up(up)
        return torch.cat((x, r), 1)


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.C1 = Conv(in_channels, 16)
        self.D1 = DownSampling(16)
        self.C2 = Conv(16, 32)
        self.D2 = DownSampling(32)
        self.C3 = Conv(32, 64)
        self.D3 = DownSampling(64)
        self.C4 = Conv(64, 128)
        self.D4 = DownSampling(128)
        self.C5 = Conv(128, 256)

        self.U1 = UpSampling(256)
        self.C6 = Conv(256, 128)
        self.U2 = UpSampling(128)
        self.C7 = Conv(128, 64)
        self.U3 = UpSampling(64)
        self.C8 = Conv(64, 32)
        self.U4 = UpSampling(32)
        self.C9 = Conv(32, 16)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(
            in_channels=16,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        return self.Th(self.pred(O4))


class UDKCONV(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(UKCONV, self).__init__()
        self.C1 = KAN_Convolutional_Layer(
            kernel_size=(3, 3), n_convs=16, stride=(1, 1), padding=(1, 1)
        )
        self.D1 = DownSampling(16)
        self.C2 = KAN_Convolutional_Layer(
            kernel_size=(3, 3), n_convs=2, stride=(1, 1), padding=(1, 1)
        )
        self.D2 = DownSampling(32)
        self.C3 = KAN_Convolutional_Layer(
            kernel_size=(3, 3), n_convs=2, stride=(1, 1), padding=(1, 1)
        )
        self.D3 = DownSampling(64)
        self.C4 = KAN_Convolutional_Layer(
            kernel_size=(3, 3), n_convs=2, stride=(1, 1), padding=(1, 1)
        )
        self.D4 = DownSampling(128)
        self.C5 = KAN_Convolutional_Layer(
            kernel_size=(3, 3), n_convs=2, stride=(1, 1), padding=(1, 1)
        )

        self.U1 = UpSampling(256)
        self.C6 = Conv(256, 128)
        self.U2 = UpSampling(128)
        self.C7 = Conv(128, 64)
        self.U3 = UpSampling(64)
        self.C8 = Conv(64, 32)
        self.U4 = UpSampling(32)
        self.C9 = Conv(32, 16)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(
            in_channels=16,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        return self.Th(self.pred(O4))


class UKCONV(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(UKCONV, self).__init__()
        self.C1 = KConv(in_channels, 16)
        self.D1 = DownSampling(16)
        self.C2 = KConv(16, 32)
        self.D2 = DownSampling(32)
        self.C3 = KConv(32, 64)
        self.D3 = DownSampling(64)
        self.C4 = KConv(64, 128)
        self.D4 = DownSampling(128)
        self.C5 = KConv(128, 256)

        self.U1 = UpSampling(256)
        self.C6 = KConv(256, 128)
        self.U2 = UpSampling(128)
        self.C7 = KConv(128, 64)
        self.U3 = UpSampling(64)
        self.C8 = KConv(64, 32)
        self.U4 = UpSampling(32)
        self.C9 = KConv(32, 16)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(
            in_channels=16,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        return self.Th(self.pred(O4))
