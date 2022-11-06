from __future__ import absolute_import, division, print_function

from layers import *
import numpy as np
import torch.nn as nn
import torch
from networks.rsu_layer import *

class EncoderDisp(nn.Module):
    def __init__(self, bott_channels, out_channels, bottleneck):
        super(EncoderDisp, self).__init__()
        self.bottleneck = bottleneck
        self.disp = nn.Sequential(
            nn.Conv2d(bott_channels, out_channels, 3, 1, 1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        features = upsample(self.bottleneck(inputs))
        out = self.disp(features)
        return out

class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels: int = 1, encoder_type = "resnet", full_scale=False):
        super(HRDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.full_scale = full_scale
        self.encoder_type = encoder_type
        self.mobile_encoder = encoder_type == "mobile"
        self.densenet_encoder = encoder_type == "densenet"
        if self.mobile_encoder:
            self.num_ch_dec = np.array([4, 12, 20, 40, 80])
        elif self.densenet_encoder:
            self.num_ch_dec = np.array([64, 128, 256, 512 ,512])
        elif encoder_type == "resnet":
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)
                
                # for densenet encoder
                if self.densenet_encoder:
                    if i == 3 and j == 1:
                        self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_out, num_ch_out)
                    if i == 0 and j == 4:
                        self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(128, num_ch_out)
                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            if self.mobile_encoder:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row]*2*(col-1),
                                                                        output_channel=self.num_ch_dec[row] * 2)
            else:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if self.mobile_encoder:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(
                    self.num_ch_enc[row]+ self.num_ch_enc[row + 1] // 2 +
                    self.num_ch_dec[row]*2*(col-1), self.num_ch_dec[row] * 2)
            else:
                if col == 1:
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                            self.num_ch_enc[row], self.num_ch_dec[row + 1])
                else:
                    self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        if self.mobile_encoder:
            self.convs["dispConvScale0"] = Conv3x3(4, self.num_output_channels)
            self.convs["dispConvScale1"] = Conv3x3(8, self.num_output_channels)
            self.convs["dispConvScale2"] = Conv3x3(24, self.num_output_channels)
            self.convs["dispConvScale3"] = Conv3x3(40, self.num_output_channels)
        else:
            for i in range(4):
                self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        if self.full_scale:
            self.encoder_disps = nn.ModuleList()
            bottlenecks = [RSU7, RSU6, RSU5, RSU4]
            mid_channels = [4, 4, 8, 24]
            in_channels = num_ch_enc[:-1]
            out_channels = [8, 12, 20, 40]
            for c, mid_c, bott_c, bottleneck in zip(in_channels, mid_channels, out_channels, bottlenecks):
                self.encoder_disps.append(EncoderDisp(bott_c, num_output_channels, bottleneck(c, mid_c, bott_c, False)))


    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1 and not self.mobile_encoder:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)
            #print(f'index:{index} - feature shape:{features["X_" + index].shape}')
        x = features["X_04"] # X^5_e
        x = self.convs["X_04_Conv_0"](x) # X^5_d
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[("disp", 0)] = self.sigmoid(self.convs["dispConvScale0"](x))
        outputs[("disp", 1)] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        outputs[("disp", 2)] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))

        if self.full_scale:
            for i in range(len(input_features[:-1])):
                outputs[("encoder_disp", i)] = self.encoder_disps[i](input_features[i])
        return outputs