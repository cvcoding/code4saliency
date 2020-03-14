import torch
import torch.nn as nn
from Attn import Attn
import math
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)


            # nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, padding=1, bias=False),
            # nn.BatchNorm1d(outchannel),
            # nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm1d(outchannel),
            # nn.Conv1d(outchannel, outchannel, kernel_size=1, stride=1, padding=1, bias=False),
            # nn.BatchNorm1d(outchannel),
            # nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MyDNNDecoder(nn.Module):
    def __init__(self):
        super(MyDNNDecoder, self).__init__()
        self.attn = Attn('general', 64, 64).cuda()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=9,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(),
        )
        self.maxpooling = nn.MaxPool1d(3)  # Adaptive

        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)   #3
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)  # 4
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)  # 6
        self.layer4 = self.make_layer(ResidualBlock, 128, 3, stride=2)  # 3
        self.avgpooling = nn.AvgPool1d(3, stride=1)

        self.fc = nn.Sequential(
            nn.Linear(512, 180),  # (96,136)
            nn.Sigmoid(),
        )
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)



    def forward(self, encoder_outputs, edges_batch):

        # target_length = encoder_outputs.data.shape[1]
        # batchsize = encoder_outputs.data.shape[0]
        # feature_len = encoder_outputs.data.shape[2]
        # mtix = torch.zeros(batchsize, target_length,target_length)
        # for ind in range(batchsize):
        #     for edge in edges_batch[ind]:
        #         mtix[ind, edge[0], edge[1]] = 1
        #
        # cnn_input = []
        # neighbor_num = 6
        # for current_index in range(target_length):
        #     mtix4sp = mtix[:, current_index, :]
        #
        #     encoder_outputs_neighbor = torch.zeros(batchsize, neighbor_num, 64).cuda()
        #
        #     for kk in range(batchsize):
        #         jj = 0
        #         for ii in range(target_length):
        #             if mtix4sp[kk, ii] == 1:
        #                 if jj >= neighbor_num:
        #                     break
        #                 else:
        #                     encoder_outputs_neighbor[kk, jj, :] = encoder_outputs[kk, ii, :]
        #                     jj = jj + 1
        #
        #     attn_weights = self.attn(encoder_outputs, current_index, encoder_outputs_neighbor,neighbor_num)
        #     context = torch.zeros(batchsize, feature_len).cuda()
        #
        #     for i in range(batchsize):
        #         # mm = attn_weights[i, :].unsqueeze(1).repeat(1, 64)
        #         # context[i, :] = mm.mul(encoder_outputs_neighbor[i, :, :]).sum(0)
        #         context[i, :] = attn_weights[i, :].matmul(encoder_outputs_neighbor[i, :, :])
        #
        #     cnn_input.append(context)
        # cnn_input = torch.stack(cnn_input).permute(1, 2, 0).cuda()

        target_length = encoder_outputs.data.shape[1]
        batchsize = encoder_outputs.data.shape[0]
        feature_len = encoder_outputs.data.shape[2]

        cnn_input = []
        attn_weights = self.attn(encoder_outputs, target_length)
        for current_index in range(target_length):

            context = torch.zeros(batchsize, feature_len).cuda()
            for i in range(batchsize):
                # mm = attn_weights[current_index, i, :].unsqueeze(1).repeat(1, 64)
                # context[i, :] = mm.mul(encoder_outputs[i, :, :]).sum(0)
                context[i, :] = attn_weights[current_index, i, :].matmul(encoder_outputs[i, :, :])
            cnn_input.append(context)
        cnn_input = torch.stack(cnn_input).permute(1, 2, 0).cuda()


        out = self.conv1(cnn_input)

        out = self.maxpooling(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpooling(out)

        out = out.view(out.size(0), -1).unsqueeze(1)
        out = F.dropout(out, 0.1, training=self.training)

        # out = F.relu(out)
        # out = self.maxpooling(out)
        output = self.fc(out).squeeze(1)

        return output

