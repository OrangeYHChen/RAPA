from __future__ import absolute_import
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['Net']

class Net(nn.Module):
    def __init__(self, num_classes, final_dim = 1024, **kwargs):
        super(Net, self).__init__()
        self.feat_dim = 2048
        self.final_dim = final_dim

        resnet50 = torchvision.models.resnet50(pretrained=False)
        resnet50.load_state_dict(torch.load('./weights/resnet50-19c8e357.pth'))
        self.backbone1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,)
        self.backbone2 = nn.Sequential(resnet50.layer2,)
        self.backbone3 = nn.Sequential(resnet50.layer3,)
        self.backbone4 = nn.Sequential(resnet50.layer4,)

        self.weight_norm = nn.Sequential(nn.BatchNorm2d(self.feat_dim),
                                         nn.Sigmoid())  # weight map norm

        self.temporalBlock = TemporalAttentionBlock(2048)
        self.referenceBlock = ReferenceSelectBlock(2048)


        self.classifier1 = ClassBlock(self.feat_dim, num_classes, self.final_dim)
        self.classifier2 = ClassBlock(self.feat_dim * 2, num_classes, self.final_dim)
        self.classifier3 = ClassBlock(self.feat_dim * 2, num_classes, self.final_dim)
        self.classifier4 = ClassBlock(self.feat_dim * 2, num_classes, self.final_dim)

    def forward(self, x, head_map, body_map, leg_map):  # x_align size: b*3*256*128
        if not self.training:
            head_map = head_map.squeeze(dim=0)
            body_map = body_map.squeeze(dim=0)
            leg_map = leg_map.squeeze(dim=0)

        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.backbone3(x)
        x = self.backbone4(x)

        # global branch
        global_x = x  # size: bt*2048*8*4
        global_x = self.temporalBlock(global_x, b, t)  # size: b*t*2048
        global_f = torch.sum(global_x, 1)  # size: b*2048
        global_f = global_f.unsqueeze(-1).unsqueeze(-1)  # size: b*2048*1*1
        global_f, _, global_label = self.classifier1(global_f)  # size: b*256

        # part branch
        part_x = x
        map_index = self.referenceBlock(part_x, b, t)  # size: b
        # print(head_map.size())  # size: b*t*8*4
        head_map, body_map, leg_map = self.get_part_map(head_map, body_map, leg_map, map_index, b)
        part_x = part_x.view(b, t, part_x.size(1), part_x.size(2), part_x.size(3))
        kernel1, kernel2 = self.get_part_kernel(head_map, part_x, map_index)  # size: b*2048*1*1
        kernel3, kernel4 = self.get_part_kernel(body_map, part_x, map_index)
        kernel5, kernel6 = self.get_part_kernel(leg_map, part_x, map_index)

        # region align branch
        align_part_x = x  # size: bt*2048*8*4
        align_part_x = align_part_x.view(b, t, align_part_x.size(1), align_part_x.size(2), align_part_x.size(3))  # size: b*t*2048*8*4
        align_output1, align_weight_output1 = self.region_align(align_part_x, kernel1)  # 0: size: b*t*2048; 1: size: b*2048*1*1
        align_output2, align_weight_output2 = self.region_align(align_part_x, kernel2)
        align_output3, align_weight_output3 = self.region_align(align_part_x, kernel3)
        align_output4, align_weight_output4 = self.region_align(align_part_x, kernel4)
        align_output5, align_weight_output5 = self.region_align(align_part_x, kernel5)
        align_output6, align_weight_output6 = self.region_align(align_part_x, kernel6)
        region1_f = torch.cat([align_weight_output1, align_weight_output2], dim=1)  # size: b*4096*1*1
        region2_f = torch.cat([align_weight_output3, align_weight_output4], dim=1)
        region3_f = torch.cat([align_weight_output5, align_weight_output6], dim=1)
        region1_f, _, region1_label = self.classifier2(region1_f)
        region2_f, _, region2_label = self.classifier3(region2_f)
        region3_f, _, region3_label = self.classifier4(region3_f)

        if not self.training:
            return torch.cat([global_f, region1_f, region2_f, region3_f], dim=1)
        else:
            return global_label, global_f,\
                   region1_label, region1_f, region2_label, region2_f, region3_label, region3_f, \
                   align_output1, align_output2, align_output3, align_output4, align_output5, align_output6

    def get_part_kernel(self, part_map, part_x, index):
        # first_frame = part_x[:, 0, :, :, :]
        refer_frame = []
        for i in range(index.size(0)):
            refer = part_x[i:i+1, index[i], :, :, :]
            refer_frame.append(refer)
        refer_frame = torch.cat(refer_frame, dim=0)  # size: b*2048*8*4
        first_x = torch.mul(refer_frame, part_map)  # size: b*2048*8*4
        kernel_max = F.max_pool2d(first_x, first_x.size()[2:])  # size: b*2048*1*1

        part_count = torch.sum(part_map, dim=2, keepdim=False)  # map size: b*1*4
        part_count = torch.sum(part_count, dim=2, keepdim=False)  # map size: b*1
        part_count = part_count.unsqueeze(dim=1).unsqueeze(dim=1)  # map size: b*1*1*1
        kernel_avg = F.avg_pool2d(first_x, first_x.size()[2:])  # size: b*2048*1*1
        kernel_avg = torch.div(kernel_avg, part_count)
        kernel_avg = kernel_avg * 32

        return kernel_max, kernel_avg

    def region_align(self, align_part_x, kernel):
        b = align_part_x.size(0)
        t = align_part_x.size(1)
        align_map_x = self.get_relation_map(align_part_x, kernel)  # size: b*t*2048*8*4
        align_map_x = align_map_x.view(b * t, align_map_x.size(2), align_map_x.size(3), align_map_x.size(4))  # size: bt*2048*8*4
        align_map_x = self.weight_norm(align_map_x)  # size: bt*2048*8*4
        align_map_x = 1 - align_map_x  # size: bt*2048*8*4
        align_part_x = align_part_x.view(b * t, align_part_x.size(2), align_part_x.size(3), align_part_x.size(4))  # size: bt*2048*8*4
        align_output = torch.mul(align_part_x, align_map_x)  # size: bt*2048*8*4
        align_output = F.max_pool2d(align_output, align_output.size()[2:])  # size: bt*2048*1*1
        align_output = align_output.view(b, t, -1)  # size: b*t*2048
        info_score = F.avg_pool2d(align_map_x, align_map_x.size()[2:])  # size: bt*2048*1*1
        info_score = info_score.view(b, t, info_score.size(1))  # size: b*t*2048
        info_score = torch.sigmoid(info_score)  # size: b*t*2048
        align_weight_output = torch.mul(align_output, info_score)  # size: b*t*2048
        align_weight_output = torch.sum(align_weight_output, dim=1, keepdim=False)  # size: b*2048
        align_weight_output = align_weight_output.unsqueeze(-1).unsqueeze(-1)  # size: b*2048*1*1
        return align_output, align_weight_output

    def get_relation_map(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        frame = x.size(1)
        kernel = kernel.unsqueeze(1)  # size: batch * 1 * channel * k_h * k_w
        kernel = kernel.expand(batch, frame, channel, kernel.size(3), kernel.size(4)).contiguous()  # size: batch * frame * channel * k_h * k_w
        out = torch.pow((x-kernel), 2)  # size: batch * frame * channel * k_h * k_w
        return out

    def get_part_map(self, head_map, body_map, leg_map, index, b):
        # map size: b*t*8*4; index size: b
        head_map_selected = []
        body_map_selected = []
        leg_map_selected = []
        for i in range(b):
            head_map_selected.append(torch.index_select(head_map[i], 0, index[i]).unsqueeze(0))
            body_map_selected.append(torch.index_select(body_map[i], 0, index[i]).unsqueeze(0))
            leg_map_selected.append(torch.index_select(leg_map[i], 0, index[i]).unsqueeze(0))
        head_map_selected = torch.cat(head_map_selected, dim=0)  # size: b*1*8*4
        body_map_selected = torch.cat(body_map_selected, dim=0)
        leg_map_selected = torch.cat(leg_map_selected, dim=0)
        return head_map_selected, body_map_selected, leg_map_selected

class TemporalAttentionBlock(nn.Module):
    def __init__(self, feat_dim):
        super(TemporalAttentionBlock, self).__init__()
        self.feat_dim = feat_dim
        t_conv = nn.Conv2d(feat_dim, 1, [3, 3], padding=1)
        self.temp_att_conv = nn.Sequential(t_conv,
                                           nn.BatchNorm2d(1),
                                           nn.Sigmoid())  # temporal attention conv

    def forward(self, x, b, t):
        x = F.avg_pool2d(x, x.size()[2:])  # size: bt*2048*1*1
        t_att = self.temp_att_conv(x)  # size: bt * 1 * 1 * 1
        t_att = t_att.view(b, t, 1)  # size: b*t*1
        t_att = t_att.expand(b, t, self.feat_dim)  # size: b*t*2048
        x = x.view(b, t, -1)  # size: b*t*2048
        x = torch.mul(x, t_att)  # size: b*t*2048
        return x


class ReferenceSelectBlock(nn.Module):
    def __init__(self, feat_dim):
        super(ReferenceSelectBlock, self).__init__()
        q_conv = nn.Conv2d(feat_dim, 1, [3, 3], padding=1)
        self.quality_conv = nn.Sequential(q_conv,
                                          nn.BatchNorm2d(1),
                                          nn.Sigmoid())

    def forward(self, x, b, t):
        # size: bt*2048*8*4
        x = F.avg_pool2d(x, x.size()[2:])  # size: bt*2048*1*1
        quality = self.quality_conv(x)  # size: bt*1*1*1
        quality = quality.view(b, t)  # size: b*t
        quality_index = quality.argmax(dim=1)  # size: b
        return quality_index




class ClassBlock(nn.Module):
    def __init__(self, feat_dim, class_num, num_bottleneck=1024):
        super(ClassBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(feat_dim, num_bottleneck, [1, 1])
        self.conv1x1.apply(weights_init_kaiming)

        self.bottleneck = nn.BatchNorm1d(num_bottleneck)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(num_bottleneck, class_num, bias=False)
        self.classifier.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conv1x1(x)  # b * 512 * 1 * 1
        x_1 = x.view(x.size(0), -1)  # b * 512
        x_2 = self.bottleneck(x_1)
        x_label = self.classifier(x_2)
        return x_1, x_2, x_label


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)




