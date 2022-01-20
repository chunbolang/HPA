import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP
from model.atten import Feature_Reweighting
from util.util import get_train_val_set, get_metirc


class Decoder(nn.Module):
    def __init__(self, low_dim=256):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(low_dim, 48, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)

    def forward(self, x, low_fea):
        low_fea = self.conv1(low_fea)
        low_fea = self.relu(low_fea)
        x_4 = F.interpolate(x, size=low_fea.size()[2:4], mode='bilinear', align_corners=True)
        x_4_cat = torch.cat((x_4, low_fea), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        return x_4_cat        

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.ppm_scales = args.ppm_scales
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        if args.aux_criterion == 'MSE':
            self.aux_criterion = nn.MSELoss()
        elif args.aux_criterion == 'BCE':
            self.aux_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.count = 0
        self.print_freq = args.print_freq/2
        # self.print_freq = 1

        self.pretrained = True
        self.classes = 2
        
        assert self.layers in [50, 101, 152]
    
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('INFO: Using ResNet {}'.format(self.layers))
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            else:
                resnet = models.resnet152(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
            arch_str = 'vgg'
            low_dim_dict = {'layer0': 64,
                            'layer1': 128,
                            'layer2': 256,
                            'layer3': 512,
                            'layer4': 512}
            self.low_fea_id = args.low_fea[-1]
            low_fea_dim = low_dim_dict[args.low_fea]
        else:
            fea_dim = 1024 + 512       
            arch_str = 'resnet' + str(self.layers)
            low_dim_dict = {'layer0': 128,
                            'layer1': 256,
                            'layer2': 512,
                            'layer3': 1024,
                            'layer4': 2048}
            self.low_fea_id = args.low_fea[-1]                            
            low_fea_dim = low_dim_dict[args.low_fea]


        self.Global_Pros_2 = torch.load('./initmodel/prototypes/{}/{}/stage2.pth'.format(self.dataset, arch_str))  # 512-d
        self.Global_Pros_3 = torch.load('./initmodel/prototypes/{}/{}/stage3.pth'.format(self.dataset, arch_str))  # 1024-d

        if self.dataset == 'pascal' or self.dataset == 'coco':
            self.train_set, self.val_set = get_train_val_set(args)
            self.Global_Pros_2 = self.Global_Pros_2[np.array(self.train_set)-1]
            self.Global_Pros_3 = self.Global_Pros_3[np.array(self.train_set)-1]
        self.Global_Pros_2 = torch.nn.Parameter(self.Global_Pros_2, requires_grad=False)
        self.Global_Pros_3 = torch.nn.Parameter(self.Global_Pros_3, requires_grad=False)
        

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )
        self.down_pros = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )

        # ---------------------- PAM ----------------------
        self.mode = 'Learnable'
        self.correlation_dim = 8
        self.relation_coding = nn.Sequential(                  
            nn.Conv2d(reduce_dim*2, self.correlation_dim, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.correlation_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid())

        mask_add_num = 2
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))

        self.ASPP = ASPP(reduce_dim)

        # Segmentation
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        ) 

        # ---------------------- CRD ----------------------
        self.FR = Feature_Reweighting()  
        self.decoder = Decoder(low_dim=low_fea_dim)

        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
            {'params': model.down_query.parameters()},
            {'params': model.down_supp.parameters()},
            {'params': model.down_pros.parameters()},
            {'params': model.relation_coding.parameters()},
            {'params': model.init_merge.parameters()},
            {'params': model.ASPP.parameters()},
            {'params': model.res1.parameters()},
            {'params': model.FR.parameters()},
            {'params': model.decoder.parameters()},
            {'params': model.res2.parameters()},        
            {'params': model.cls.parameters()}],
            lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        
        return optimizer


    def forward(self, x, s_x, s_y, y, cat_idx=None):
        self.count += 1
        x_size = x.size()
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        # Support Feature
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_ave_list = [0 for _ in range(5)] 
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask)
            supp_pro_list.append(supp_pro)
            for j in range(5):
                supp_feat_ave_list[j] += eval('supp_feat_'+str(j)) / self.shot

        # Global Prototypes
        bs = supp_feat.shape[0]
        num_Pros = self.Global_Pros_2.shape[0]
        Global_Pros_2 = self.Global_Pros_2.expand(bs,-1,-1).unsqueeze(-1).unsqueeze(-1)
        Global_Pros_3 = self.Global_Pros_3.expand(bs,-1,-1).unsqueeze(-1).unsqueeze(-1)
        concat_Pros = torch.cat([Global_Pros_2,Global_Pros_3], dim=2)
        for p_id in range(num_Pros):
            if p_id==0:
                Global_Pros = self.down_pros(concat_Pros[:,p_id,:,:,:]).unsqueeze(1)
            else:
                Global_Pros = torch.cat([Global_Pros,self.down_pros(concat_Pros[:,p_id,:,:,:]).unsqueeze(1)], dim=1)

        # Prior Similarity Mask (see PFENet for more details)
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask         
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)

        if self.shot > 1:
            supp_pro = supp_pro_list[0]
            for i in range(1, len(supp_pro_list)):
                supp_pro += supp_pro_list[i]
            supp_pro /= len(supp_pro_list) 

        # --------------  Prototype Activation Module (PAM) --------------
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(0,num_Pros, device='cuda')
            # using <torch.gather> might simplify the loop below
            for b_id in range(bs):
                c_id = cat_idx[0][b_id].reshape(1)
                c_ids = c_id_array[c_id_array!=c_id]
                if b_id == 0:
                    Gfg_Pro = Global_Pros[b_id,c_id].unsqueeze(0)
                    bg_Pros = Global_Pros[b_id,c_ids].unsqueeze(0)
                else:
                    Gfg_Pro = torch.cat([Gfg_Pro,Global_Pros[b_id,c_id].unsqueeze(0)], dim=0)
                    bg_Pros = torch.cat([bg_Pros,Global_Pros[b_id,c_ids].unsqueeze(0)], dim=0)
            fg_Pros = torch.cat([supp_pro.unsqueeze(1),Gfg_Pro], dim=1)
        else:
            bg_Pros = Global_Pros
            fg_Pros = supp_pro.unsqueeze(1)

        bg_Pro_act_maps = self.query_region_activate(query_feat, bg_Pros, self.mode)
        fg_Pro_act_maps = self.query_region_activate(query_feat, fg_Pros, self.mode)

        bg_map_value, bg_map_indice = torch.max(bg_Pro_act_maps, dim=1)
        fg_map_value, fg_map_indice = torch.max(fg_Pro_act_maps, dim=1)

        concat_map = fg_map_value.clone()
        bg_valid_coord = torch.where(bg_map_value > fg_map_value)                                   # Max
        concat_map[bg_valid_coord[0],bg_valid_coord[1],bg_valid_coord[2]] = 0                       # Erasing
        concat_map = concat_map.unsqueeze(1)

        # Tile & Cat
        concat_feat = torch.zeros_like(query_feat)

        for b_id in range(bs):
            fg_map_indice_batch = fg_map_indice[b_id]
            fg_Pros_batch = fg_Pros[b_id, fg_map_indice_batch].squeeze(-1).squeeze(-1).permute(2,0,1)
            concat_feat[b_id] = fg_Pros_batch
        
        bg_map_indice_valid = bg_map_indice[bg_valid_coord[0],bg_valid_coord[1],bg_valid_coord[2]]
        concat_feat[bg_valid_coord[0],:,bg_valid_coord[1],bg_valid_coord[2]] = bg_Pros[bg_valid_coord[0],bg_map_indice_valid].squeeze(-1).squeeze(-1)

        merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask, concat_map], 1)   # 256+256+1+1
        merge_feat = self.init_merge(merge_feat)

        # Segmentation
        query_feat = self.ASPP(merge_feat)
        query_feat = self.res1(query_feat)               # 1024->256

        # --------------  Cross-Referenced Decoder (CRD) --------------
        referenced_feat, alpha = self.FR(supp_feat_ave_list[int(self.low_fea_id)], eval('query_feat_' + self.low_fea_id))
        query_feat = self.decoder(query_feat, referenced_feat)

        query_feat = self.res2(query_feat) + query_feat
        net_out = self.cls(query_feat)
        
        # Output
        if self.zoom_factor != 1:
            net_out = F.interpolate(net_out, size=(h, w), mode='bilinear', align_corners=True)

        # Viz
        if self.count % self.print_freq == 0:
            pred = 2*torch.ones((bs,query_feat.shape[-2],query_feat.shape[-1]),device='cuda')

            pred[bg_valid_coord[0],bg_valid_coord[1],bg_valid_coord[2]] = 0
            y_resize = F.interpolate(y.float().unsqueeze(1), size=(query_feat.shape[-2],query_feat.shape[-1]), mode='nearest').squeeze(1)
            
            Pre, Rec = get_metirc(pred, y_resize, 2, 255)
            print('alpha: {:.2f} \t fg_map: {:.2f}|{:.2f} (min|max) \t bg: {:.2f}|{:.2f} (Pre|Rec)'.format(\
                                        alpha.item(), \
                                        fg_map_value.min(), fg_map_value.max(),\
                                        Pre[0].item(), Rec[0].item()))

        # Loss
        if self.training:
            main_loss = self.criterion(net_out, y.long())

            prob_map = F.interpolate(concat_map, size=(h, w), mode='bilinear', align_corners=True)
            if isinstance(self.aux_criterion, nn.modules.loss.MSELoss):
                aux_loss = self.aux_criterion(prob_map[0][y!=255], y[y!=255])/bs                
            elif isinstance(self.aux_criterion, nn.modules.loss.CrossEntropyLoss):
                map_out = torch.cat([1-prob_map,prob_map], dim=1)
                aux_loss = self.criterion(map_out, y.long())
            else:
                print('Error, this type of loss is not supported...')

            return net_out.max(1)[1], main_loss, aux_loss
        else:
            return net_out


    def query_region_activate(self, query_fea, prototypes, mode):
        """             
        Input:  query_fea:      [b, c, h, w]
                prototypes:     [b, n, c, 1, 1]
                mode:           Cosine/Conv/Learnable
        Output: activation_map: [b, n, h, w]
        """
        b, c, h, w = query_fea.shape
        n = prototypes.shape[1]
        que_temp = query_fea.reshape(b, c, h*w)

        if mode == 'Conv':
            map_temp = torch.bmm(prototypes.squeeze(-1).squeeze(-1), que_temp)  # [b, n, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            return activation_map

        elif mode == 'Cosine':
            que_temp = que_temp.unsqueeze(dim=1)           # [b, 1, c, h*w]
            prototypes_temp = prototypes.squeeze(dim=-1)   # [b, n, c, 1]
            map_temp = nn.CosineSimilarity(2)(que_temp, prototypes_temp)  # [n, c, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            activation_map = (activation_map+1)/2          # Normalize to (0,1)
            return activation_map

        elif mode == 'Learnable':
            for p_id in range(n):
                prototypes_temp = prototypes[:,p_id,:,:,:]                         # [b, c, 1, 1]
                prototypes_temp = prototypes_temp.expand(b, c, h, w)
                concat_fea = torch.cat([query_fea, prototypes_temp], dim=1)        # [b, 2c, h, w]                
                if p_id == 0:
                    activation_map = self.relation_coding(concat_fea)              # [b, 1, h, w]
                else:
                    activation_map_temp = self.relation_coding(concat_fea)              # [b, 1, h, w]
                    activation_map = torch.cat([activation_map,activation_map_temp], dim=1)
            return activation_map

