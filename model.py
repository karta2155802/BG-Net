import torch
import torch.nn.functional as F
from torch import nn
from torchvision import ops
import matplotlib.pyplot as plt
import numpy as np
import cv2

from networks.backbone import FPN
from networks.hand_head import hand_Encoder, hand_regHead
from networks.object_head import obj_regHead, Pose2DLayer
from networks.mano_head import mano_regHead
from networks.CR import Transformer
from networks.loss import Joint2DLoss, ManoLoss, ObjectLoss
from networks.seg_head import SegmHead
from utils.vis import show_figure, save_figure, vis_bbox, vis_keypoints, save_combined_figure


class HONet(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, mano_neurons=[1024, 512], coord_change_mat=None,
                 reg_object=True, pretrained=True):

        super(HONet, self).__init__()

        self.out_res = roi_res

        # FPN-Res50 backbone
        self.hand_base_net = FPN(pretrained=pretrained)
        self.obj_base_net = FPN(pretrained=pretrained)

        self.seg_head = SegmHead(in_dim=channels, class_dim=1, upsample=False)
        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead(mano_layer, feature_size=mano_neurons[0],
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks
        self.o_transformer = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth, num_heads=transformer_head)


    def forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, batch_idx=None):
        batch = imgs.shape[0]

        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)

        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)
        
        # P2 from FPN Network
        hand_P2 = self.hand_base_net(imgs) # B x 256 x 64 x 64
        
        # 4 here is the downscale size in FPN network(P2)
        Fh = ops.roi_align(hand_P2, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0, sampling_ratio=-1)  # hand B x 256 x 32 x 32
        Fho = ops.roi_align(hand_P2, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0, sampling_ratio=-1)
        Finter = ops.roi_align(hand_P2, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0, sampling_ratio=-1)
        # hand forward
        out_hm, encoding, preds_joints = self.hand_head(Fh) # B x 256 x 32 x 32
        mano_encoding = self.hand_encoder(out_hm, encoding) # B x 1024
        pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, mano_params=mano_params)

        # obj forward
        pred_obj_results = None
        obj_mask = None
        if self.reg_object:
            obj_P2 = self.obj_base_net(imgs)
            Fo = ops.roi_align(obj_P2, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0, sampling_ratio=-1)  # obj
            obj_mask = self.seg_head(Fo)
            y, attn_score = self.o_transformer(Fo, Fho)
            out_fm, x = self.obj_head(y)
            pred_obj_results = self.obj_reorgLayer(out_fm)
            
            '''ii = imgs[-1].permute(1,2,0).cpu().numpy()
            hh = hand_P2[-1].sum(dim=0).cpu().numpy()[1:63, 1:63]
            oo = obj_P2[-1].sum(dim=0).cpu().numpy()[1:63, 1:63] * -1
            Ro = Fo[-1].sum(dim=0).cpu().numpy() * -1
            Rho = Fho[-1].sum(dim=0).cpu().numpy()
            Rinter = Finter[-1].sum(dim=0).cpu().numpy()
            o_fm = x[-1].sum(dim=0).cpu().numpy() * -1
            mm = obj_mask[-1].permute(1,2,0).repeat(1,1,3).detach().cpu().numpy()
            
            ii = np.ascontiguousarray((ii*255).astype(np.uint8))
            bbox = bbox_obj.cpu().numpy()[-1].astype(np.int32)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            vv = vis_bbox(ii, bbox, thickness=1)

            q_point = np.random.randint(0, 32, size=2) # random query point(x, y)
            q_point_ori = [q_point[0]*w/31+bbox[0], q_point[1]*h/31+bbox[1]]
            print(q_point)
            query_img = vis_keypoints(ii, [q_point_ori], radius=5)
            output_list = [query_img]
            # attn_socre 16x4x1024x1024
            try:
                for i in range(1,2):
                    attn = attn_score[-1][i].view(self.out_res, self.out_res, self.out_res, self.out_res).cpu().numpy()
                    attn = np.expand_dims(attn, axis=-1).repeat(3, axis=2) # 32x32x32x32x3
                    attn1 = attn[q_point[1], q_point[0], :,:]
                    attn1 = cv2.resize(attn1, (w, h))
                    attn1 = (((attn1-attn1.min()) / (attn1.max()-attn1.min())) * 255).astype(np.uint8) # normalize
                    
                    attn1 = cv2.applyColorMap(attn1, cv2.COLORMAP_JET)
                    attn_img = np.copy(ii)
                    attn_img[bbox[1]:bbox[1]+h, bbox[0]:bbox[0]+w] = cv2.addWeighted(ii[bbox[1]:bbox[1]+h, bbox[0]:bbox[0]+w], 1.0 - 0.5, attn1, 0.5, 0)
                    output_list.append(attn_img)

                #show_figure(output_list)
                save_combined_figure(output_list, "exp-results/attn", img_name=str(batch_idx))
            except:
                print("fail")'''
            #save_figure([ii, hh, oo, mm, o_fm, Ro, Rho, Rinter], "exp-results")
            #show_figure([ii, hh, oo, mm, o_fm, Ro, Rho, Rinter])
            #save_combined_figure([vv, mm], "exp-results/mask", img_name=str(batch_idx))

        return preds_joints, pred_mano_results, gt_mano_results, pred_obj_results, obj_mask


class HOModel(nn.Module):

    def __init__(self, honet, mano_lambda_verts3d=None,
                 mano_lambda_joints3d=None,
                 mano_lambda_manopose=None,
                 mano_lambda_manoshape=None,
                 mano_lambda_regulshape=None,
                 mano_lambda_regulpose=None,
                 lambda_joints2d=None,
                 lambda_objects=None):

        super(HOModel, self).__init__()
        self.honet = honet
        # supervise when provide mano params
        self.mano_loss = ManoLoss(lambda_verts3d=mano_lambda_verts3d,
                                  lambda_joints3d=mano_lambda_joints3d,
                                  lambda_manopose=mano_lambda_manopose,
                                  lambda_manoshape=mano_lambda_manoshape)
        self.joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)
        # supervise when provide hand joints
        self.mano_joint_loss = ManoLoss(lambda_joints3d=mano_lambda_joints3d,
                                        lambda_regulshape=mano_lambda_regulshape,
                                        lambda_regulpose=mano_lambda_regulpose)
        # object loss
        self.object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)

    def forward(self, imgs, bbox_hand, bbox_obj, joints_uv=None,
                mano_params=None, obj_p2d_gt=None, obj_mask=None, batch_idx=None):
        if self.training:
            losses = {}
            total_loss = 0
            preds_joints2d, pred_mano_results, gt_mano_results, preds_obj, pred_obj_mask = self.honet(imgs, bbox_hand, bbox_obj, mano_params=mano_params)
            if mano_params is not None:
                mano_total_loss, mano_losses = self.mano_loss.compute_loss(pred_mano_results, gt_mano_results)
                total_loss += mano_total_loss
                for key, val in mano_losses.items():
                    losses[key] = val
            if joints_uv is not None:
                joint2d_loss, joint2d_losses = self.joint2d_loss.compute_loss(preds_joints2d, joints_uv)
                for key, val in joint2d_losses.items():
                    losses[key] = val
                total_loss += joint2d_loss
            if preds_obj is not None:
                obj_total_loss, obj_losses = self.object_loss.compute_loss(obj_p2d_gt, obj_mask, preds_obj)
                for key, val in obj_losses.items():
                    losses[key] = val
                total_loss += obj_total_loss

                border_loss = self.object_loss.compute_border_loss(preds_obj)
                losses["border_loss"] = border_loss.detach().cpu()
                total_loss += border_loss
            if pred_obj_mask is not None:
                mask_loss = 100 * F.binary_cross_entropy(pred_obj_mask.squeeze(), obj_mask.squeeze())
                losses["obj_mask_loss"] = mask_loss.detach().cpu()
                total_loss += mask_loss
            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0
            return total_loss, losses
        else:
            preds_joints, pred_mano_results, gt_mano_results, preds_obj, _ = self.honet(imgs, bbox_hand, bbox_obj, mano_params=mano_params, batch_idx=batch_idx)
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj