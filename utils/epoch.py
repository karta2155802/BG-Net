import os
import time
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils.utils import progress_bar as bar, AverageMeters, dump
from dataset.ho3d_util import filter_test_object_ho3d, get_unseen_test_object, filter_test_object_dexycb
from utils.metric import eval_object_pose, eval_batch_obj, eval_hand, eval_hand_pose_result
from utils.vis import vis_3d, vis_keypoints, vis_keypoints_with_skeleton, vis_bbox


def single_epoch(args, loader, model, epoch=None, optimizer=None, indices_order=None):
    save_path = args.host_folder
    use_cuda = args.use_cuda
    train = not args.evaluate

    time_meters = AverageMeters()

    if train:
        print(f"training epoch: {epoch + 1}")
        avg_meters = AverageMeters()
        model.train()

    else:
        model.eval()
        # object evaluation
        REP_res_dict, ADD_res_dict= {}, {}
        diameter_dict = loader.dataset.obj_diameters
        mesh_dict = loader.dataset.obj_mesh
        if args.use_ho3d:
            mesh_dict, diameter_dict = filter_test_object_ho3d(mesh_dict, diameter_dict)
            unseen_objects = get_unseen_test_object()
        else:
            mesh_dict, diameter_dict = filter_test_object_dexycb(mesh_dict, diameter_dict)
            unseen_objects = []
        
        for k in mesh_dict.keys():
            REP_res_dict[k] = []
            ADD_res_dict[k] = []
        if args.use_ho3d:
             # save hand results for online evaluation
             xyz_pred_list, verts_pred_list = list(), list()
        else:
            #hand evaluation
            hand_eval_result = [[], []]

    end = time.time()
    for batch_idx, sample in enumerate(loader):
        if train:
            assert use_cuda and torch.cuda.is_available(), "requires cuda for training"
            imgs = sample["img"].float().cuda()
            bbox_hand = sample["bbox_hand"].float().cuda()
            bbox_obj = sample["bbox_obj"].float().cuda()

            mano_params = sample["mano_param"].float().cuda()
            joints_uv = sample["joints2d"].float().cuda()
            obj_p2d_gt = sample["obj_p2d"].float().cuda()
            obj_mask = sample["obj_mask"].float().cuda()
            #print(sample["mano_param"])

            # measure data loading time
            time_meters.add_loss_value("data_time", time.time() - end)
            # model forward
            model_loss, model_losses = model(imgs, bbox_hand, bbox_obj, mano_params=mano_params,
                                             joints_uv=joints_uv, obj_p2d_gt=obj_p2d_gt, obj_mask=obj_mask)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            for key, val in model_losses.items():
                if val is not None:
                    avg_meters.add_loss_value(key, val)

            # measure elapsed time
            time_meters.add_loss_value("batch_time", time.time() - end)

            # plot progress
            suffix =["Epoch: %d" % (epoch + 1), 
                     "Itr: %d/%d" % (batch_idx+1, len(loader)),
                     "%.2fh/epoch" % (time_meters.average_meters["batch_time"].avg / 3600 * len(loader))]
            suffix += ["%s: %.4f" % (k, v.avg) for k,v in avg_meters.average_meters.items()]
            suffix = " | ".join(suffix)
            bar(suffix)
            end = time.time()
        else:
            if args.vis is not None:
                if args.use_ho3d and (batch_idx + 1) * args.test_batch > 7693: # unseen object in HO3D training set
                    exit()
                if (batch_idx % args.vis_freq != 0):
                    continue
            '''try:
                img = sample["img"][-1].permute(1,2,0).numpy() * 255
                bbox_obj = sample["bbox_obj"]
                bbox_hand = sample["bbox_hand"]
                inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
                inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
                bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
                bbox_obj = bbox_obj[-1].numpy()
                bbox_inter = bbox_inter[-1].numpy()
                from PIL import Image
                img = Image.fromarray(img.astype(np.uint8))
                i_o = img.crop((bbox_obj[0], bbox_obj[1], bbox_obj[2], bbox_obj[3]))
                i_o = i_o.resize((256, 256), Image.NEAREST)
                i_inter = img.crop((bbox_inter[0], bbox_inter[1], bbox_inter[2], bbox_inter[3]))
                i_inter = i_inter.resize((256, 256), Image.NEAREST)
                plt.imshow(np.array(i_o))
                plt.axis("off")
                plt.savefig(os.path.join(save_path, "sample0.png"), bbox_inches="tight", pad_inches=0)
                plt.imshow(np.array(i_inter))
                plt.axis("off")
                plt.savefig(os.path.join(save_path, "sample1.png"), bbox_inches="tight", pad_inches=0)
                plt.show()
            except:
                print("lower less than upper")'''

            if use_cuda and torch.cuda.is_available():
                imgs = sample["img"].float().cuda()
                bbox_hand = sample["bbox_hand"].float().cuda()
                bbox_obj = sample["bbox_obj"].float().cuda()
                mano_params = None if args.use_ho3d else sample["mano_param"].float().cuda()
            else:
                imgs = sample["img"].float()
                bbox_hand = sample["bbox_hand"].float()
                bbox_obj = sample["bbox_obj"].float()
                mano_params = None if args.use_ho3d else sample["mano_param"].float()

            # measure data loading time
            time_meters.add_loss_value("data_time", time.time() - end)

            preds_joints, results, gt_mano_results, preds_obj = model(imgs, bbox_hand, bbox_obj,
                                                                      mano_params=mano_params,
                                                                      batch_idx=batch_idx)
                        
            # from torchviz import make_dot
            # g = make_dot(preds_joints)
            # g.render('espnet_model', view=False) 
            
            root_joint = np.expand_dims(sample["root_joint"].cpu().numpy(), axis=1)
            pred_xyz = results["joints3d"].detach().cpu().numpy() + root_joint
            pred_verts = results["verts3d"].detach().cpu().numpy() + root_joint
            bbox_hand = bbox_hand.cpu().numpy()
            bbox_obj = bbox_obj.cpu().numpy()

            for i in range(len(preds_obj)):
                preds_obj[i] = preds_obj[i].cpu().numpy()

            obj_pose = sample['obj_pose'].numpy()
            obj_bbox3d = sample['obj_bbox3d'].numpy()
            affinetrans = None
            if not args.use_ho3d:
                affinetrans = sample["affinetrans"].numpy()
                cam_intr = sample["K_no_trans"].numpy()
                sample["obj_cls"] = sample["obj_cls"].numpy()
                gt_xyz = gt_mano_results["joints3d"].cpu().numpy() + root_joint
                gt_verts = gt_mano_results["verts3d"].cpu().numpy() + root_joint
            else:
                cam_intr = sample["cam_intr"].numpy()

            obj_cls = sample['obj_cls']
            batch_hand_type = sample["hand_type"]
            obj_vertices_dict = loader.dataset.obj_vertices_dict
            obj_faces_dict = loader.dataset.obj_faces_dict

            # object predictions and evaluation(online)
            REP_res_dict, ADD_res_dict, pred_obj, gt_obj, p2d = eval_batch_obj(preds_obj, bbox_obj,
                                                        obj_pose, mesh_dict, obj_vertices_dict, obj_bbox3d, obj_cls,
                                                        cam_intr, REP_res_dict, ADD_res_dict, batch_affinetrans=affinetrans, batch_hand_type=batch_hand_type)
            # hand predictions and evaluation
            if args.use_ho3d:
                for xyz, verts in zip(pred_xyz, pred_verts):
                    if indices_order is not None:
                        xyz = xyz[indices_order]
                    xyz_pred_list.append(xyz * np.array([1, -1, -1]))
                    verts_pred_list.append(verts * np.array([1, -1, -1]))
            else:
                hand_eval_result = eval_hand(pred_xyz, sample["hand_type"], sample["joints_coord_cam"], hand_eval_result)

            #--------------------------------------- visualization ---------------------------------------#
            if args.vis is not None:
                sample["img"] = sample["img"].cpu().numpy()
                sample["affinetrans"] = sample["affinetrans"].numpy()
                sample["K_no_trans"] = sample["K_no_trans"].numpy()

                ii = (sample["img"][-1].transpose(1,2,0) * 255).astype(np.uint8)
                ii = np.ascontiguousarray(ii)
            if args.vis in ["3D", "3D_capture"]:
                # visualize last sample in batch
                #print(sample["hand_type"][-1])
                vis_3d(args, sample, obj_faces_dict, img=ii,
                       pred_hand=pred_verts[-1], #gt_hand = gt_verts[-1],
                       pred_obj=pred_obj, gt_obj=gt_obj)
            elif args.vis == "2D_kps":
                vv = vis_bbox(ii, bbox_obj[-1], thickness=1)
                vv = vis_keypoints(vv, p2d, radius=3, color=(255, 0, 0)).astype(np.uint8)
                save_dir = os.path.join(args.host_folder, "2D_kps")
                os.makedirs(save_dir, exist_ok=True)
                vv_out = Image.fromarray(vv)
                vv_out.save(os.path.join(args.host_folder, "2D_kps", sample["img_name"][-1] + ".png"))
                #plt.imshow(vv)
                #plt.show()
            elif args.vis == "3D_bbox":
                lines = [[0,2], [1,3], [2,3], [0,1], [4,6], [5,7], [6,7], [4,5], [2,6], [0,4], [3,7], [1,5]]
                vv = vis_keypoints_with_skeleton(ii, p2d, lines, radius=3, color=(0, 245, 255)).astype(np.uint8)
                save_dir = os.path.join(args.host_folder, "3D_bbox")
                os.makedirs(save_dir, exist_ok=True)
                vv_out = Image.fromarray(vv)
                vv_out.save(os.path.join(args.host_folder, "3D_bbox", sample["img_name"][-1] + ".png"))
                #plt.imshow(vv)
                #plt.show()
            elif args.vis == "2D_joints":
                j2d = preds_joints[-1][-1].cpu().numpy()
                bbox_hand = bbox_hand[-1]
                width, height = bbox_hand[2] - bbox_hand[0], bbox_hand[3] - bbox_hand[1]
                j2d[:, 0], j2d[:, 1] = j2d[:, 0] * width + bbox_hand[0], j2d[:, 1] * height + bbox_hand[1]
                lines = [[0,1], [1,2], [2,3], [3,4], [0,5], [5,6], [6,7], [7,8], [0,9], [9,10], [10,11], 
                         [11,12], [0,13], [13,14], [14,15], [15,16], [0,17], [17,18], [18,19], [19,20]]
                vv = vis_keypoints_with_skeleton(ii, j2d, lines, radius=3).astype(np.uint8)
                vv_out = Image.fromarray(vv)
                vv_out.save("2D_joints.png")
                plt.imshow(vv)
                plt.show()
            elif args.vis == "inputs":
                save_dir = os.path.join(args.host_folder, "inputs")
                os.makedirs(save_dir, exist_ok=True)
                ii_out = Image.fromarray(ii)
                ii_out.save(os.path.join(save_dir, sample["img_name"][-1] + ".jpg"))
            elif args.vis == "2D_bbox":
                inter_topleft = np.max((bbox_hand[-1, :2], bbox_obj[-1, :2]), axis=0)
                inter_bottomright = np.min((bbox_hand[-1, 2:], bbox_obj[-1, 2:]), axis=0)
                bbox_inter = np.concatenate([inter_topleft, inter_bottomright], axis=0)
                
                vv = vis_bbox(ii, bbox_hand[-1], thickness=2, color=(255,0,0))
                vv = vis_bbox(vv, bbox_obj[-1], thickness=2, color=(0,0,255))
                vv = vis_bbox(vv, bbox_inter, thickness=2, color=(0,255,0))
                save_dir = os.path.join(args.host_folder, "2D_bbox")
                os.makedirs(save_dir, exist_ok=True)
                ii_out = Image.fromarray(vv)
                ii_out.save(os.path.join(save_dir, sample["img_name"][-1] + ".jpg"))
            #--------------------------------------- visualization ---------------------------------------#

            # measure elapsed time
            time_meters.add_loss_value("batch_time", time.time() - end)

            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.2f}h"\
                .format(batch=batch_idx + 1, size=len(loader),
                        data=time_meters.average_meters["data_time"].val,
                        bt=time_meters.average_meters["batch_time"].avg / 3600 * len(loader))

            bar(suffix)
            end = time.time()

    if train:
        return avg_meters
    else:
        # object pose evaluation
        # if REP_res_dict is not None and ADD_res_dict is not None \
        #         and diameter_dict is not None and unseen_objects is not None:
        if REP_res_dict is not None and ADD_res_dict is not None \
                      and diameter_dict is not None:
           eval_object_pose(REP_res_dict, ADD_res_dict, diameter_dict, outpath=save_path, unseen_objects=unseen_objects,
                            epoch=epoch if epoch is not None else None)
        #hand evalution
        # if hand_eval_result is not None:
        #     eval_hand_pose_result(hand_eval_result, outpath=save_path, 
        #                     epoch=epoch+1 if epoch is not None else None)
        if args.use_ho3d:
            pred_out_path = os.path.join(save_path, "pred_epoch_{}.json".format(epoch) if epoch is not None else "pred_{}.json")
            dump(pred_out_path, xyz_pred_list, verts_pred_list)
            pred_zip_path = os.path.join(args.host_folder, "pred_epoch{}.zip".format(epoch) if epoch is not None else "pred{}.zip")
            cmd = "zip -j " + pred_zip_path + " " + pred_out_path
            print(cmd)
            os.system(cmd)
        elif hand_eval_result is not None:
            eval_hand_pose_result(hand_eval_result, outpath=save_path, 
                            epoch=epoch if epoch is not None else None)
        return None