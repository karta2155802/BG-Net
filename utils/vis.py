import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
import copy

from dataset import dataset_util

def vis_keypoints_with_skeleton(img, kps, kps_lines, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap("brg")
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    if color is not None:
        colors = [color for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[i1, 0].astype(np.int32), kps[i1, 1].astype(np.int32)
        p2 = kps[i2, 0].astype(np.int32), kps[i2, 1].astype(np.int32)
        cv2.line(
            kp_mask, p1, p2, 
            color=colors[l], thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p1, 
            radius=radius, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p2, 
            radius=radius, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_bbox(img, bbox, alpha=1, thickness=1, color=(255,255,255)):
    bbox = bbox.astype(np.int32)
    upper_left = bbox[0], bbox[1]
    upper_right = bbox[2], bbox[1]
    lower_left = bbox[0], bbox[3]
    lower_right = bbox[2], bbox[3]
    kp_mask = np.copy(img)
    cv2.line(kp_mask, upper_left, upper_right, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, upper_left, lower_left, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, upper_right, lower_right, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, lower_left, lower_right, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1, radius=3, color_base=255, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps))]
    colors = [(c[2] * color_base, c[1] * color_base, c[0] * color_base) for c in colors]
    if color is not None:
        colors = [color for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d(args, sample, obj_faces_dict, img=None, pred_hand=None, gt_hand=None, pred_obj=None, gt_obj=None):
    img_size = args.img_size
    width, height = args.inp_res, args.inp_res
    obj_faces = obj_faces_dict[sample["obj_cls"][-1]]
    if sample["hand_type"][-1] == "left":
        pred_obj[:,0] *= -1
        gt_obj[:,0] *= -1
        obj_faces = np.flip(obj_faces, axis=1) # flip vertex position in 3D, so normal order has to change

    sealed_faces = np.load(os.path.join(args.mano_root, "../sealed_faces.npy"), allow_pickle=True).item()
    hand_faces = sealed_faces["sealed_faces_right"]
    pred_hand = add_seal_vertex(pred_hand)
        
    img_name = sample["img_name"][-1]
    affinetrans = sample["affinetrans"][-1]
    K = sample["K_no_trans"][-1]
    save_path = os.path.join(args.host_folder, "3D")

    
    mesh_list1 = []
    mesh_list2 = []

    verts = np.concatenate([pred_hand, pred_obj], axis=0)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2

    if pred_hand is not None:
        hand_mesh = generate_triangle_mesh(pred_hand, hand_faces, [0.7, 0.7, 0.7]) # gray
        mesh_list1.append(hand_mesh)

        hand_mesh2 = copy.deepcopy(hand_mesh)
        R = hand_mesh2.get_rotation_matrix_from_xyz((np.pi, 0, np.pi))
        hand_mesh2.rotate(R, center=center)
        mesh_list2.append(hand_mesh2)
    
    if gt_hand is not None:
        gt_hand = add_seal_vertex(gt_hand)
        gt_hand_mesh = generate_triangle_mesh(gt_hand, hand_faces, [1, 0.8, 0]) # yellow
        #mesh_list1.append(gt_hand_mesh)

    vertex_color = None
    if args.no_obj_error:
        vertex_color = set_vertex_color_by_error(pred_obj, gt_obj, img_name)

    if pred_obj is not None:
        obj_mesh = generate_triangle_mesh(pred_obj, obj_faces, uniform_color=[1, 0.7, 0.7], vertex_color=vertex_color) # red
        mesh_list1.append(obj_mesh)

        obj_mesh2 = copy.deepcopy(obj_mesh)
        R = obj_mesh2.get_rotation_matrix_from_xyz((np.pi, 0, np.pi))
        obj_mesh2.rotate(R, center=center)
        mesh_list2.append(obj_mesh2)

    if gt_obj is not None:
        gt_obj_mesh = generate_triangle_mesh(gt_obj, obj_faces, [0.2, 0.8, 0.2]) # green
        #mesh_list1.append(gt_obj_mesh)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3) # draw coordinate frame
    #mesh_list1.append(coord)

    render1 = renderer(img_name, img_size, K, mesh_list1)

    if args.vis == "3D":
        #o3d.visualization.draw_geometries(mesh_list1, width=720, height=720)
        render1.run()
    if args.vis == "3D_capture":
        # for screenshot
        view1 = render1.capture_screen_float_buffer()
        view1 = crop_img(view1, width, height, affinetrans)
        render1.destroy_window()

        render2 = renderer(img_name, img_size, K, mesh_list2)
        view2 = render2.capture_screen_float_buffer()
        view2 = crop_img(view2, width, height, affinetrans)
        render2.destroy_window()

        # render test on ori_img on Dex_YCB
        #ori_img = (sample["ori_img"][-1].transpose(1,2,0) * 255).astype(np.uint8)
        #ori_out = cv2.addWeighted(ori_img, 0.5, screenshot, 0.5, 0)
        
        '''if projection:
            depth = render1.capture_depth_float_buffer()
            depth = np.asarray(depth)
            mask = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0)).astype(np.uint8)
            mask = Image.fromarray(mask)
            mask = dataset_util.transform_img(mask, affinetrans, [width, height])
            mask = mask.crop((0, 0, width, height))
            mask = np.asarray(mask).astype(np.bool8)[:, :, None]
            mix = screenshot * mask + img * ~mask
            output.append(mix)'''
        if args.save_vis:
            save_figure([view1, view2], save_path, img_name)
        else:
            show_figure([img, view1, view2], img_name)


def train_vis_3d(args, sample, obj_faces_dict, gt_hand=None, gt_obj=None):
    obj_faces = obj_faces_dict[sample["obj_cls"][-1]]
    coord_change_mat = np.array([1, -1, -1])
    
    from utils.manolayer_ho3d import ManoLayer
    mano_layer = ManoLayer(ncomps=45, center_idx=0, flat_hand_mean=False,
                        side="right", mano_root=args.mano_root, use_pca=False)
    hand_faces = mano_layer.th_faces.numpy()
    mesh_list = []

    if gt_hand is not None:
        gt_hand_mesh = generate_triangle_mesh(gt_hand, hand_faces, [1, 0.8, 0], coord_change_mat=coord_change_mat) # yellow
        mesh_list.append(gt_hand_mesh)

    if gt_obj is not None:
        gt_obj_mesh = generate_triangle_mesh(gt_obj, obj_faces, [0.2, 0.8, 0.2], coord_change_mat=coord_change_mat) # green
        mesh_list.append(gt_obj_mesh)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3) # draw coordinate frame
    #mesh_list.append(coord)
    o3d.visualization.draw_geometries(mesh_list, width=720, height=720)


def renderer(img_name, img_size, K, mesh_list):
    render = o3d.visualization.Visualizer()
    render.create_window(window_name=img_name, width=img_size[0], height=img_size[1])
    for mesh in mesh_list:
        render.add_geometry(mesh)
    
    view_ctl = render.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(img_size[0], img_size[1], K)
    extrinsic = np.eye(4)
    cam.extrinsic = extrinsic
    cam.intrinsic = intrinsic
    view_ctl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

    render.poll_events()
    render.update_renderer()
    return render


def crop_img(img, width, height, affinetrans):
    img = (255.0 * np.asarray(img)).astype(np.uint8)
    img = Image.fromarray(img)
    img = dataset_util.transform_img(img, affinetrans, [width, height])
    img = img.crop((0, 0, width, height))
    img = np.asarray(img)
    return img


def set_vertex_color_by_error(pred, gt, img_name):
    '''
    pred, gt: [N, 3]
    '''
    error = np.linalg.norm((pred - gt), axis=1) # in meter
    if (error > 1e4).any():
        print(img_name)
        vertex_color_idx = (np.ones_like(error) * 100).astype(np.int32)
    else:
        vertex_color_idx = np.round(error * 1e3).astype(np.int32)
        vertex_color_idx[np.where(vertex_color_idx > 100)] = 100
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, 101)]
    vertex_color = np.asarray([colors[i] for i in vertex_color_idx])[:, :3]
    return vertex_color


def generate_triangle_mesh(verts, faces, uniform_color=None, vertex_color=None, coord_change_mat=np.array([1, 1, 1])):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts * coord_change_mat) # change coord
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    if vertex_color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_color)
    else:
        mesh.paint_uniform_color(uniform_color)    
    return mesh


def add_seal_vertex(vertex):
    circle_v_id = np.array(
        [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
        dtype=np.int32,
    )
    center = (vertex[circle_v_id, :]).mean(0)
    vertex = np.vstack([vertex, center])
    return vertex


def show_figure(img_list, img_name="sample", cmap="viridis"):
    cols = len(img_list)
    f = plt.figure(img_name, figsize=(3*cols, 3))
    for i in range(cols):
        f.add_subplot(1, cols, i+1)
        plt.imshow(img_list[i], cmap=cmap)
        plt.axis("off")
    plt.subplots_adjust(wspace=0)
    plt.show()


def save_figure(img_list, save_path="exp-results", img_name="sample", cmap="viridis"):
    cols = len(img_list)
    os.makedirs(save_path, exist_ok=True)
    for i in range(cols):
        plt.imshow(img_list[i], cmap=cmap)
        plt.axis("off")
        plt.savefig(os.path.join(save_path, img_name + "_" + str(i) +".png"), bbox_inches="tight", pad_inches=0)
        plt.close()


def save_combined_figure(img_list, save_path="exp-results", img_name="sample", cmap="viridis"):
    cols = len(img_list)
    os.makedirs(save_path, exist_ok=True)
    f = plt.figure(img_name, figsize=(3*cols, 3))
    for i in range(cols):
        f.add_subplot(1, cols, i+1)
        plt.imshow(img_list[i], cmap=cmap)
        plt.axis("off")
    
    plt.subplots_adjust(wspace=0)
    plt.savefig(os.path.join(save_path, img_name +".png"), bbox_inches="tight", pad_inches=0)
    plt.close()    