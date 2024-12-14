import pyrender
import numpy as np
import os
import pickle
import cv2
from tqdm import tqdm
from common.utils.human_models import smpl_x
import torch
import copy
import trimesh
from argparse import ArgumentParser
import urllib.request
import csv
import json
import pandas as pd


os.environ['DISPLAY'] = ':0.0'
os.environ["PYOPENGL_PLATFORM"] = "egl"
torch.manual_seed(3407)

print('initialization...')
text_dict = {}
with open('../datasets/hamnosys2motion/data.json', 'rb') as f:
    hamnosys_text_list = json.load(f)
for i in hamnosys_text_list.keys():
    text_dict[i] = hamnosys_text_list[i]['hamnosys_text']

csv_file_path = '../datasets/language2motion/text/how2sign_realigned_train.csv'
text_all = pd.read_csv(csv_file_path, 
        sep='\t', 
        names=["VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME", "START_REALIGNED","END_REALIGNED","SENTENCE"])
sentence_name_all=np.array(text_all["SENTENCE_NAME"])
text_all=np.array(text_all["SENTENCE"])
for idx, i in enumerate(sentence_name_all):
    text_dict[i] = text_all[idx]

# default params
predefined_height, predefined_width = 720, 1280 # from pjm_541
pred_focals =  [14921.82254791, 14921.82254791] # from pjm_541
pred_princpts = [620.60418701, 413.40108109] # from pjm_541
input_body_shape = (256, 192)
output_hm_shape = (16, 16, 12)
focal = (5000, 5000)  # virtual focal lengths
princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)  # virtual principal point position
smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
background = cv2.imread('../assets/blender.png')
org = (10, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)
thickness = 2
DEFAULT_DTYPE = torch.float32
print('end of init...')

def put_text_with_newline(image, text, org, font, font_scale, color, thickness, line_type=cv2.LINE_AA):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    img_height, img_width, _ = image.shape
    x, y = org

    lines = []
    line = ""

    for word in text.split(" "):
        word_size, _ = cv2.getTextSize(line + " " + word, font, font_scale, thickness)
        word_width, _ = word_size

        if x + word_width > img_width / 3:
            lines.append(line)
            line = word
        else:
            if line:
                line += " "
            line += word

    lines.append(line)

    for i, line in enumerate(lines):
        y_offset = i * text_height*1.2
        cv2.putText(image, line, (x, int(y + y_offset)), font, font_scale, color, thickness, line_type)

def get_img_list(folder_name, from_url=False):

    if not from_url:
        video = cv2.VideoCapture(os.path.join(args.video_path, folder_name+'.mp4'))
    else:
        print('current version does not support generate online. Please download the videos.')
        raise NotImplementedError

    if not video.isOpened():
        print(f"Error: Could not open video {folder_name}.")
        exit()

    frames = []
    while True:

        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    video.release()
    return frames

def get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode='test', zero_global=False, mesh=False):
    batch_size = root_pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
    if not zero_global:
        output = smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                    left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                    reye_pose=zero_pose, expression=expr)
    else:
        raise ValueError
        output = smplx_layer(betas=shape, body_pose=body_pose, global_orient=zero_pose, right_hand_pose=rhand_pose,
                            left_hand_pose=lhand_pose, jaw_pose=zero_pose, leye_pose=zero_pose,
                            reye_pose=zero_pose, expression=expr)

    # camera-centered 3D coordinate
    mesh_cam = output.vertices
    joint_cam = output.joints[:, smpl_x.joint_idx, :]

    # project 3D coordinates to 2D space
    x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
        focal[0] + princpt[0]
    y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
        focal[1] + princpt[1]
    x = x / input_body_shape[1] * output_hm_shape[2]
    y = y / input_body_shape[0] * output_hm_shape[1]
    joint_proj = torch.stack((x, y), 2)

    # mesh
    if mesh:
        tx = (mesh_cam[:, :, 0] + cam_trans[:, None, 0]) / (mesh_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
            focal[0] + princpt[0]
        ty = (mesh_cam[:, :, 1] + cam_trans[:, None, 1]) / (mesh_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
            focal[1] + princpt[1]
        tx = tx / input_body_shape[1] * output_hm_shape[2]
        ty = ty / input_body_shape[0] * output_hm_shape[1]
        mesh_proj = torch.stack((tx, ty), 2)
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        render_mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering

        return render_mesh_cam
    else:
        return joint_proj

def render(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
	np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.125, roughnessFactor=0.6, alphaMode='OPAQUE', baseColorFactor=(0.425, 0.72, 0.8, 1))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # # light
    light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    spot_l = pyrender.SpotLight(color=np.ones(3),
                                intensity=15.0,
                                innerConeAngle=np.pi / 3,
                                outerConeAngle=np.pi / 2)

    light_pose[:3, 3] = [1, 2, 2]
    scene.add(spot_l, pose=light_pose)

    light_pose[:3, 3] = [-1, 2, 2]
    scene.add(spot_l, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    return img

def process_single_video(file_name, result_path):
    video_name = os.path.basename(file_name).split('.pkl')[0]

    with open(file_name, 'rb') as f:
        results_dict = pickle.load(f)

    focals = results_dict['focal']
    princpts = results_dict['princpt']
    all_pose = results_dict['smplx']
    height, width = results_dict['height'], results_dict['width']
    all_pose = torch.tensor(all_pose).cuda()

    g, b, l, r, j, s, exp, cam_trans = \
    all_pose[:, :3], all_pose[:, 3:66], all_pose[:, 66:111], all_pose[:, 111:156], all_pose[:, 156:159], \
    all_pose[:, 159:169], all_pose[:, 169:179], all_pose[:, 179:182]
    assert len(all_pose[0]) == 182

    meshes = get_coord(g, b, l, r, j, s, exp, cam_trans[0][None], mesh=True).cpu().numpy()

    if os.path.isdir(args.pkl_file_path):
        bar = enumerate(results_dict['total_valid_index'])
    else:
        bar = enumerate(tqdm(results_dict['total_valid_index']))

    if args.overlay:
        if args.video_path is None:
            raw_img_list = get_img_list(video_name, from_url=True)
        else:
            raw_img_list = get_img_list(video_name)

    img_list = []
    text = text_dict[video_name.replace('/', '')]
    for idx, index in bar:

        if args.overlay:
            # render overlay
            raw_img = raw_img_list[index]
            img = render(raw_img.copy(), meshes[idx], smpl_x.face, {'focal': focals[0], 'princpt': princpts[0]})
            img = np.array(np.concatenate((raw_img,img), axis=1), dtype=np.uint8)
            size = (2*width,height)
        else:
            # render with background
            img = render(background, meshes[idx], smpl_x.face, {'focal': pred_focals, 'princpt': pred_princpts}).astype(np.uint8)
            size = (predefined_width,predefined_height)
            img_list.append(img)

        put_text_with_newline(img, text, org, font, font_scale, color, thickness)

    out = cv2.VideoWriter(result_path + f'/{video_name}.mp4', 0x7634706d, 24, size)
    for idx in range(len(img_list)):
        out.write(img_list[idx])
    out.release()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--pkl_file_path',
        type=str,
        required=True)
    parser.add_argument(
        '--video_path',
        default=None,
        type=str)
    parser.add_argument(
        '--overlay',
        help='whether to render overaly',
        action='store_true'
    )

    args = parser.parse_args()

    if args.overlay:
        result_path = './render_results_overlay/'
    else:
        result_path = './render_results/'

    os.makedirs(result_path, exist_ok=True)
    if os.path.isdir(args.pkl_file_path):
        for file_name in tqdm(os.listdir(args.pkl_file_path)):
            process_single_video(os.path.join(args.pkl_file_path, file_name), result_path)
    else:
        print(f'processing {args.pkl_file_path}')
        process_single_video(args.pkl_file_path, result_path)
