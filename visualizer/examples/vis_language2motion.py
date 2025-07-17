import os
import re

import cv2
import json
import joblib
import numpy as np
import pandas as pd

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import WeakPerspectiveCamera, OpenCVCamera
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer
from argparse import ArgumentParser

# Set to True for rendering in headless mode, no window will be created and
# a video will be exported to 'headless/test.mp4' in the export directory
HEADLESS = False
text_dict = {}
csv_file_path = '../../datasets/language2motion/text/how2sign_realigned_train.csv'
text_all = pd.read_csv(csv_file_path, 
        sep='\t', 
        names=["VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME", "START_REALIGNED","END_REALIGNED","SENTENCE"])
sentence_name_all=np.array(text_all["SENTENCE_NAME"])
text_all=np.array(text_all["SENTENCE"])
for idx, i in enumerate(sentence_name_all):
    text_dict[i] = text_all[idx]

import numpy as np

def quat2mat(quat):

    norm_quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rot_mat = np.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wx + 2*yz,
                        2*xy + 2*wz, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                        2*wz - 2*xy, 2*wx - 2*yz, w2 - x2 - y2 + z2], axis=1).reshape(-1, 3, 3)

    return rot_mat

def batch_rodrigues(param):

    l1norm = np.linalg.norm(param + 1e-8, axis=1)
    angle = np.expand_dims(l1norm, -1)
    normalized = np.divide(param, angle)
    angle = angle * 0.5

    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    quat = np.concatenate([v_cos, v_sin * normalized], axis=1)

    return quat2mat(quat)

def get_img_list(video_path):
    cap = cv2.VideoCapture(video_path)
    img_list = []
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.flipud(img)
        img_list.append(img)

    img_list = np.array(img_list)
    return img_list


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--pkl_file_path',
        type=str,
        default='../../datasets/language2motion/annotations/how2sign_pkls_cropTrue_shapeTrue/')
    parser.add_argument(
        '--video_folder',
        default='../../datasets/language2motion/videos/',
        type=str)
    parser.add_argument(
        '--video_id',
        default='11ysPjNRN4Y_15-1-rgb_front',
        type=str)

    args = parser.parse_args()

    # Load camera and SMPL-X data from our pkl path.
    video_id = args.video_id
    data = joblib.load(open(os.path.join(args.pkl_file_path, args.video_id+'.pkl'), "rb"))
    text = text_dict[video_id]

    all_pose = data['smplx']
    g, b, l, r, j, s, exp, cam_trans = \
    all_pose[:, :3], all_pose[:, 3:66], all_pose[:, 66:111], all_pose[:, 111:156], all_pose[:, 156:159], \
    all_pose[:, 159:169], all_pose[:, 169:179], all_pose[:, 179:182]

    # Create the viewer, set a size that has 16:9 aspect ratio to match the input data
    if HEADLESS:
        viewer = HeadlessRenderer(size=(1600, 900))
    else:
        viewer = Viewer(size=(1600, 900))

    smpl_layer = SMPLLayer(model_type="smplx", gender="neutral", flat_hand_mean=False, device=C.device)
    smpl_sequence = SMPLSequence(
        poses_body=b,
        poses_root=g,
        poses_left_hand=l,
        poses_right_hand=r,
        betas=s,
        smpl_layer=smpl_layer,
        color=(0.8, 0.72, 0.425, 1),
        # color=(1, 1, 1, 1),
        rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi),
        text=text
    )
    viewer.scene.add(smpl_sequence)

    # load imgs
    video_path = os.path.join(args.video_folder, video_id+'.mp4')
    if os.path.exists(video_path):
    
        input_imgs_list = get_img_list(video_path)
        cols, rows = input_imgs_list[0].shape[1], input_imgs_list[0].shape[0]

        # Size in pixels of the image data.
        cols, rows = 1920, 1080

        # Create a sequence of weak perspective cameras.
        fov = 60
        f = max(cols, rows) / 2.0 * 1.0 / np.tan(np.radians(fov / 2))
        cam_intrinsics = np.array([[f, 0.0, cols / 2], [0.0, f, rows / 2], [0.0, 0.0, 1.0]])

        cam_extrinsics = np.eye(4)
        cam_extrinsics[:3, 3] = 0,0,0
        cameras = OpenCVCamera(cam_intrinsics, cam_extrinsics[:3], cols, rows, viewer=viewer)

        # Path to the directory containing the video frames.
        # images_path = "resources/vibe/frames"
        pc = Billboard.from_camera_and_distance(cameras, 4.0, cols, rows, input_imgs_list)
        pc.position = np.array([0.71, 0.85, -0.26])
        pc.scale = 0.172

        pc.rotation = batch_rodrigues(np.array([[-6.54 / 180 * np.pi, 0.00, 0.00]]))

        # Create a billboard.
        # Add all the objects to the scene.
        viewer.scene.add(pc)

    # Viewer settings.
    viewer.scene.light_mode = 'dark'
    viewer.auto_set_camera_target = False
    viewer.scene.camera.position = np.array([0, 0.865, 1.293])
    viewer.scene.camera.target = np.array([0, 0.725, 0.047])
    viewer.auto_set_floor = False
    viewer.playback_fps = 25
    viewer.scene.fps = 25
    viewer.scene.floor.position = np.array([0, -0.29, 0])
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = True

    if HEADLESS:
        viewer.save_video(video_dir=os.path.join(C.export_dir, "headless/render.mp4"), output_fps=25)
    else:
        viewer.run()
