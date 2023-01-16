import os
import json
import torch
import pickle
import trimesh
from smplx import SMPLX
import numpy as np

## input arguments
SET = 'train'
SEQ_NAME = 'ParkingLot1_005_pushup2'
SCENE_NAME, SUB_ID, _ = SEQ_NAME.split('_')
FRAME_ID = 150
CAMERA_ID = 0
gender_mapping = json.load(open('resource/gender.json','r'))
GENDER = gender_mapping[f'{int(SUB_ID)}']
imgext = json.load(open('resource/imgext.json','r'))
EXT = imgext[SCENE_NAME]

## SMPLX model
SMPLX_MODEL_DIR = 'body_models/smplx'
body_model = SMPLX(
            SMPLX_MODEL_DIR,
            gender=GENDER,
            num_pca_comps=12,
            flat_hand_mean=False,
            create_expression=True,
            create_jaw_pose=True,
        )

## passing the parameters through SMPL-X
smplx_params_fn = os.path.join('data/bodies',SET, SEQ_NAME, f'{FRAME_ID:05d}', f'{SUB_ID}.pkl')
body_params = pickle.load(open(smplx_params_fn,'rb'))
body_params = {k: torch.from_numpy(v) for k, v in body_params.items()}
body_model.reset_params(**body_params)
model_output = body_model(return_verts=True,   
                        body_pose=body_params['body_pose'],
                        return_full_pose=True) 
mesh = trimesh.Trimesh(vertices = model_output.vertices.detach().cpu().squeeze().numpy(), faces = body_model.faces, process=False)

with open(f'data/multicam2world/{SCENE_NAME}_multicam2world.json', 'r') as f:
    cam2scan = json.load(f)
rot_mat = np.array(cam2scan['R'])
translation = cam2scan['t']


## scan
scene_scan = trimesh.load(f'data/scan_calibration/{SCENE_NAME}/scan_camcoord.ply', process=False)
scan = scene_scan + mesh
print(scan.vertices.shape)
scan.vertices = cam2scan['c'] * scan.vertices @ rot_mat + translation
scan.export(f'samples/body_scene_world.ply')

