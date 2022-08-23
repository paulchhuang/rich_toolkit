import pickle
import numpy as np

## input arguments
SET = 'train'
SEQ_NAME = 'ParkingLot1_005_pushup2'
SCENE_NAME, SUB_ID, _ = SEQ_NAME.split('_')
FRAME_ID = 150

hsc_params = pickle.load(open(f'data/human_scene_contact/{SET}/{SEQ_NAME}/{FRAME_ID:05d}/{SUB_ID}.pkl', 'rb'))

contact_labels = hsc_params['contact']                              # (6890,): per-vertex 0/1 contact label in smpl format
hsc_vert_id_smpl = np.where(contact_labels > 0.)[0]          
distplacement_vec_smplx = hsc_params['s2m_dist_id']                 # (10475,3): the vector that points from each vertex in smplx to the closest point on the scene scan
closest_faces_on_scene_smplx = hsc_params['closest_triangles_id']   # (10475,3,3): the triangle which stores the point above

import trimesh
smplx_mesh = trimesh.load(f'data/human_scene_contact/{SET}/{SEQ_NAME}/{FRAME_ID:05d}/{SUB_ID}.obj', process=False)
hsc_vert_id_smplx = np.where((smplx_mesh.visual.vertex_colors == (0, 255, 0, 255)).all(axis=1))[0]
