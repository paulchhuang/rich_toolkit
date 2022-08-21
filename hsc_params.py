import pickle
import numpy as np

## input arguments
SET = 'val'
SEQ_NAME = 'LectureHall_003_wipingchairs1'
SCENE_NAME, SUB_ID, _ = SEQ_NAME.split('_')
FRAME_ID = 17

hsc_params = pickle.load(open(f'data/human_scene_contact/{SET}/{SEQ_NAME}/{FRAME_ID:05d}/{SUB_ID}.pkl', 'rb'))

hsc_vert_id_smpl = np.where(hsc_params['contact'] > 0.)[0]          # (6890,): per-vertex 0/1 contact label in smpl format
distplacement_vec_smplx = hsc_params['s2m_dist_id']                 # (10475,3): the vector that points from each vertex in smplx to the closest point on the scene scan
closest_faces_on_scene_smplx = hsc_params['closest_triangles_id']   # (10475,3,3): the triangle which stores the point above

print(1)