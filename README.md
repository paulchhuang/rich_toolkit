# RICH: Real scenes, Interaction, Contacts and Humans
 <img src="docs/rich_visualization.gif" width="650"> 

This is the toolkit for RICH dataset in [Capturing and Inferring Dense Full-BodyHuman-Scene Contact](https://rich.is.tue.mpg.de/index.html). It consists of a few light weight scripts demonstrating how to use the released files, e.g., visualizing body keypoints on images, parsing meta data etc. 

The body-scene contact network (BSTRO) is released in another [repo.](https://github.com/paulchhuang/bstro)

## Install Dependencies
```
python3 -m venv PATH/2/VENV
source PATH/2/VENV/bin/activate
pip install -r requirements.txt
```

## Download necessary files
Please download the [RICH dataset](https://rich.is.tue.mpg.de/) and [SMPL-X model](https://smpl-x.is.tue.mpg.de/) from the official websites and organize them following the structure below:
```
${REPO_DIR}  
|-- body_models  
|   |-- smplx
|   |   |-- SMPLX_FEMALE.pkl
|   |   |-- SMPLX_FEMALE.npz
|   |   |-- SMPLX_MALE.pkl
|   |   |-- SMPLX_MALE.npz
|   |   |-- SMPLX_NEUTRAL.pkl
|   |   |-- SMPLX_NEUTRAL.npz
|   |   |-- ...
|-- data
|   |-- bodies
|   |   |-- train
|   |   |   |--BBQ_001_guitar
|   |   |   |--BBQ_001_juggle
|   |   |   |--...
|   |   |-- val
|   |   |-- test
|   |-- human_scene_contact
|   |   |-- train
|   |   |   |--BBQ_001_guitar
|   |   |   |--BBQ_001_juggle
|   |   |   |--...
|   |   |-- val
|   |   |-- test
|   |-- images
|   |   |-- train
|   |   |   |--BBQ_001_guitar
|   |   |   |--BBQ_001_juggle
|   |   |   |--...
|   |   |-- val
|   |   |-- test
|   |-- multicam2world
|   |   |-- BBQ_multicam2world.json
|   |   |-- Gym_multicam2world.json
|   |   |-- ... 
|   |   |-- ... 
|   |-- scan_calibration
|   |   |-- BBQ
|   |   |-- Gym
|   |   |-- ... 
|   |   |-- ... 
```

## Examples
1. Get 3D joints from SMPL-X params and project them onto an image:
    ```
    python smplx2images.py
    ```
    and check the results in `samples` folder.

2. Load human-scene contact annotations:
    ```
    python hsc_params.py
    ```
    and check the variables `hsc_vert_id_smpl` and `hsc_vert_id_smplx`.

3. Visualize scans and SMPL-X bodies in world frames:
    The released SMPL-X params and the scene scan reside in the calibrated multi-camera coordinate, where the first camera is conventionally chosen as the reference (R=I, t=0) so the ground plane is often not axis-aligned. When an axis-aligned ground plane is required, one can consider transforming bodies and the scene mesh to the world frame defined during the scanning process:
    ```
    python multicam2world.py
    ```
    Visualizing the generated `body_scene_world.ply` in `samples` folder with meshlab, one shall see:
     <img src="docs/rich_in_worldframe.png" width="650"> 

## Citations
If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{huang2022rich,
    title = {Capturing and Inferring Dense Full-Body Human-Scene Contact},
    author = {Huang, Chun-Hao P. and Yi, Hongwei and H{\"o}schle, Markus and Safroshkin, Matvey and Alexiadis, Tsvetelina and Polikovsky, Senya and Scharstein, Daniel and Black, Michael J.},
    booktitle = {IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR) },
    pages = {13274-13285},
    month = jun,
    year = {2022},
    month_numeric = {6}
}
```
