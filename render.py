#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
from scene.cameras import Camera
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]
            
        if not args.include_feature:
            gt = view.original_image[0:3, :, :]
            
        else:
            gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)

        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def create_Cameras_from_path(path: str):
    all_cameras =[]

    with open(path, 'r') as f:
        data = json.load(f)
    # Extraktion und Berechnung der Parameter
    fovy_deg = data["default_fov"]
    fovy = np.radians(fovy_deg)  
    aspect_ratio = data["keyframes"][0]["aspect"]  
    height = data["render_height"]
    width = data["render_width"]
    fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)
    dummy_image = torch.ones(3, int(height), int(width), dtype=torch.float32)
    for idx, keyframe in enumerate(data["keyframes"]):
        transform = np.array(keyframe["matrix"]).reshape(4, 4)
        transform_t = np.transpose(transform)
        trans = transform[:3, 3]
        print(trans)
        R = transform_t[:3, :3]
        print(R)
        #T = transform
        cam = Camera(colmap_id=0, R=R, T=trans, FoVx=fovx, FoVy=fovy, image=dummy_image, gt_alpha_mask=None, image_name=f"{idx}", uid=None, trans = trans, scale=1, data_device="cuda")
        all_cameras.append(cam)
    
    return all_cameras

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print(scene.getTrainCameras()[0])
        cams = create_Cameras_from_path("F:\\Studium\\Master\\Thesis\\data\\unity_fuwa_small_ls\\camera_paths\\1_20_test.json")
        render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, cams, gaussians, pipeline, background, args)

        # if not skip_train:
        #      render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        # if not skip_test:
        #      render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)