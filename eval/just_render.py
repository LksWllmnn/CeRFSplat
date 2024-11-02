#!/usr/bin/env python
from __future__ import annotations

import os
import glob
import random
from pathlib import Path
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import colormap_saving

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

def activate_stream(sem_map, image, clip_model, image_name: Path = None, thresh: float = 0.5, colormap_options=None):
    # Aktivierungskarten auf maximalen Wert trimmen
    valid_map = clip_model.get_max_across(sem_map) 
    n_head, n_prompt, h, w = valid_map.shape
    #logger.info("hello world act")
    for k in range(n_prompt): 
        for i in range(n_head):
            scale = 30
            kernel = np.ones((scale, scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

            # Farbkarte und kombinierte Visualisierung speichern
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            
            # Hintergrund maskieren, falls Wert unter Threshold
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            colormap_saving(valid_composited, colormap_options, output_path_compo)
    return 

def render_image():
    logger = get_logger("test_logger")
    ae_ckpt_path = "../autoencoder/ckpt/unity_fuwa_small_2/best_ckpt.pth"
    encoder_hidden_dims=[256, 128, 64, 32, 3]
    decoder_hidden_dims=[16, 32, 64, 128, 256, 256, 512]
    mask_thresh=0.4
    npy_path = "../output/unity_fuwa_small_2_1/train/ours_None/gt_npy/00000.npy"
    image_path = "../data/unity_fuwa_small_2/images/Fuwa_001_0022.jpg"

    output_path = Path("../test_renders_lw")
    output_path.mkdir(exist_ok=True, parents=True)  # Ordner wird erstellt, falls er nicht existiert

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(colormap="turbo", normalize=True, colormap_min=-1.0, colormap_max=1.0)
    
    # Autoencoder und CLIP initialisieren
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # CLIP Prompt auf "cubes" setzen
    clip_model.set_positives(["building"])
    # Überprüfe, ob ein Bildpfad existiert und lade das Bild
    rgb_img = cv2.imread(image_path)[..., ::-1]
    rgb_img = (rgb_img / 255.0).astype(np.float32)
    rgb_img = torch.from_numpy(rgb_img).to(device)
    
    # Lade Feature und Bild
    compressed_sem_feats = np.zeros((1, 1, 1080, 1920, 3))
    compressed_sem_feats[0][0] = np.load(npy_path)
    logger.info(compressed_sem_feats.shape)
    sem_feat = compressed_sem_feats[:, 0, ...]
    sem_feat = torch.from_numpy(sem_feat).float().to(device)

    #Feature dekodieren
    with torch.no_grad():
        lvl, h, w, _ = sem_feat.shape
        logger.info(sem_feat.shape)
        restored_feat = model.decode(sem_feat.flatten(0, 1))
        restored_feat = restored_feat.view(lvl, h, w, -1)

    # Aktivierungs-Stream für das aktuelle Bild
    activate_stream(
        restored_feat,
        rgb_img,
        clip_model,
        Path("test"),
        thresh=mask_thresh,
        colormap_options=colormap_options
    )

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    render_image()
