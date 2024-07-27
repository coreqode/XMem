import numpy as np
import h5py
from tqdm import tqdm, trange
from PIL import Image
import cv2
import os
from os import path
import argparse
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import params as param_utils

## Load XMeM dependency
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis


def save_hdf5_data(h5_path, cam_name, data): 
    logger.info(f"Appending data to hdf5 file at {h5_path}")

    total_frames = data.shape[0]

    with h5py.File(h5_path, "a", libver='latest') as file:
        if not 'frames' in file.keys(): 
            frames_group = file.create_group("frames")
        else: 
            frames_group = file.get('frames')

        if not cam_name in frames_group: 
            frames_group.create_dataset(cam_name, dtype = np.int32, data = data, compression = 'lzf')
        else: 
            del frames_group[cam_name]
            frames_group.create_dataset(cam_name, dtype = np.int32, data = data, compression = 'lzf')

def xmem_seg(video_path, mask_path, start_frame_idx, cam): 

    torch.set_grad_enabled(False)

    # default configuration
    config = {
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': True,
        'num_prototypes': 128,
        'min_mid_term_frames': 5,
        'max_mid_term_frames': 10,
        'max_long_term_elements': 10000,
    }

    pretrained_weight_path = './.cache/weights/XMem.pth'

    if not os.path.exists(pretrained_weight_path): 
        os.system('wget -P ./.cache/weights https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth')
    
    device = torch.device('cuda')
    network = XMem(config, pretrained_weight_path).eval().to(device)
    torch.cuda.empty_cache()

    mask = np.array(Image.open(mask_path))
    num_objects = len(np.unique(mask)) - 1

    processor = InferenceCore(network, config=config)
    processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
    cap = cv2.VideoCapture(video_path)

    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(cv2.VideoCapture.get(cap, property_id))

    # You can change these two numbers
    end_frame_idx = 10000
    visualize_every = 20

    current_frame_index = 0

    prediction_list = []
    with torch.cuda.amp.autocast(enabled=True):
        while (cap.isOpened()):
            # load frame-by-frame
            _, frame = cap.read()
            if frame is None or current_frame_index > end_frame_idx:
                break

            if current_frame_index < start_frame_idx: 
                current_frame_index+=1
                continue
                
            ## undistort the frame
            if cam is not None: 
                extr = param_utils.get_extr(cam)
                K, dist = param_utils.get_intr(cam)
                new_K, roi = param_utils.get_undistort_params(K, dist, (frame.shape[1], frame.shape[0]))
                frame = param_utils.undistort_image(K, new_K, dist, frame)

            # convert numpy array to pytorch tensor format
            frame_torch, _ = image_to_torch(frame, device=device)
            if current_frame_index == start_frame_idx:
                # initialize with the mask
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
                # the background mask is not fed into the model
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                # propagate only
                prediction = processor.step(frame_torch)

            # argmax, convert to numpy
            prediction = torch_prob_to_numpy_mask(prediction)

            # if current_frame_index % visualize_every == 0:
            #     visualization = Image.fromarray(overlay_davis(frame, prediction))
            #     visualization.save(f'.cache/{current_frame_index}.png')

            current_frame_index += 1
            prediction_list.append(prediction)

    prediction_list = np.stack(prediction_list)
    return prediction_list


def parser(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, help="Input file path", required=True)
    parser.add_argument("--mask_path", type=str, help="Output file path", required=True)
    parser.add_argument("--cam_path", type=str, help="cam file path", required=True)
    parser.add_argument("--start_frame_idx", type=int, required=True)
    parser.add_argument("--undistort", type=bool, required=True)

    args = parser.parse_args()
    return args

def main(): 
    args = parser()
    cam_name = args.mask_path.split('/')[-1].split('.')[0]
    seq_name = args.mask_path.split('/')[-3]

    if args.undistort: 
        assert (args.cam_path is not None)
        cameras = param_utils.read_params(args.cam_path)

        ## get cam for this particular camera index
        idx = np.where(cameras[:]['cam_name']==cam_name)[0][0]
        cam = cameras[idx]
        assert (cam['cam_name'] == cam_name)

        logger.info("Undistorting active for sequence!!")
    else: 
        cam = None

    out_list = xmem_seg(args.video_path, args.mask_path, args.start_frame_idx, cam)
    h5_path = os.path.join(os.getcwd(), f".cache/{seq_name}/annot.hdf5")
    save_hdf5_data(h5_path, cam_name, out_list)
    logger.info(f"Mask generated for {cam_name} and {seq_name}")

if __name__ == '__main__': 
    main()

