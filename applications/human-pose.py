# modified from https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/tasks/human_pose/live_demo.ipynb
import os
import time

import json
import numpy as np
from PIL import Image

import torch
import torchvision
import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import torch2trt
from torch2trt import TRTModule

from gst_app_src_and_sink import run_pipeline

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
WIDTH = 224
HEIGHT = 224

def main(args):

    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    if  not os.path.isfile(OPTIMIZED_MODEL):
        print("Optimizing the model with TensorRT")
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])

        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
        model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
        torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
        print("fTensorRT model saved at: {OPTIMIZED_MODEL}")

    print("Loading TensorRT model")
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

    device = torch.device("cuda")
    print(f"Running inference on device: {device}")


    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((HEIGHT,WIDTH)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)


    def user_callback(image_data):

        start_time = time.monotonic()

        tensor_image = preprocess(image_data)
        tensor_image = tensor_image.unsqueeze(0)
        cmap, paf = model_trt(tensor_image.to(device))
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()

        inference_time_ms = (time.monotonic() - start_time) * 1000

        counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        draw_objects(image_data, counts, objects, peaks)

        return image_data

    run_pipeline(
        user_callback,
        src_frame_rate=args.frame_rate,
        src_height=args.source_height,
        src_width=args.source_width,
        binning_level=args.binning_level,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="The model to load")

    parser.add_argument("--label_path", type=str, help="The labels to use")

    parser.add_argument("--source_width", type=int)

    parser.add_argument("--source_height", type=int)

    parser.add_argument("--frame_rate", type=int)

    parser.add_argument("--binning_level", type=int, default=2)

    main(parser.parse_args())
