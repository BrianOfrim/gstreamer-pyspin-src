import fnmatch
import os
import time

import numpy as np
import svgwrite
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

from gst_overlay_pipeline import run_pipeline


def find_file(name, path) -> str:
    for root, dirs, files in os.walk(path):
        filematch = next((f for f in files if fnmatch.fnmatch(f, name)), None)
        if filematch:
            return os.path.join(root, filematch)
    return None


def get_model(num_classes, **kwargs):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, **kwargs
    )
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def draw_text(dwg, x, y, text, font_size=25):
    dwg.add(
        dwg.text(text, insert=(x, max(y, font_size)), fill="white", font_size=font_size)
    )


def draw_rect(dwg, x, y, w, h, stroke_color="red", stroke_width=4):
    dwg.add(
        dwg.rect(
            insert=(x, y),
            size=(w, h),
            fill="none",
            stroke=stroke_color,
            stroke_width=stroke_width,
        )
    )


def main(args):

    model_path = args.model_path

    if model_path is not None:
        if not os.path.isfile(model_path):
            print(f"Error: Could not locate model file: {model_path}")
            return
    else:
        model_path = find_file("*.pt", os.getcwd())
        if model_path is None:
            print(f"Error: Could not locate a model '.pt' file in {os.getcwd()}")
            return

    label_path = args.label_path

    if label_path is not None:
        if not os.path.isfile(label_path):
            print(f"Error: Could not locate label file: {label_path}")
            return
    else:
        label_path = find_file("labels.txt", os.getcwd())
        if label_path is None:
            print(f"Error: Could not find label text file at: {label_path}")
            return
    labels = open(label_path).read().splitlines()

    if len(labels) == 0:
        print("Error: No labels found in file")

    # Add the background as the first class
    labels.insert(0, "background")

    print(f"Labels: {labels}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running inference on device: {device}")

    model = get_model(
        len(labels),
        box_score_thresh=args.threshold,
        min_size=00,
        max_size=600,
        box_nms_thresh=0.3,
    )

    print(f"Loading model state from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    model.to(device)

    model.eval()

    def user_callback(image_data):
        print("Entered callback")

        with torch.no_grad():

            if image_data is None:
                return None

            tensor_image = F.to_tensor(image_data)
            tensor_image = tensor_image.to(device)

            start_time = time.monotonic()
            outputs = model([tensor_image])
            outputs = [
                {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs
            ]
            inference_time_ms = (time.monotonic() - start_time) * 1000

            # filter out the background labels and scores bellow threshold
            filtered_outputs = [
                (
                    outputs[0]["boxes"][j],
                    outputs[0]["labels"][j],
                    outputs[0]["scores"][j],
                )
                for j in range(len(outputs[0]["boxes"]))
                if outputs[0]["scores"][j] > args.threshold
                and outputs[0]["labels"][j] > 0
            ]

            dwg = svgwrite.Drawing("", size=(image_data.shape[1], image_data.shape[0]))
            draw_text(dwg, 5, 30, f"Infernce time: {inference_time_ms:.2f}ms")

            print(f"Infernce time: {inference_time_ms:.2f}ms")

            for detection in filtered_outputs:
                bbox = detection[0].tolist()
                label_index = int(detection[1])
                score = float(detection[2])
                x, y = bbox[0], bbox[1]
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                draw_text(dwg, x, y - 5, f"{labels[label_index]} [{(100*score):.2f}%]")
                draw_rect(dwg, x, y, w, h, stroke_color="red", stroke_width=4)

                print(
                    f"\t{labels[label_index]}: [{(100*score):.2f}%] @ x={x} y={y} w={w} h={h}"
                )

            return dwg.tostring()

    run_pipeline(
        user_callback,
        src_frame_rate=args.frame_rate,
        src_height=args.source_height,
        src_width=args.source_width,
        binning_level=1,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="The model to load")

    parser.add_argument("--label_path", type=str, help="The labels to use")

    parser.add_argument("--source_width", type=int)

    parser.add_argument("--source_height", type=int)

    parser.add_argument("--frame_rate", type=float)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The threshold above which to display predicted bounding boxes",
    )

    args = parser.parse_args()

    main(args)
