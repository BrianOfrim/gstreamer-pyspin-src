import os
import time

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import torch
import torchvision

from gst_overlay_pipeline import run_pipeline


def get_model(model_name):
    return torch.hub.load("pytorch/vision:v0.6.0", model_name, pretrained=True)


def draw_text(dwg, x, y, text, font_size=25):
    dwg.add(
        dwg.text(text, insert=(x, max(y, font_size)), fill="white", font_size=font_size)
    )


def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running inference on device: {device}")

    model = get_model(args.model_name)

    model.eval()

    model.to(device)

    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    def user_callback(image_data):

        if image_data is None:
            return None

        input_image = Image.fromarray(image_data.astype("uint8"), "RGB")
        tensor_image = preprocess(input_image)

        input_batch = tensor_image.unsqueeze(0).to(device)

        # input_batch = input_batch.to(device)

        start_time = time.monotonic()
        with torch.no_grad():
            output = model(input_batch)["out"][0]

        inference_time_ms = (time.monotonic() - start_time) * 1000

        # print(output.shape)

        output_predictions = output.argmax(0)

        # print(output_predictions.shape)
        # index = output.data.numpy().argmax()
        # prob = torch.nn.functional.softmax(output[0], dim=0)[index]
        # label_text = labels[index] if labels and index <= len(labels) else index

        dwg = svgwrite.Drawing("", size=(image_data.shape[1], image_data.shape[0]))
        # draw_text(dwg, 5, 30, f"Class: {label_text}, {(100* prob): .2f}")
        draw_text(dwg, 5, 60, f"Inference time: {inference_time_ms:.2f}ms")

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(
            input_image.size
        )
        r.putpalette(colors)

        mask = Image.new("RGBA", input_image.size, (0, 0, 0, 123))

        overlay_image = Image.composite(r, input_image, mask).convert("RGB")

        cv2.imshow("segmentation", np.array(overlay_image))

        cv2.waitKey(1)

        return dwg.tostring()

    run_pipeline(
        user_callback,
        src_frame_rate=args.frame_rate,
        src_height=args.source_height,
        src_width=args.source_width,
        binning_level=args.binning_level,
        image_sink_sub_pipeline=args.sink_pipeline,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default="fcn_resnet101", help="The model to load"
    )

    parser.add_argument("--source_width", type=int)

    parser.add_argument("--source_height", type=int)

    parser.add_argument("--frame_rate", type=int)

    parser.add_argument("--binning_level", type=int, default=4)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The threshold above which to display class",
    )

    parser.add_argument(
        "--sink_pipeline",
        type=str,
        default="ximagesink sync=false",
        help="GStreamer pipline section for the image sink",
    )

    args = parser.parse_args()

    main(args)
