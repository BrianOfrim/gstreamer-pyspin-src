import os
import time


from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

from gst_appsink_display import run_pipeline


def draw_text(img_draw, x, y, text, font=None):
    img_draw.text((x, y), text, font=font)


def draw_rect(img_draw, box, boxcolor="black"):
    img_draw.rectangle(box, fill=boxcolor)


def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running inference on device: {device}")

    mtcnn = MTCNN(device=device, keep_all=True)

    text_font = ImageFont.truetype(
        "/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf", 25
    )

    def user_callback(image_data):

        if image_data is None:
            return None

        start_time = time.monotonic()
        boxes, probs, _ = mtcnn.detect(image_data, landmarks=True)
        inference_time_ms = (time.monotonic() - start_time) * 1000

        augmented_image = Image.fromarray(image_data.astype("uint8"), "RGB")

        img_draw = ImageDraw.Draw(augmented_image)

        draw_text(
            img_draw, 5, 5, f"Infernce time: {inference_time_ms:.2f}ms", font=text_font
        )

        print(f"Inference time: {inference_time_ms:.2f}ms")

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < args.threshold:
                    continue

                draw_rect(img_draw, box)
                print(f"\tbox={box} ({(100*prob):.2f}%)")

        return augmented_image

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

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="The threshold above which to display predicted bounding boxes",
    )

    main(parser.parse_args())
