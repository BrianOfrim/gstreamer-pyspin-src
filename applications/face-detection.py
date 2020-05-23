import os
import time


from facenet_pytorch import MTCNN
import numpy as np
import svgwrite
import torch

from gst_overlay_pipeline import run_pipeline


def draw_text(dwg, x, y, text, font_size=25):
    dwg.add(
        dwg.text(text, insert=(x, max(y, font_size)), fill="white", font_size=font_size)
    )


def draw_rect(dwg, x, y, w, h, stroke_color="red", stroke_width=4):
    dwg.add(
        dwg.rect(
            insert=(int(x), int(y)),
            size=(int(w), int(h)),
            fill="none",
            stroke=stroke_color,
            stroke_width=stroke_width,
        )
    )


def draw_pts(dwg, pts, color="green", rad=5):
    for pt in pts:
        dwg.add(dwg.circle(center=(int(pt[0]), int(pt[1])), r=rad, fill=color))


def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running inference on device: {device}")

    mtcnn = MTCNN(device=device, keep_all=True)

    def user_callback(image_data):

        if image_data is None:
            return None

        start_time = time.monotonic()
        boxes, probs, points = mtcnn.detect(image_data, landmarks=True)
        inference_time_ms = (time.monotonic() - start_time) * 1000

        dwg = svgwrite.Drawing("", size=(image_data.shape[1], image_data.shape[0]))
        draw_text(dwg, 5, 30, f"Infernce time: {inference_time_ms:.2f}ms")

        print(f"Inference time: {inference_time_ms:.2f}ms")

        if boxes is not None:
            for box, prob, pts in zip(boxes, probs, points):
                if prob < args.threshold:
                    continue

                x, y = box[0], box[1]
                w, h = box[2] - box[0], box[3] - box[1]

                draw_rect(dwg, x, y, w, h, stroke_color="red", stroke_width=4)
                draw_pts(dwg, pts)

                print(f"\tx={x} y={y} w={w} h={h} ({(100*prob):.2f}%)")

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

    parser.add_argument(
        "--sink_pipeline",
        type=str,
        default="ximagesink sync=false",
        help="GStreamer pipline section for the image sink",
    )

    main(parser.parse_args())
