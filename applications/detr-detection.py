# Model from https://github.com/facebookresearch/detr
# Inference based on https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
import fnmatch
import os
import time

from PIL import Image
import numpy as np
import svgwrite
import torch
import torchvision

from gst_overlay_pipeline import run_pipeline


def get_model(model_name):
    return torch.hub.load("facebookresearch/detr", model_name, pretrained=True)


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


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def main(args):

    labels = None
    if args.label_path is not None and os.path.isfile(args.label_path):
        labels = open(args.label_path).read().splitlines()
        print("Number of labels loaded: {labels}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running inference on device: {device}")

    model = get_model(args.model_name)

    model.eval()

    model.to(device)

    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(600),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    def user_callback(image_data):

        if image_data is None:
            return None

        pil_image = Image.fromarray(image_data.astype("uint8"), "RGB")
        tensor_image = preprocess(pil_image)

        input_batch = tensor_image.unsqueeze(0).to(device)

        start_time = time.monotonic()
        with torch.no_grad():
            outputs = model(input_batch)
        inference_time_ms = (time.monotonic() - start_time) * 1000

        # outputs = [
        #     {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs
        # ]

        # keep only predictions with 0.7+ confidence
        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > args.threshold

        # convert boxes from [0; 1] to image scales
        boxes = rescale_bboxes(outputs["pred_boxes"][0, keep], pil_image.size)

        scores = probas[keep]

        dwg = svgwrite.Drawing("", size=(image_data.shape[1], image_data.shape[0]))
        # draw_text(dwg, 5, 30, f"Infernce time: {inference_time_ms:.2f}ms")

        print(f"Infernce time: {inference_time_ms:.2f}ms")

        for p, bbox in zip(scores, boxes.tolist()):

            x, y = bbox[0], bbox[1]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            cl = p.argmax()
            text = f"{labels[cl] if labels else cl}: {p[cl]:0.2f}"

            draw_text(dwg, x, y - 5, text)
            draw_rect(dwg, x, y, w, h, stroke_color="red", stroke_width=4)

            print(f"\t{text} @ x={x} y={y} w={w} h={h}")

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
        "--model_name", type=str, default="detr_resnet50", help="The model to load",
    )

    parser.add_argument(
        "--label_path", type=str, default="val5k-labels.txt", help="The labels to use",
    )

    parser.add_argument("--source_width", type=int)

    parser.add_argument("--source_height", type=int)

    parser.add_argument("--frame_rate", type=int)

    parser.add_argument("--binning_level", type=int, default=2)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="The threshold above which to display predicted bounding boxes",
    )

    parser.add_argument(
        "--sink_pipeline",
        type=str,
        default="ximagesink sync=false",
        help="GStreamer pipline section for the image sink",
    )

    args = parser.parse_args()

    main(args)
