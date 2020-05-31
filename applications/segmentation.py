import os
import time

from PIL import Image, ImageMath
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import torch
import torchvision

from gst_overlay_pipeline import run_pipeline


def get_model(model_name):
    # return torch.hub.load("pytorch/vision:v0.6.0", model_name, pretrained=True)
    return torchvision.models.segmentation.__dict__[model_name](pretrained=True)


def get_mask(image):
    r, g, b, _ = image.split()
    r_mask = r.point(lambda i: i > 0 and 255)
    g_mask = g.point(lambda i: i > 0 and 255)
    b_mask = b.point(lambda i: i > 0 and 255)
    return ImageMath.eval("convert(a | b | c, 'L')", a=r_mask, b=g_mask, c=b_mask)


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

        start_time = time.monotonic()
        with torch.no_grad():
            output = model(input_batch)["out"][0]

        inference_time_ms = (time.monotonic() - start_time) * 1000

        print(f"Inference time: {inference_time_ms}ms")

        output_predictions = output.argmax(0)

        seg_mask = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(
            input_image.size
        )

        seg_mask.putpalette(colors)

        seg_mask = seg_mask.convert("RGBA")

        seg_mask.putalpha(get_mask(seg_mask))

        return seg_mask

    run_pipeline(
        user_callback,
        src_frame_rate=args.frame_rate,
        src_height=args.source_height,
        src_width=args.source_width,
        binning_level=args.binning_level,
        overlay_element="gdkpixbufoverlay",
        image_sink_sub_pipeline=args.sink_pipeline,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default="fcn_resnet50", help="The model to load",
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
