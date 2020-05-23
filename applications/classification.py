import os
import time

from PIL import Image
import svgwrite
import torch
import torchvision

from gst_overlay_pipeline import run_pipeline


def get_model(model_name):
    return torch.hub.load("pytorch/vision", model_name, pretrained=True)


def draw_text(dwg, x, y, text, font_size=25):
    dwg.add(
        dwg.text(text, insert=(x, max(y, font_size)), fill="white", font_size=font_size)
    )


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
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def user_callback(image_data):

        if image_data is None:
            return None

        tensor_image = preprocess(Image.fromarray(image_data.astype("uint8"), "RGB"))

        input_batch = tensor_image.unsqueeze(0).to(device)

        # input_batch = input_batch.to(device)

        start_time = time.monotonic()
        with torch.no_grad():
            output = model(input_batch)
        inference_time_ms = (time.monotonic() - start_time) * 1000

        index = output.data.numpy().argmax()
        prob = torch.nn.functional.softmax(output[0], dim=0)[index]
        label_text = labels[index] if labels and index <= len(labels) else index

        dwg = svgwrite.Drawing("", size=(image_data.shape[1], image_data.shape[0]))
        draw_text(dwg, 5, 30, f"Class: {label_text}, {(100* prob): .2f}")
        draw_text(dwg, 5, 60, f"Inference time: {inference_time_ms:.2f}ms")

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
        "--model_name", type=str, default="mobilenet_v2", help="The model to load"
    )

    parser.add_argument(
        "--label_path",
        type=str,
        default="imagenet-classes.txt",
        help="The labels to use",
    )

    parser.add_argument("--source_width", type=int)

    parser.add_argument("--source_height", type=int)

    parser.add_argument("--frame_rate", type=int)

    parser.add_argument("--binning_level", type=int, default=2)

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
