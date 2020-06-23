import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from gst_appsink_display import run_pipeline


def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running inference on device: {device}")

    model = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    color_map = plt.get_cmap("inferno")

    def user_callback(image_data):

        if image_data is None:
            return None

        input_batch = transform(image_data).to(device)

        start_time = time.monotonic()

        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_data.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        inference_time_ms = (time.monotonic() - start_time) * 1000

        print(f"Inference time: {inference_time_ms:.2f}ms")
        output = prediction.cpu().numpy()

        depth_min = output.min()
        depth_max = output.max()

        output = (output - depth_min) / (depth_max - depth_min)
        output = color_map(output)
        output = output[:, :, :3]
        output = output * 255
        output = output.astype(np.uint8)

        return np.concatenate((image_data, output), axis=1)

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

    parser.add_argument("--source_width", type=int)

    parser.add_argument("--source_height", type=int)

    parser.add_argument("--frame_rate", type=int)

    parser.add_argument("--binning_level", type=int, default=2)

    main(parser.parse_args())
