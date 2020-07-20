import math
import os
import time


from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision

from gst_app_src_and_sink import run_pipeline

def draw_text(img_draw, x, y, text, font=None):
    img_draw.text((max(0,x), max(0,y)), text, font=font)


def draw_rect(img_draw, box, stroke_color="red", stroke_width=4):
    img_draw.rectangle(box, outline=stroke_color, width=stroke_width)


def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running inference on device: {device}")

    mtcnn = MTCNN(device=device, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True)

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


    header_text_font = ImageFont.truetype(
        "/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf", 25
    )

    box_text_font = ImageFont.truetype(
        "/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf", 15
    ) 

    to_pil = torchvision.transforms.ToPILImage()

    image_set_num = [0]
    
    def user_callback(image_data):

        if image_data is None:
            return None


        augmented_image = Image.fromarray(image_data.astype("uint8"), "RGB")
        image_draw = ImageDraw.Draw(augmented_image)

        start_time = time.monotonic()
        faces, probs = mtcnn(image_data, return_prob=True)

        if faces is not None:
            print(faces.shape)
            device_faces = faces.to(device)
            embeddings = resnet(device_faces).detach().cpu()
            print(embeddings.shape)

            faces = [to_pil(face) for face in torch.unbind(faces)]
            face_width = faces[0].width 
            face_height = faces[0].height  
            face_per_row = math.floor(augmented_image.width/face_width)
            for i, face in enumerate(faces):
                x = (i % face_per_row) *  face_width
                y = math.floor(i/face_per_row) * face_height

                augmented_image.paste(face, (x,y))
                draw_text(image_draw, x, y-20, str(i), box_text_font)

            for i1, e1 in enumerate(embeddings):
                for i2, e2 in enumerate(embeddings):
                    print(f"{i1} - {i2}: {(e1 - e2).norm().item()}")

        inference_time_ms = (time.monotonic() - start_time) * 1000
        print(f"Inference time: {inference_time_ms}")
        
        image_set_num[0] += 1 

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
