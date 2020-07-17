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

    mtcnn = MTCNN(device=device, keep_all=True)

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


    header_text_font = ImageFont.truetype(
        "/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf", 25
    )

    box_text_font = ImageFont.truetype(
        "/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf", 15
    ) 

    to_pil = torchvision.transforms.ToPILImage()
    
    def user_callback(image_data):

        if image_data is None:
            return None
        
        augmented_image = Image.fromarray(image_data.astype("uint8"), "RGB")

        start_time = time.monotonic()
        faces, probs = mtcnn(image_data, return_prob=True)

        if faces is not None:
            print(faces.shape)
            # alligned = torch.stack([faces.to(device)
            device_faces = faces.to(device)
            embbeddings = resnet(device_faces).detach().cpu()
            print(embbeddings.shape)
            # print(embbeddings)

            faces = [to_pil(face) for face in torch.unbind(faces)]
            face_width = faces[0].width 
            face_height = faces[0].height  
            face_per_row = math.floor(augmented_image.width/face_width)
            for i, face in enumerate(faces):
                augmented_image.paste(face, ((i % face_per_row) *  face_width, math.floor(i/face_per_row)* face_height))
            
        
        
        inference_time_ms = (time.monotonic() - start_time) * 1000
        print(f"Inference time: {inference_time_ms}")



        

        # augmented_image = Image.fromarray(image_data.astype("uint8"), "RGB")

        # img_draw = ImageDraw.Draw(augmented_image)

        # draw_text(
        #     img_draw, 5, 5, f"Infernce time: {inference_time_ms:.2f}ms", font=header_text_font
        # )

        # print(f"Inference time: {inference_time_ms:.2f}ms")

        # if boxes is not None and len(boxes) > 0:
        #     tracked_faces = mot_tracker.update(np.append(boxes, np.expand_dims(probs, axis=0).T, axis=1))

        #     for tf in tracked_faces:
        #         print(tf)
        #         draw_rect(img_draw, tf[0:4].astype(np.float32), stroke_color="green")
        #         draw_text(img_draw, int(tf[0]), int(tf[1]) - 15 , f"person: {int(tf[-1])}", font=box_text_font)

        #     # for box, prob in zip(boxes, probs):
        #     #     if prob < args.threshold:
        #     #         continue
        #     #     draw_rect(img_draw, box)
        #     #     print(f"\tbox={box} ({(100*prob):.2f}%)")
        # else:
        #     mot_tracker.update(np.empty((0, 5)))

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
