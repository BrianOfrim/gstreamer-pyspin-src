import math
import os
import time
from typing import List


from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
import torch
import torchvision

from gst_app_src_and_sink import run_pipeline

class ClusterCenter:
    def __init__(self,  center = None, beta = 0.95):
        self.beta = beta
        self.items_clustered = 0 if center is None else 1
        self.center = center
    def re_center(self, new_embedding, print_delta = True):
        self.items_clustered += 1
        if self.center is None:
            self.center = new_embedding
        else:
            prev_center = self.center
            self.center = (self.beta * self.center) + (1 - self.beta) * new_embedding
            if print_delta:
                print(f"Cluster center moved by: {(self.center - prev_center).norm().item()}")
                print(f"Number of faces clustered total: {self.items_clustered}")


def stack_cluster_centers(clusters: List[ClusterCenter]):
    return torch.stack([cluster.center for cluster in clusters], dim=0)

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

    db = DBSCAN(eps=0.9, min_samples=1, metric='precomputed')

    cluster_centers: List[ClusterCenter] = []
    
    def user_callback(image_data):

        nonlocal cluster_centers 

        if image_data is None:
            return None

        augmented_image = Image.fromarray(image_data.astype("uint8"), "RGB")

        img_draw = ImageDraw.Draw(augmented_image)

        start_time = time.monotonic()

        with torch.no_grad():
            boxes, probs = mtcnn.detect(image_data)

        if boxes is not None and len(boxes) > 0:
            faces = []

            for box in boxes:
                face = extract_face(image_data, box)
                faces.append(fixed_image_standardization(face))

            print(f"Number of faces: {len(faces)}")

            faces = torch.stack(faces, dim=0)
            device_faces = faces.to(device)
            embeddings = resnet(device_faces).detach().cpu()

            if len(cluster_centers) > 0:
                # Add new face embeddings to existing cluster center embeddings
                embeddings = torch.cat([stack_cluster_centers(cluster_centers), embeddings], dim=0)

            print(f" Embedding shape after adding existing clusters: {embeddings.shape}")

            # face_images = [to_pil(face) for face in torch.unbind(faces)]

            matrix = np.zeros((embeddings.shape[0], embeddings.shape[0]))

            num_existing_clusters = len(cluster_centers)

            print(f"Number of existing clusters: {num_existing_clusters}")

            print('')
            # Print distance matrix
            print('Distance matrix')
            print(f"{'':10}", end='')
            for i in range(embeddings.shape[0]):
                if i < num_existing_clusters:
                    print(f"{f'clust{i}':^10}", end="")
                else:
                    print(f"{f'face{i - num_existing_clusters}':^10}", end="")
            print('')
            for i1, e1 in enumerate(embeddings):
                if i1 < num_existing_clusters:
                    print(f"{f'clust{i1}':^10}", end="")
                else:
                    print(f"{f'face{i1 - num_existing_clusters}':^10}", end="") 
                for i2, e2 in enumerate(embeddings):
                    dist = (e1 - e2).norm().item()
                    matrix[i1][i2] = dist
                    print(f"{dist:^10.4f}", end='')
                print('')

            print('')

            db.fit(matrix)
            labels = db.labels_


            # get number of clusters
            no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            print('No of clusters:', no_clusters)
            print(labels)

            if no_clusters > 0:
                    for i in range(no_clusters):
                        # skip any clusters that are just make of existing cluster centers
                        cluster_members = np.nonzero(labels == i)[0]
                        face_members = [ cm for cm in cluster_members if cm >= num_existing_clusters]
                        center_members = [cm for cm in cluster_members if cm < num_existing_clusters] 
                        print(f"Cluster {i}: {cluster_members}, faces: {face_members}, centers: {center_members}")

                        if len(face_members) == 0:
                            print("No new faces in the cluster")
                            continue

                        face_embeddings = embeddings[face_members]
                        print(f"Face embeddings shape :{face_embeddings.shape}")

                        if(len(center_members) > 0):
                            print(f"Cluster containes existing center: {center_members[0]}")
                            cluster_centers[center_members[0]].re_center(torch.mean(face_embeddings ,0))
                        else:
                            print("Cluster contains all new faces")
                            cluster_centers.append(ClusterCenter(center=torch.mean(face_embeddings ,0)))

                        face_indecies = [cm - num_existing_clusters for cm in face_members]
                        for box in boxes[face_indecies]:
                            draw_rect(img_draw, box)
                            draw_text(img_draw, box[0] , box[1] , str(center_members[0]) if len(center_members) > 0 else "?", font=box_text_font)
                        

        inference_time_ms = (time.monotonic() - start_time) * 1000
        # print(f"Inference time: {inference_time_ms}")

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
