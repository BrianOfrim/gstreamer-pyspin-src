import math
import os
import time


from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
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

    db = DBSCAN(eps=1.05, min_samples=1, metric='precomputed')

    cluster_centers = torch.Tensor()
    
    def user_callback(image_data):

        nonlocal cluster_centers 

        if image_data is None:
            return None


        augmented_image = Image.fromarray(image_data.astype("uint8"), "RGB")

        start_time = time.monotonic()
        faces, probs = mtcnn(image_data, return_prob=True)

        if faces is not None:
            print(faces.shape)
            device_faces = faces.to(device)
            embeddings = resnet(device_faces).detach().cpu()
            print(embeddings.shape)

            # Add existing cluster center embeddings
            embeddings = torch.cat([embeddings, cluster_centers], dim=0)
            print("shape after adding existing clusters")
            print(embeddings.shape)


            faces = [to_pil(face) for face in torch.unbind(faces)]
            face_width = faces[0].width 
            face_height = faces[0].height

            matrix = np.zeros((embeddings.shape[0], embeddings.shape[0]))

            print('')
            # Print distance matrix
            print('Distance matrix')
            print(f"{'':10}", end='')
            for i in range(embeddings.shape[0]):
                if i < len(faces):
                    print(f"{f'face{i}':^10}", end="")
                else:
                    print(f"{f'clus{i - len(faces)}':^10}", end="")
            print('')
            for i1, e1 in enumerate(embeddings):
                if i1 < len(faces):
                    print(f"{f'face{i1}':^10}", end="")
                else:
                    print(f"{f'clus{i1 - len(faces)}':^10}", end="") 
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
                        print(f"Cluster {i}: {cluster_members}")
                        if (cluster_members >= len(faces)).all():
                            print("No new faces in the cluster")
                            continue
                        existing_cluster_center = next((c for c in cluster_members if c >= len(faces)), None)
                        if(existing_cluster_center is not None):
                            print(f"Cluster containes existing center: {existing_cluster_center}")
                            # filter out any other existing cluster centers that might be in the current cluster for some reason
                            cluster_members = [ cm for cm in cluster_members if cm == existing_cluster_center or cm < len(faces)]
                            cluster_embeddings = embeddings[cluster_members]
                            print(cluster_embeddings.shape)
                            cluster_center = torch.mean(cluster_embeddings ,0)
                            # update the center of the cluster to incluse the new faces
                            old_cluster_center = embeddings[existing_cluster_center]
                            cluster_centers[existing_cluster_center - len(faces)] = cluster_center
                            print(f"Cluster center moved by: {(cluster_center - old_cluster_center).norm().item()}")
                        else:
                            print("Cluster contains all new faces")

                            cluster_embeddings = embeddings[cluster_members]
                            print(cluster_embeddings.shape)
                            cluster_center = torch.mean(cluster_embeddings ,0)
                            cluster_centers = torch.cat([cluster_centers, cluster_center.unsqueeze(0)], dim=0)
                            # print("Distance to cluster center:")
                            # for m, e in zip(cluster_members, cluster_embeddings):
                            #     print(f"Center to {m}: {(cluster_center - e).norm().item()}")
                            

                        
                        k = 0
                        for j in np.nonzero(labels == i)[0]:
                            x = k * face_width
                            y = i * face_height
                            if j < len(faces):
                                augmented_image.paste(faces[j], (x,y))
                                k+=1

        inference_time_ms = (time.monotonic() - start_time) * 1000
        print(f"Inference time: {inference_time_ms}")

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
