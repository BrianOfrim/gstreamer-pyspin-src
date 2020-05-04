# Based on https://github.com/google-coral/examples-camera/tree/master/gstreamer
import sys
import svgwrite
import threading
import fnmatch
import os

import numpy as np

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, GObject, Gst, GstBase, Gtk

GObject.threads_init()
Gst.init(None)


class GstPipeline:
    def __init__(self, pipeline, user_function):
        self.user_function = user_function
        self.running = False
        self.gstbuffer = None
        self.npbuffer = None
        self.sink_size = None
        self.box = None
        self.condition = threading.Condition()

        self.pipeline = Gst.parse_launch(pipeline)
        self.overlay = self.pipeline.get_by_name("overlay")
        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)

        # Set up a pipeline bus watch to catch errors.
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

    def run(self):
        # Start inference worker.
        self.running = True
        worker = threading.Thread(target=self.inference_loop)
        worker.start()

        # Run pipeline.
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            Gtk.main()
        except:
            pass

        # Clean up.
        self.pipeline.set_state(Gst.State.NULL)
        while GLib.MainContext.default().iteration(False):
            pass
        with self.condition:
            self.running = False
            self.condition.notify_all()
        worker.join()

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            Gtk.main_quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            Gtk.main_quit()
        return True

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not self.sink_size:
            s = sample.get_caps().get_structure(0)
            self.sink_size = (s.get_value("height"), s.get_value("width"))
        if self.npbuffer is None:
            self.npbuffer = np.zeros((*self.sink_size, 3), dtype=np.uint8)
        with self.condition:
            self.gstbuffer = sample.get_buffer()
            self.condition.notify_all()
        return Gst.FlowReturn.OK

    def inference_loop(self):
        while True:
            with self.condition:
                while not self.gstbuffer and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                gstbuffer = self.gstbuffer
                self.gstbuffer = None

            # """Copies data to input tensor."""
            result, mapinfo = gstbuffer.map(Gst.MapFlags.READ)
            if result:
                self.npbuffer[:] = np.ndarray(
                    (*self.sink_size, 3), buffer=mapinfo.data, dtype=np.uint8
                )
                gstbuffer.unmap(mapinfo)

                svg = self.user_function(self.npbuffer)
                if svg:
                    if self.overlay:
                        self.overlay.set_property("data", svg)


def run_pipeline(
    user_function,
    src_frame_rate: float = None,
    src_height: int = None,
    src_width: int = None,
):

    SRC_CAPS = "video/x-raw,format=RGB"
    if src_frame_rate is not None:
        SRC_CAPS += f",framerate={src_frame_rate}"

    if src_height is not None:
        SRC_CAPS += f",height={src_height}"

    if src_width is not None:
        SRC_CAPS += f",width={src_width}"

    PIPELINE = "pyspinsrc ! {src_caps} "

    PIPELINE += """ ! tee name=t
        t. ! {leaky_q} ! videoconvert ! {sink_caps} ! {sink_element}
        t. ! {leaky_q} ! videoconvert ! rsvgoverlay name=overlay ! videoconvert ! ximagesink sync=false
        """

    SINK_ELEMENT = "appsink name=appsink emit-signals=true max-buffers=1 drop=true"
    SINK_CAPS = "video/x-raw,format=RGB"
    LEAKY_Q = "queue max-size-buffers=1 leaky=downstream"

    pipeline = PIPELINE.format(
        leaky_q=LEAKY_Q,
        src_caps=SRC_CAPS,
        sink_caps=SINK_CAPS,
        sink_element=SINK_ELEMENT,
    )

    print("Gstreamer pipeline:\n", pipeline)

    pipeline = GstPipeline(pipeline, user_function)
    pipeline.run()


def find_file(name, path) -> str:
    for root, dirs, files in os.walk(path):
        filematch = next((f for f in files if fnmatch.fnmatch(f, name)), None)
        if filematch:
            return os.path.join(root, filematch)
    return None


def fasterrcnn_resnet50(num_classes, **kwargs):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, **kwargs
    )
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def shadow_text(dwg, x, y, text, font_size=20):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill="black", font_size=font_size))
    dwg.add(dwg.text(text, insert=(x, y), fill="white", font_size=font_size))


def main(args):

    model_path = args.model_path

    if model_path is not None:
        if not os.path.isfile(model_path):
            print(f"Error: Could not locate model file: {model_path}")
            return
    else:
        model_path = find_file("*.pt", os.getcwd())
        if model_path is None:
            print(f"Error: Could not locate a model '.pt' file in {os.getcwd()}")
            return

    label_path = args.label_path

    if label_path is not None:
        if not os.path.isfile(label_path):
            print(f"Error: Could not locate label file: {label_path}")
            return
    else:
        label_path = find_file("labels.txt", os.getcwd())
        if label_path is None:
            print(f"Error: Could not find label text file at: {label_path}")
            return
    labels = open(label_path).read().splitlines()

    if len(labels) == 0:
        print("Error: No labels found in file")

    # Add the background as the first class
    labels.insert(0, "background")

    print("Labels:")
    print(labels)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # get the model using our helper function
    model = fasterrcnn_resnet50(
        len(labels),
        box_score_thresh=args.threshold,
        min_size=600,
        max_size=800,
        box_nms_thresh=0.3,
    )

    print("Loading model state from: %s" % model_path)
    checkpoint = torch.load(model_path, map_location=device)

    print(checkpoint.keys())

    model.load_state_dict(checkpoint["model"])

    # move model to the right device
    model.to(device)

    model.eval()

    def user_callback(image_data):
        print("Usercallback")
        print(image_data.size)

        image_height = image_data.shape[0]
        image_width = image_data.shape[1]

        with torch.no_grad():

            if image_data is None:
                return None

            tensor_image = F.to_tensor(image_data)
            tensor_image = tensor_image.to(device)

            outputs = model([tensor_image])
            outputs = [
                {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs
            ]

            # filter out the background labels and scores bellow threshold
            filtered_outputs = [
                (
                    outputs[0]["boxes"][j],
                    outputs[0]["labels"][j],
                    outputs[0]["scores"][j],
                )
                for j in range(len(outputs[0]["boxes"]))
                if outputs[0]["scores"][j] > args.threshold
                and outputs[0]["labels"][j] > 0
            ]

            print(filtered_outputs)

            dwg = svgwrite.Drawing("", size=(image_width, image_height))

            for detection in filtered_outputs:
                bbox = detection[0].tolist()
                label_index = int(detection[1])
                score = float(detection[2])
                box_x, box_y = bbox[0], bbox[1]
                box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

                box_text = f"{labels[label_index]} [{(100*score):.2f}%]"
                shadow_text(dwg, box_x, box_y - 5, box_text)
                dwg.add(
                    dwg.rect(
                        insert=(box_x, box_y),
                        size=(box_width, box_height),
                        fill="none",
                        stroke="red",
                        stroke_width="2",
                    )
                )

            return dwg.tostring()

    run_pipeline(
        user_callback,
        src_frame_rate=args.frame_rate,
        src_height=args.source_height,
        src_width=args.source_width,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="The model to load")

    parser.add_argument("--label_path", type=str, help="The labels to use")

    parser.add_argument("--source_width", type=int)

    parser.add_argument("--source_height", type=int)

    parser.add_argument("--inference_width", type=int)

    parser.add_argument("--inference_height", type=int)

    parser.add_argument("--frame_rate", type=float)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The threshold above which to display predicted bounding boxes",
    )

    args = parser.parse_args()

    main(args)
