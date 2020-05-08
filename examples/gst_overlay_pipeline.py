# Based on https://github.com/google-coral/examples-camera/tree/master/gstreamer
import sys
import svgwrite
import threading

import numpy as np

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
        # Start vision worker.
        self.running = True
        worker = threading.Thread(target=self.vision_loop)
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
        with self.condition:
            self.gstbuffer = sample.get_buffer()
            self.condition.notify_all()
        return Gst.FlowReturn.OK

    def vision_loop(self):
        local_np_buffer = None

        while True:
            with self.condition:
                while not self.gstbuffer and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                local_gst_buffer = self.gstbuffer
                self.gstbuffer = None

            if local_np_buffer is None:
                local_np_buffer = np.zeros((*self.sink_size, 3), dtype=np.uint8)

            result, mapinfo = local_gst_buffer.map(Gst.MapFlags.READ)
            if result:
                local_np_buffer[:] = np.ndarray(
                    (*self.sink_size, 3), buffer=mapinfo.data, dtype=np.uint8
                )
                local_gst_buffer.unmap(mapinfo)

                svg = self.user_function(local_np_buffer)
                if svg and self.overlay:
                    self.overlay.set_property("data", svg)


def run_pipeline(
    user_function,
    src_frame_rate: float = None,
    src_height: int = None,
    src_width: int = None,
    binning_level: int = 1,
    image_sink_sub_pipeline: str = "ximagesink sync=false",
):

    image_src_element = "pyspinsrc"
    if binning_level != 1:
        image_src_element += f" h-binning={binning_level} v-binning={binning_level}"

    image_src_caps = "video/x-raw,format=RGB"
    if src_frame_rate is not None:
        image_src_caps += f",framerate={src_frame_rate}"

    if src_height is not None:
        image_src_caps += f",height={src_height}"

    if src_width is not None:
        image_src_caps += f",width={src_width}"

    appsink_element = "appsink name=appsink emit-signals=true max-buffers=1 drop=true"
    appsink_caps = "video/x-raw,format=RGB"
    leaky_queue = "queue max-size-buffers=1 leaky=downstream"
    overlay_element = "rsvgoverlay name=overlay"

    pipeline = f""" {image_src_element} ! {image_src_caps} ! tee name=t
        t. ! {leaky_queue} ! videoconvert ! {appsink_caps} ! {appsink_element}
        t. ! {leaky_queue} ! videoconvert ! {overlay_element} ! videoconvert ! {image_sink_sub_pipeline}
        """

    print("Gstreamer pipeline:\n", pipeline)

    pipeline = GstPipeline(pipeline, user_function)
    pipeline.run()
