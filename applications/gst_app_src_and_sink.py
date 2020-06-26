# Based on https://github.com/google-coral/project-bodypix/blob/master/gstreamer.py
from functools import partial
import sys
import time

import numpy as np

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import GLib, GObject, Gst, GstBase

GObject.threads_init()
Gst.init(None)


def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def on_new_sample(sink, appsrc, user_function):
    sample = sink.emit("pull-sample")
    sample_struct = sample.get_caps().get_structure(0)

    buf = sample.get_buffer()
    result, mapinfo = buf.map(Gst.MapFlags.READ)

    if result:
        img = np.frombuffer(mapinfo.data, np.uint8)
        img = np.reshape(
            img,
            (sample_struct.get_value("height"), sample_struct.get_value("width"), -1),
        )
        next_image = user_function(img.copy())

        if appsrc is not None:
            next_image = np.array(next_image)
            data = next_image.tobytes()
            next_buffer = Gst.Buffer.new_allocate(None, len(data), None)
            next_buffer.fill(0, data)
            appsrc.emit("push-buffer", next_buffer)

    buf.unmap(mapinfo)
    return Gst.FlowReturn.OK


def run_pipeline(
    user_function,
    src_frame_rate: int = None,
    src_height: int = None,
    src_width: int = None,
    binning_level: int = 1,
    image_sink_bin: str = "ximagesink sync=false",
):

    image_src_element = "pyspinsrc name=imagesrc"
    if binning_level is not None and binning_level != 1:
        image_src_element += f" h-binning={binning_level} v-binning={binning_level}"

    image_src_caps = "video/x-raw,format=RGB"
    if src_frame_rate is not None:
        image_src_caps += f",framerate={int(src_frame_rate)}/1"

    if src_height is not None:
        image_src_caps += f",height={src_height}"

    if src_width is not None:
        image_src_caps += f",width={src_width}"

    appsink_element = "appsink name=appsink emit-signals=true max-buffers=1 drop=true"
    appsink_caps = "video/x-raw,format=RGB"
    leaky_queue = "queue max-size-buffers=1 leaky=downstream"
    appsrc_element = "appsrc name=appsrc"

    image_src_pipeline = f" {image_src_element} ! {image_src_caps} ! {leaky_queue} ! videoconvert ! {appsink_caps} ! {appsink_element}"
    print("Image src pipeline:\n", image_src_pipeline)
    image_src_pipeline = Gst.parse_launch(image_src_pipeline)

    appsink = image_src_pipeline.get_by_name("appsink")

    # start image source pipeling and block until playing
    image_src_pipeline.set_state(Gst.State.PLAYING)
    state_change_info = image_src_pipeline.get_state(Gst.CLOCK_TIME_NONE)
    print(
        f"Image src pipeline state change to running successful? : {state_change_info[0] == Gst.StateChangeReturn.SUCCESS}"
    )

    image_sink_pipeline = f"{appsrc_element} ! {str(appsink.sinkpad.get_current_caps())} ! {leaky_queue} ! videoconvert ! {image_sink_bin}"
    print("Image sink pipeline:\n", image_sink_pipeline)
    image_sink_pipeline = Gst.parse_launch(image_sink_pipeline)

    appsrc = image_sink_pipeline.get_by_name("appsrc")

    appsink.connect(
        "new-sample",
        partial(on_new_sample, appsrc=appsrc, user_function=user_function),
    )

    loop = GObject.MainLoop()

    bus = image_src_pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    image_sink_pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    image_src_pipeline.set_state(Gst.State.NULL)
    image_sink_pipeline.set_state(Gst.State.NULL)
    while GLib.MainContext.default().iteration(False):
        pass
