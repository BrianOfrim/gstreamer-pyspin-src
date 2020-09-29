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


def log_timestamp(pad, info, u_data):

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    print(f"FrameID: {gst_buffer.offset:<10} Timestamp: {gst_buffer.pts}ns")
    return Gst.PadProbeReturn.OK


def run_pipeline(pipelineStr):

    print("Launch string:", pipelineStr)

    pipeline = Gst.parse_launch(pipelineStr)
    pipeline.set_state(Gst.State.PLAYING)
    pyspinsrc = pipeline.get_by_name("pyspinsrc")
    srcpad = pyspinsrc.get_static_pad("src")
    srcpad.add_probe(Gst.PadProbeType.BUFFER, log_timestamp, 0)

    loop = GObject.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

    while GLib.MainContext.default().iteration(False):
        pass


if __name__ == "__main__":
    run_pipeline(
        "pyspinsrc name=pyspinsrc num-buffers=100 ! video/x-raw, framerate=10/1, width=480 ! videoconvert ! xvimagesink"
    )
