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


def run_pipeline(pipelineStr):

    print("Launch string:", pipelineStr)
    pipeline = Gst.parse_launch(pipelineStr)

    pipeline.set_state(Gst.State.PLAYING)
    state_change_info = pipeline.get_state(Gst.CLOCK_TIME_NONE)
    print(
        f"Image src pipeline state change to running successful? : {state_change_info[0] == Gst.StateChangeReturn.SUCCESS}"
    )

    loop = GObject.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

    pipeline.set_state(Gst.State.PAUSED)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

    while GLib.MainContext.default().iteration(False):
        pass


if __name__ == "__main__":
    run_pipeline(
        "pyspinsrc num-buffers=10 exposure=10000 ! video/x-raw, framerate=10/1, width=480 ! videoconvert ! xvimagesink"
    )
    run_pipeline(
        "pyspinsrc num-buffers=10 exposure=20000 ! video/x-raw, framerate=15/1, height=480 ! videoconvert ! xvimagesink"
    )
