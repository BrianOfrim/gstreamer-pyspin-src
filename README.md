# GStreamer PySpin Src Plug-in

A GStreamer source plug in for the PySpin Image Acquisition SDK

## Install

Download the latest version of **Spinnaker** and the matching version of **PySpin** (spinnaker-python) from:
https://flir.app.box.com/v/SpinnakerSDK/

    git clone https://github.com/BrianOfrim/gstreamer-pyspin-src.git
    cd gstreamer-pyspin-src

    python3 -m venv venv
    source venv/bin/activate
    pip install -U wheel pip setuptools

    pip install -r requirements.txt
    pip install <path-to-pyspin-package>\spinnaker_python<version-info>.whl

## Usage

Tell GStreamer where our plugin is located:

    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/

Example debugging pipeline:

    GST_DEBUG=python:6 gst-launch-1.0 --gst-disable-segtrap --gst-disable-registry-fork pyspinsrc ! videoconvert ! autovideosink

Example pileline:

    gst-launch-1.0 pyspinsrc ! videoconvert  ! autovideosink

View plug-in info and properties:

    gst-inspect-1.0 pyspinsrc

## References

Uses the following for gst buffer to numpy mapping utilities and to install [gst-python](https://github.com/GStreamer/gst-python):  
https://github.com/jackersson/gstreamer-python

Project based on:  
https://github.com/jackersson/gst-python-plugins