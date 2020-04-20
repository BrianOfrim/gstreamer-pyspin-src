# GStreamer PySpin Src Plug-in

A GStreamer source plug in for the PySpin Image Acquisition SDK

## Install

Download the latest version of **Spinnaker** and the matching version of **PySpin** (spinnaker-python) from:  
https://flir.app.box.com/v/SpinnakerSDK/

Install some dependancies:  

    sudo apt-get install gstreamer-1.0
    sudo apt-get install gstreamer1.0-dev
    sudo apt-get install python3.6 python3.6-dev python-dev python3-dev
    sudo apt-get install python3-pip python-dev 
    sudo apt-get install python3.6-venv
    sudo apt-get install git autoconf automake libtool
    sudo apt-get install python3-gi python-gst-1.0 
    sudo apt-get install libgirepository1.0-dev
    sudo apt-get install libcairo2-dev gir1.2-gstreamer-1.0
    sudo apt-get install python-gi-dev

Clone this repo and install dependencies: 
 
    git clone https://github.com/BrianOfrim/gstreamer-pyspin-src.git
    cd gstreamer-pyspin-src

    python3 -m venv venv
    source venv/bin/activate
    pip install -U wheel pip setuptools

    pip install -r requirements.txt
    pip install <path-to-pyspin-package>\spinnaker_python-2.x.x.x-cp36-cp36m-linux_x86_64.whl

Verify that hte installation was successful and view plug-in info and properties:

    gst-inspect-1.0 pyspinsrc

## Usage

Tell GStreamer where our plugin is located:

    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/

Example debugging pipeline:

    GST_DEBUG=python:6 gst-launch-1.0 --gst-disable-segtrap --gst-disable-registry-fork pyspinsrc ! videoconvert ! autovideosink

Example pileline:

    gst-launch-1.0 pyspinsrc ! videoconvert  ! autovideosink



## References

Uses the following for gst buffer to numpy mapping utilities and to install [gst-python](https://github.com/GStreamer/gst-python):  
https://github.com/jackersson/gstreamer-python

Project based on:  
https://github.com/jackersson/gst-python-plugins
