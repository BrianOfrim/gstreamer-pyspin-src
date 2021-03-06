# GStreamer PySpin Src Plug-in

An unofficial GStreamer source plug in for the PySpin (spinnaker-python) Image Acquisition SDK

![GstreamerSpinnaker](docs/assets/gstreamerSpinnakerMedium.png)

## Install

**The following instructions apply to Ubuntu 18.04**

#### Spinnaker and PySpin

Download the latest version of **Spinnaker** and the matching version of **PySpin** (spinnaker-python) from:  
https://www.flir.com/products/spinnaker-sdk/

Extract and install the Spinnaker SDK package (PySpin will be installed later)

#### System dependencies 

Update package manager sources:  

    sudo apt-get update 

Python:  

    sudo apt-get -y install python3.6 python3-pip python3.6-dev python3.6-venv python3-dev

Build tools:  

    sudo apt-get install -y git build-essential pkg-config libtool autoconf automake libgirepository1.0-dev gir1.2-gstreamer-1.0 libcairo2-dev python-gi-dev

GStreamer:  

    sudo apt-get install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio python3-gst-1.0 libgstreamer-plugins-base1.0-dev

#### GStreamer pyspinsrc
Clone this repo: 
 
    git clone https://github.com/BrianOfrim/gstreamer-pyspin-src.git
    cd gstreamer-pyspin-src

Complete **one** of the 2 following options for installing the dependacnies.  

##### Option 1: Install pip packages in a virtual environment (Recommended)

    python3 -m venv venv
    source venv/bin/activate
    pip install -U wheel pip setuptools Cython
    pip install -r requirements.txt
    pip install <path to pyspin>/spinnaker_python-*.whl

Tell GStreamer where the python plugin loader (libgstpython.*.so) and the pyspinsrc plugins are located:  

    cd gstreamer-pyspin-src # If not already there
    export GST_PLUGIN_PATH=$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/:$GST_PLUGIN_PATH

The previous step will either need to be added to the user's ~/.bashrc file (replace $PWD with absoulute path) or repeated when a new termial is opened

##### Option 2: Install pip packages for the current user, outside of a virtual environment

    python3 -m pip install -U wheel pip setuptools Cython
    GST_PREFIX=$HOME/.local/ python3 -m pip install -r requirements.txt
    python3 -m pip install <path to pyspin>/spinnaker_python-*.whl

Note: This should result in the gstreamer python plugin loader (libgstpython.*.so) being installed at $HOME/.local/lib/gstreamer-1.0/  
If the variable GST_PREFIX is not set then the plugin loader will be installed at /usr/lib/gstreamer-1.0/   
Doing so would require the pip installtion of requirements.txt to be done as sudo which is not recommended as it may lead to problems.  

Tell GStreamer where the python plugin loader (libgstpython.*.so) and the pyspinsrc plugins are located:  

    cd gstreamer-pyspin-src # If not already there
    export GST_PLUGIN_PATH=$HOME/.local/lib/gstreamer-1.0/:$PWD/gst/:$GST_PLUGIN_PATH

The previous step will either need to be added to the user's ~/.bashrc file (replace $PWD with absoulute path) or repeated when a new termial is opened

## Usage

Clear the GStreamer registry cache:  

    sudo rm -rf ~/.cache/gstreamer-1.0/

Verify that the gstreamer python plugin loader can be found and instepected with no errors:  
    
    gst-inspect-1.0 python

Verify that the pyspinsrc element can be found and instepected with no errors:  

    gst-inspect-1.0 pyspinsrc

Example pileline:  

    gst-launch-1.0 pyspinsrc ! videoconvert ! xvimagesink sync=false

Example debugging pipeline:  

    GST_DEBUG=python:6 gst-launch-1.0 --gst-disable-segtrap --gst-disable-registry-fork pyspinsrc ! videoconvert ! xvimagesink sync=false

## Benchmarks
Streaming and saving video on an Nvidia Jetson Nano using hardware or software video encoding: [Jetson Nano Benchmarks](nvidia-jetson-nano-benchmarks.md)

Streaming and saving video on an Laptop with an Intel i5 CPU using hardware (vaapi) or software encoding: [VAAPI Benchmarks](vaapi-benchmarks.md)

## Use Cases
Streaming video over a network using HLS, and RTP: [Network Streaming](local-network-streaming.md)

Preforming object detection on an edge device and streaming an augmented video feed to a remote monitoring device: [Detection Stream](edge-inference-remote-monitoring.md)


## Examples
Examples require Pillow, matplotlib, svgwrite, pytorch and torchvision. Install them with pip.

Note: for arm platforms torch and torchvision may not be available through pip. So you may need to find an alternative installation method. For example with Nvidia Jetson devices follow the instructions here: https://elinux.org/Jetson_Zoo 

### Overlay Examples
The following examples use the same pipeline created in [gst_overlay_pipeline.py](applications/gst_overlay_pipeline.py) and supply different image processing functions.

Example Application pipeline:

![ApplicationPipeline](docs/assets/OverlayPipeline.jpg)

#### Object detection
Location: **applications/detection.py**  
Recycling detection trained with the [Boja](https://github.com/BrianOfrim/boja) process  
![Detection](docs/assets/RecyclingDetection.jpg)  

#### Face Detection
Location: **applications/face-detection.py**  
Face Detection model from https://github.com/timesler/facenet-pytorch  
Requires python module: facenet-pytorch (pip install facenet-pytorch)  
![FaceDetection](docs/assets/FaceDetection.jpg)  

#### Classification
Location: **applications/classification.py**  
Apply a torchvision pretrained classification model  
![Classification](docs/assets/ReggieClassification.jpg)  

#### Segmentation
Location: **applications/segmentation.py**  
Apply a torchvision pretrained segmentation model  
![Segmentation](docs/assets/ReggieSegmentation.jpg)

#### Detr Detection
Location: **applications/detr-detection.py**  
Apply a pretrained [detr](https://github.com/facebookresearch/detr) detection model  


### Simple Display Examples
The following examples use the same pipeline created in [gst_appsink_display.py](applications/gst_appsink_display.py) and supply different image processing functions.

#### Face Mask
Location: **applications/face-mask.py**  
Draw filled boxes over detected faces in order to hide identity  
![FaceMask](docs/assets/OfficeFaceMask.jpg)

#### Relative Depth
Location: **applications/face-mask.py**  
Apply [MiDaS](https://github.com/intel-isl/MiDaS) for relative depth estimation  
![RelativeDepth](docs/assets/OfficeRelativeDepth.jpg)

### Appsrc -> process -> Appsink Examples
The following examples use two pipelines created in [gst_app_src_and_sink.py](applications/gst_app_src_and_sink.py). Images are produced via a pipeline ending with an [appsink](https://gstreamer.freedesktop.org/documentation/app/appsink.html) element, processed with a user application, then consumed by a pipeline starting with an [appsrc](https://gstreamer.freedesktop.org/documentation/app/appsrc.html) element.

#### Human Pose Detection
Location: **applications/human-pose.py**  
Apply human pose detection from [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose). If a cuda device is detected it will optimize and save a TensorRT model.  
![HumanPose](docs/assets/OfficeHumanPose.jpg)

#### Face Recognition Tracking
Location: **applications/face-recognition-tracking.py**  
Uses FaceNet to obtain facial embeddings and a DBSCAN clustering algorithm to cluster the embeddings and track individual faces across a video.  
![FaceRecognitionTracking](docs/assets/annotate-cut.gif)


## References
Uses the following for gst buffer to numpy mapping utilities and to install [gst-python](https://github.com/GStreamer/gst-python):  
https://github.com/jackersson/gstreamer-python

Object Detection example inspired by:  
https://github.com/google-coral/examples-camera/tree/master/gstreamer
