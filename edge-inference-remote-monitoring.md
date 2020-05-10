# Edge Inference with Remote Monitoring and Logging

This experiment combines the responsiveness of inference on an edge device with the visibility of a real-time augmented RTP stream to a remote device.

**Edge Inference device:**  
Nvidia Jetson Nano  

**Remote Monitoring Device:**  
Lenovo Laptop with Intel i5 CPU  

## Edge Device
The Nvidia Jetson Nano device will run the detection.py example and stream the augmented output to the remote device using RTP. 
The detection.py example will load a pyTorch model state and preform object detection on the image stream. It will then augment the image stream by overlaying labelled bounding boxes. The stream will then be sent to the remote monitoring device over RTP.

The Edge device will be configured as a wireless hotspot so we can have a peer to peer Wi-Fi RTP stream. To do this in Ubuntu 18.04:  
1. Click on the network icon in the top right of the top bar -> Create New Wi-Fi network.  
2. Network icon -> Edit Connections. Edit the created network changing Mode to Hotspot. Ensure the the Wi-Fi network card is selected as the Device. Save settings.  
3. Network icon -> Connect to Hidden Wi-Fi Network -> select the new network.

This network should be visible on the **remote** device now. Connect to it from the remote device too. Note the remote device IP. Eg if Wifi interface is wlp2s0:  

    $ ifconfig wlp2s0  

For example lets say the remote device ip address is: 10.42.0.25

To start the stream on the edge device:  

    $ cd gstreamer-pyspin-src
    $ source setup_util.sh
    $ export GST_DEBUG=2,python:4
    $ export RTP_SINK_PIPELINE="nvvidconv ! omxh264enc profile=high insert-sps-pps=true ! video/x-h264, profile=high, stream-format=avc ! rtph264pay pt=96 config-interval=1 ! udpsink host=10.42.0.25 port=5000"
    $ python object_detection/detection.py --frame_rate 10 --binning_level 4 --sink_pipeline "${RTP_SINK_PIPELINE}"

## Remote Device
On the remote device we will view and save the augmented live video stream.  
To view the live stream you can use the following pipeline:  

    $ gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=H264 ! rtph264depay ! h264parse ! vaapih264dec ! videoconvert ! xvimagesink sync=false

To view the live stream and **also** save it to disk you can use:

    $ GST_DEBUG=2 gst-launch-1.0 -v -e udpsrc port=5000 ! application/x-rtp, encoding-name=H264 ! rtph264depay ! h264parse ! tee name=t t. ! queue leaky=1 ! vaapih264dec low-latency=true !  xvimagesink sync=false t. ! queue ! mp4mux ! filesink location="inference_test_stream.mp4"

Note: vaapih264dec can be replaced with avdec_h264 if the remote system does not support vaapi (ie does not have intel graphics hardware).  