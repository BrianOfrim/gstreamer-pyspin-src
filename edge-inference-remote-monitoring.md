# Edge Inference with Remote Monitoring and Logging

This experiment combines the responsiveness of inference on an edge device with the visibility of a real-time augmented RTP stream to a remote device.

Edge Inference device:  
Nvidia Jetson Nano  

Remote Monitoring Device:  
Lenovo Laptop with Intel i5 CPU  

## Edge
The Nvidia Jetson Nano device will run the detection.py example and stream the augmented output to the remote device using RTP.  
The detection.py example will load a pyTorch model state and preform object detection on the image stream. It will then augment the image stream by overlaying labelled bounding boxes. The stream will then be sent to the remote monitoring device over RTP.

The Edge device will be configured as a wireless hotspot so we can have a peer to peer wifi RTP stream.
To do this in Ubuntu 18.04 click on the network icon in the top right of the top bar -> Create New Wi-Fi network. 
Click on the network icon again, click "Edit Connections". Edit the created newtork changing Mode to Hotspot. Ensure the the wifi network card is selected as the Device. Save settings.
Click on the network icon again, click "Connect to Hidden Wi-Fi Network" and select the new network.
Connect to this network on the edge device. You may have to connect to it through "Connect to Hidden Wi-Fi Network"

This network should be visible on the remote device now. Connect to it from the remote device. Note the remote device IP. Eg if wifi interface is wlp2s0:  

    $ ifconfig wlp2s0  

For example lets say the the ip address is: 10.42.0.25

To start the stream on the edge device:  

    $ cd gstreamer-pyspin-src
    $ source setup_util.sh
    $ export GST_DEBUG=2,python:4
    $ export RTP_SINK_PIPELINE="nvvidconv ! omxh264enc profile=high insert-sps-pps=true ! video/x-h264, profile=high, stream-format=avc ! rtph264pay pt=96 config-interval=1 ! udpsink host=10.42.0.25 port=5000"
    $ python examples/detection.py --frame_rate 10 --binning_level 4 --sink_pipeline "${RTP_SINK_PIPELINE}"




## Remote
On the remote device we will view and save the auto
To view the stream as well 
