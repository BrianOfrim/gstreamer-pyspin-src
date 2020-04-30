## HLS (HTTP Live Streaming)

https://en.wikipedia.org/wiki/HTTP_Live_Streaming

Pros:
 - Widely used and supported
 - HTTP is as not blocked by many firewalls
 - Simple

Cons:
 - High latency

### Server

The following pipeline will write video segments to the current directory as .ts files. It will curate a playlist of the most recent segments.
 
    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc v-binning=4 h-binning=4 ! video/x-raw, format=BGRA !  videoconvert ! vaapih264enc bitrate=4000 rate-control=cbr ! video/x-h264, profile=high ! h264parse ! mpegtsmux ! hlssink target-duration=5 -v

If vaapih264enc is not available it can be swapped with a software h264 encoder such as x264enc.

To serve these video segments we will start the following simple http server in the directory where the video playlist and segments are stored.

    python3 -m http.server

Find your address on the LAN with ifconfig or similar.  
For example say your IP is 192.168.1.68  

### Client

#### VLC
Open VLC -> Media -> Open Network Stream  
Then enter the target url of the playlist. Eg http://192.168.1.68:8000/playlist.m3u8 then click play  

#### Browser
Use a browser that supports HLS streams. https://en.wikipedia.org/wiki/HTTP_Live_Streaming#Clients  
Desktop Chrome and Firefox do not.  
Desktop Edge and Safari do. Mobile Safari, Chrome and Firefox do.  
Then enter the target url of the playlist. Eg http://192.168.1.68:8000/playlist.m3u8  

## RTP (Real-time Transport Protocol)
https://en.wikipedia.org/wiki/Real-time_Transport_Protocol

Pros:
 - Low latency

Cons:
 - Network permissions needed are more complex

### Server


    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc v-binning=4 h-binning=4 ! video/x-raw, format=BGRA !  videoconvert ! vaapih264enc ! video/x-h264, profile=high ! rtph264pay ! udpsink host=127.0.0.1 port=5000

Using the the localhost (127.0.0.1) as the host in the above example.
If vaapih264enc is not available it can be swapped with a software h264 encoder such as x264enc.

### Client

#### GStreamer client pipeline

    GST_DEBUG=2 gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=H264 ! rtph264depay ! h264parse ! vaapih264dec ! videoconvert ! xvimagesink sync=false

If vaapih264dec is not available it can be swapped with a software h264 decoder such as avdec_h264.


#### VLC

Modified server (not as efficient as above but vlc can understand it more easily):

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc v-binning=4 h-binning=4 ! video/x-raw, format=BGRA !  videoconvert ! vaapih264enc ! video/x-h264, profile=high ! mpegtsmux ! rtpmp2tpay ! udpsink host=127.0.0.1 port=5000

Open VLC -> Media -> Open Network Stream  
Then enter the host url. Eg rtp://@127.0.0.1:5000 then click play 
