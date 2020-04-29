# Rough Video Streaming Benchmarks on a Laptop

#### Machine specs
CPU: Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz  
GPU: Intel UHD Graphics 620  
VA driver: LIBVA_DRIVER_NAME=i965  

## Using data from pyspinsrc via a BFS-U3-31S4C-C at 2048x1536, BGRA format, 25fps

### Just Stream:
Pipeline used:

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 ! fakesink -v

Execution ended after 0:00:40.132290674  
CPU%: 49.2/800  
Mem%: 1.7  

### HW encode
Pipeline used:

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 !  videoconvert ! vaapih264enc bitrate=4000 rate-control=cbr ! video/x-h264, profile=high, stream-format=avc ! fakesink -v


Execution ended after 0:00:40.158769206  
CPU%: 58.3/800  
Mem%: 1.8  

### SW encode
Pipeline used:

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 ! videoconvert ! x264enc bitrate=4000 pass=cbr ! video/x-h264, profile=high, stream-format=avc ! fakesink -v

Execution ended after 0:01:06.317899916  
CPU%: 411.3/800  
Mem%: 9.3  

### HW encode with saving to disk
Pipeline used:

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 !  videoconvert ! vaapih264enc bitrate=4000 rate-control=cbr ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location="pyspinsrc_vaapi.mp4" -v

Execution ended after 0:00:40.148467480  
CPU%: 64.5/800  
Mem%: 1.8  
Filesize: 20.1MB  
Video Duration: 0:00:40.004800000  

### SW encode with saving to disk
Pipeline used:

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 ! videoconvert ! x264enc bitrate=4000 pass=cbr ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location="pyspinsrc_xh264.mp4" -v


Execution ended after 0:01:00.237514145  
CPU%: 434.4/800  
Mem%: 9.2  
Filesize: 6.4MB  
Video Duration: 57.046400000 (Frames skipped)  


## Using data from videotestsrc at 2048x1536, RGBA format, 30fps 

### Just Stream
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! fakesink -v

Execution ended after 0:00:02.473099711  
CPU%: 76.4  
Mem%: 0.2  

### HW encode
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! videoconvert ! vaapih264enc bitrate=4000 rate-control=cbr ! video/x-h264, profile=high, stream-format=avc ! fakesink -v

Execution ended after 0:00:14.486787641  
CPU%: 84.7/800  
Mem%: 0.3  

### SW encode
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! videoconvert ! x264enc bitrate=4000  ! video/x-h264, profile=high, stream-format=avc ! fakesink -v

Execution ended after 0:00:29.503188778  
CPU%: 376.4/800  
Mem%: 6.8  

### HW encode with saving to disk
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! videoconvert ! vaapih264enc bitrate=4000 rate-control=cbr ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location="vaapienc.mp4" -v

Execution ended after 0:00:14.912940421  
Filesize: 16.7  
CPU%: 86.0/800  
Mem%: 0.3  

### SW encode with saving to disk
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! videoconvert ! x264enc bitrate=4000  ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location="x264enc.mp4" -v

Execution ended after 0:00:30.524065329  
Filesize: 16.7MB  
CPU%: 375/800  
Mem%: 7.6  
