# Rough Benchmarks on a Nvidia Jetson Nano

## Using videotestsrc

#### Just Stream
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! fakesink -v

Execution ended after 0:00:14.072180679  
CPU%: 98.7/400  
Mem%: 0.9  

#### HW encode
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! nvvidconv ! omxh264enc profile=high bitrate=4000000  ! video/x-h264, profile=high, stream-format=avc ! fakesink -v

Execution ended after 0:00:21.514255422  
CPU%: 89.5/400  
Mem%: 1.0  

#### SW encode
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! videoconvert ! x264enc bitrate=4000  ! video/x-h264, profile=high, stream-format=avc ! fakesink -v

Execution ended after 0:02:20.086161015  
CPU%: 306.6/400  
Mem%: 23.0  

#### HW encode with saving to disk
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! nvvidconv ! omxh264enc profile=high bitrate=4000000  ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location="omxenc.mp4" -v
Execution ended after 0:00:21.846840130  
Filesize: 13.0MB  
CPU%: 90.5/400  
Mem%: 1.1  

#### SW encode with saving to disk
Pipeline used:

    GST_DEBUG=2 gst-launch-1.0 videotestsrc num-buffers=1000 ! video/x-raw, width=2048, height=1536, format=RGBA, framerate=30/1 ! videoconvert ! x264enc bitrate=4000  ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location="x264enc.mp4" -v

Execution ended after 0:02:23.639391072  
Filesize: 16.7MB  
CPU%: 304.6/400  
Mem%: 23.0  

## Using pyspinsrc with a BFS-U3-31S6C-C

#### Just Stream:
Pipeline used:

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 ! fakesink -v

Execution ended after 0:00:40.191647241  
CPU%: 67.7/400  
Mem%: 6.6  

#### HW encode
Pipeline used:

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 ! videoconvert ! video/x-raw, format=BGRx ! nvvidconv ! omxh264enc profile=high bitrate=4000000  ! video/x-h264, profile=high, stream-format=avc ! fakesink -v

Execution ended after 0:00:40.221865195  
CPU%: 106.9  
Mem%: 7.0  

#### SW encode
Pipeline used:

    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 ! videoconvert ! x264enc bitrate=4000  ! video/x-h264, profile=high, stream-format=avc ! fakesink -v

Execution ended after 0:05:12.336596995  
CPU%:305.9/400  
Mem%:29.7  

#### HW encode with saving to disk
Pipeline used:

GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 ! videoconvert ! video/x-raw, format=BGRx ! nvvidconv ! omxh264enc profile=high bitrate=4000000  ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location="pyspinsrc_omx.mp4" -v

Execution ended after 0:00:40.221191910  
CPU%: 26.5/400  
Mem%: 7.2  
Filesize: 17.8MB  

#### SW encode with saving to disk
Pipeline used:

GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc num-buffers=1000 exposure=10000 auto-exposure=false ! video/x-raw, width=2048, height=1536, format=BGRA, framerate=25/1 ! videoconvert ! x264enc bitrate=4000 ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location="pyspinsrc_xh264.mp4" -v

Execution ended after 0:04:41.196435568  
CPU%:311.6  
Mem%:29.8   
Filesize: 9MB  
Lots of Skipped frames, video is ~4min 40sec  


