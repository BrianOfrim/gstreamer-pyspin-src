## HLS (HTTP Live Streaming)

https://en.wikipedia.org/wiki/HTTP_Live_Streaming

### Server

The following pipeline will write video segments to the current directory as .ts files. It will curate a playlist of the most recent segments.
 
    GST_DEBUG=2,python:4 gst-launch-1.0 pyspinsrc v-binning=4 h-binning=4 ! video/x-raw, format=BGRA !  videoconvert ! vaapih264enc bitrate=4000 rate-control=cbr ! video/x-h264, profile=high ! h264parse ! mpegtsmux ! hlssink target-duration=5 -v

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

## TODO: RTP/RTSP


## TODO: MULTICAST