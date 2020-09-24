Sample optical flow pipeling:  

    GST_DEBUG=python:4 gst-launch-1.0 pyspinsrc v-binning=2 h-binning=2 ! video/x-raw, format=RGB ! videoconvert ! nvvideoconvert ! "video/x-raw(memory:NVMM)" ! m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720 ! nvof ! queue ! nvofvisual !  nvegltransform ! nveglglessink sync=0 -v