## Using Youtube as a video source

Install youtube-dl which is used to get the video url
    
    pip install youtube-dl

Export the video portion of the pipeline to an environment variable

    export youtube_src_bin="souphttpsrc location=\"$(youtube-dl --format \\"best[ext=mp4][protocol=https]\\" --get-url https://www.youtube.com/watch?v=czWf1vbNwoQ)\" ! decodebin ! videoconvert"

Export the file sink portion of the pipeline to an environment variable
    
    export video_sink_bin="queue ! x264enc ! video/x-h264, profile=high, stream-format=avc ! h264parse ! mp4mux ! filesink location=annotate.mp4"

Start a face recognition and tracking pipeline using the above source and sink bins

    python face-recognition-tracking.py --image_src_bin "$youtube_src_bin" --image_sink_bin "$video_sink_bin" --full_queue --face_threshold 0.9
