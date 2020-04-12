
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")


from gi.repository import Gst, GObject, GLib, GstBase, GstVideo

from gstreamer.utils import gst_buffer_with_pad_to_ndarray

DEFAULT_EXPOSURE_TIME = 15000

try:
    import numpy as np
except ImportError:
    Gst.error('pyspinsrc requires numpy')
    raise

try:
    import PySpin
except ImportError:
    Gst.error('pyspinsrc requires PySpin')
    raise

Gst.init(None)

OCAPS = Gst.Caps(
    Gst.Structure(
        "video/x-raw",
        format="BGR",
        width=720,
        height=540,
        framerate=Gst.Fraction(10, 1),
    )
)


class PySpinSrc(GstBase.PushSrc):

    GST_PLUGIN_NAME = "pyspinsrc"

    __gstmetadata__ = ("pyspinsrc", "Src", "PySpin src element", "Brian Ofrim")
    
    __gsttemplates__ = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, OCAPS,
    )

    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        "exposure": (
            int,
            "exposure time",
            "Exposure time in microsecods",
            1000,
            1000000000,
            DEFAULT_EXPOSURE_TIME,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super(PySpinSrc, self).__init__()

        # Initialize properties before Base Class initialization
        self.info = GstVideo.VideoInfo()
        self.exposure = DEFAULT_EXPOSURE_TIME

        self.frame = 0
        
        self.set_live(True)
        self.set_format(Gst.Format.TIME)
    

    def do_set_caps(self, caps):
        print("set caps")
        self.info.from_caps(caps)    
        self.set_blocksize(self.info.size)

        try:
            self.cam.Width.SetValue(self.info.width)
            print("Width: ", self.cam.Width.GetValue())
            self.cam.Height.SetValue(self.info.height)
            print("Height: ", self.cam.Height.GetValue())
            #self.cam.PixelFormat.SetValue(PySpin.PixelFormat_RGB8)
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
            print("Pixel format: ", self.cam.PixelFormat.GetCurrentEntry().GetSymbolic())

            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.AcquisitionFrameRate.SetValue( self.info.fps_n / self.info.fps_d)
            print("Frame rate: ", self.cam.AcquisitionFrameRate.GetValue())

            self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            print("Acquisition mode: ", self.cam.AcquisitionMode.GetCurrentEntry().GetSymbolic())
             
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        self.cam.BeginAcquisition()
        return True

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "exposure":
            return self.exposure
        else:
            raise AttributeError("unknown property %s" % prop.name)

    def do_start(self):
        print("Start")
            # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()

        print("Number of cameras detected: %d" % num_cameras)
        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            self.cam_list.Clear()

            # Release system instance
            self.system.ReleaseInstance()

            print("Not enough cameras!")
            return False

        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()

        self.pts = 0
        return True

    def do_stop(self):
        print("stop")
        self.cam.EndAcquisition()
        self.cam.DeInit()

        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

        return True

    def do_get_times(self, buf):

        end = 0
        start = 0
        if self.is_live:
            ts = buf.pts
            if ts != Gst.CLOCK_TIME_NONE:
                duration = buf.duration
                if duration != Gst.CLOCK_TIME_NONE:
                    end = ts + duration
                start = ts
        else:
            start = Gst.CLOCK_TIME_NONE
            end = Gst.CLOCK_TIME_NONE

        return start, end

    #@vfunc(GstBase.PushSrc)
    def do_gst_push_src_fill(self, buffer: Gst.Buffer) -> Gst.FlowReturn:
        print(buffer.get_size())
        print("Try sending data")
        duration = 10**9 / self.info.fps_n / self.info.fps_d

        start = (self.frame * 10) % 680

        spinnaker_image = self.cam.GetNextImage()
        image = gst_buffer_with_pad_to_ndarray(buffer, self.srcpad)
        image[:] = spinnaker_image.GetNDArray()
        # image[:] = image[...,::-1]
        spinnaker_image.Release() 

        # set pts and duration to be able to record video, calculate fps
        self.pts += duration  # Increase pts by duration
        buffer.pts = self.pts
        buffer.duration = duration

        # return (Gst.FlowReturn.OK, buffer)

        return Gst.FlowReturn.OK

    def do_set_property(self, prop: GObject.GParamSpec, value):
        print("Setting propery")
        if prop.name == "exposure":
            self.exposure = value
        else:
            raise AttributeError("unknown property %s" % prop.name)


# Register plugin to use it from command line
GObject.type_register(PySpinSrc)
__gstelementfactory__ = (PySpinSrc.GST_PLUGIN_NAME, Gst.Rank.NONE, PySpinSrc)

