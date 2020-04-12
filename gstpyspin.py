
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")


from gi.repository import Gst, GObject, GLib, GstBase, GstVideo

from gstreamer.utils import gst_buffer_with_pad_to_ndarray

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

DEFAULT_AUTO_EXPOSURE = True
DEFAULT_EXPOSURE_TIME = 15000
DEFAULT_AUTO_GAIN = True
DEFAULT_GAIN = 0.0

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

    __gproperties__ = {
        "exposure-auto": (
            bool,
            "automatic exposure",
            "Enabling automatic exposure timing",
            DEFAULT_AUTO_EXPOSURE,
            GObject.ParamFlags.READWRITE
            ),
        "exposure-time": (
            int,
            "exposure time",
            "Exposure time in microsecods",
            1,
            GLib.MAXINT,
            DEFAULT_EXPOSURE_TIME,
            GObject.ParamFlags.READWRITE,
        ),
        "gain-auto": (
            bool,
            "automatic gain",
            "Enabling automatic gain",
            DEFAULT_AUTO_GAIN,
            GObject.ParamFlags.READWRITE
            ),
        "gain": (
            float,
            "gain",
            "Gain in decibels",
            0.0,
            100.0,
            DEFAULT_GAIN,
            GObject.ParamFlags.READWRITE,
        ),
        "serial": (
            str,
            "serial number",
            "The camera serial number",
            None,
            GObject.ParamFlags.READWRITE
        ),

    }

    # GST function
    def __init__(self):
        super(PySpinSrc, self).__init__()

        # Initialize properties before Base Class initialization
        self.info = GstVideo.VideoInfo()
        
        # Properties
        self.auto_exposure = DEFAULT_AUTO_EXPOSURE
        self.exposure_time = DEFAULT_EXPOSURE_TIME
        self.auto_gain = DEFAULT_AUTO_GAIN
        self.gain = DEFAULT_GAIN
        self.serial = None

        # Spinnaker objects
        self.system = None
        self.cam_list = None
        self.cam = None

        # Buffer timing
        self.timestamp_offset = 0
        self.previous_timestamp = 0
        
        # Base class proprties
        self.set_live(True)
        self.set_format(Gst.Format.TIME)

    
    def initialize_cam(self) -> bool:
        try:
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            # Finish if there are no cameras
            if self.cam_list.GetSize() == 0:
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                print("No cammeras detected.")
                return False

            if(self.serial is None):
                # No serial provided retrieve the first available camera
                self.cam = self.cam_list.GetByIndex(0)
            else:
                self.cam = self.cam_list.GetBySerial(self.serial)

            if not self.cam or self.cam is None:
                print("Could not retrieve camera from camera list.")
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                return False

            self.cam.Init()
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True

    # Camera helper function
    def apply_default_settings(self)-> bool:
        try:
            self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
            self.cam.UserSetLoad()
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True

    # Camera helper function
    def apply_caps_to_cam(self) -> bool:
        print("Applying caps.")
        try:
            # Apply Caps
            self.cam.Width.SetValue(self.info.width)
            print("Width: ", self.cam.Width.GetValue())

            self.cam.Height.SetValue(self.info.height)
            print("Height: ", self.cam.Height.GetValue())

            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
            print("Pixel format: ", self.cam.PixelFormat.GetCurrentEntry().GetSymbolic())

            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.AcquisitionFrameRate.SetValue( self.info.fps_n / self.info.fps_d)
            print("Frame rate: ", self.cam.AcquisitionFrameRate.GetValue())

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False
        return True

    # Camera helper function
    def apply_properties_to_cam(self) -> bool:
        print("Applying properties.")
        try:
            # Apply Properties
            if self.auto_exposure:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
                print("Auto Exposure: ", self.cam.ExposureAuto.GetValue())
            else:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                print("Auto Exposure: ", self.cam.ExposureAuto.GetValue())
                self.cam.ExposureTime.SetValue(self.exposure_time)
                print("Exposure time: ", self.cam.ExposureTime.GetValue() ,"us")

            if self.auto_gain:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
                print("Auto Gain: ", self.cam.GainAuto.GetValue())
            else:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                print("Auto Gain: ", self.cam.GainAuto.GetValue())
                self.cam.GainAuto.SetValue(self.gain)
                print("Gain: ", self.cam.Gain.GetValue() ,"db")     

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True

    # Camera helper function
    def start_streaming(self) -> bool:

        # Ensure that acquisition is stopped before applying settings
        try:
            self.cam.AcquisitionStop()
        except PySpin.SpinnakerException as ex:
            print("Acquisition stopped to apply settings")

        if not self.apply_default_settings():
            return False

        if not self.apply_caps_to_cam():
            return False

        if not self.apply_properties_to_cam():
            return False

        try:
            self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.cam.BeginAcquisition()     
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False
        
        print("Streaming started.")

        return True
    
    # GST function
    def do_set_caps(self, caps: Gst.Caps) -> bool:
        self.info.from_caps(caps)    
        self.set_blocksize(self.info.size)

        return self.start_streaming()

    # GST function
    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "exposure-auto":
            return self.auto_exposure 
        elif prop.name == "exposure-time":
            return self.exposure_time
        if prop.name == "gain-auto":
            return self.auto_gain 
        elif prop.name == "gain":
            return self.gain
        elif prop.name == 'serial':
            return self.serial
        else:
            raise AttributeError("unknown property %s" % prop.name)

    # GST function
    def do_set_property(self, prop: GObject.GParamSpec, value):
        print("Setting propery")
        if prop.name == "exposure-auto":
            self.auto_exposure = value
        elif prop.name == 'exposure-time':
            self.exposure_time = value
        elif prop.name == 'gain-auto':
            self.auto_gain = value 
        elif prop.name == 'gain':
            self.gain = value 
        elif prop.name == 'serial':
            self.serial = value 
        else:
            raise AttributeError("unknown property %s" % prop.name)

    # GST function
    def do_start(self):
        print("Starting")
        return self.initialize_cam()

    # GST function
    def do_stop(self):
        print("Stopping")
        self.cam.EndAcquisition()
        self.cam.DeInit()

        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

        return True

    # GST function
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


    # GST function
    def do_gst_push_src_fill(self, buffer: Gst.Buffer) -> Gst.FlowReturn:

        spinnaker_image = self.cam.GetNextImage()
        image_timestamp = spinnaker_image.GetTimeStamp() 
        image = gst_buffer_with_pad_to_ndarray(buffer, self.srcpad)
        image[:] = spinnaker_image.GetNDArray()
        spinnaker_image.Release() 

        if(self.timestamp_offset == 0):
            self.timestamp_offset = image_timestamp
            self.previous_timestamp = image_timestamp 

        buffer.pts = image_timestamp - self.timestamp_offset
        buffer.duration = image_timestamp - self.previous_timestamp

        self.previous_timestamp = image_timestamp

        return Gst.FlowReturn.OK

# Register plugin to use it from command line
GObject.type_register(PySpinSrc)
__gstelementfactory__ = (PySpinSrc.GST_PLUGIN_NAME, Gst.Rank.NONE, PySpinSrc)

