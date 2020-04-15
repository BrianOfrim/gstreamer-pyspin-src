import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")


from gi.repository import Gst, GObject, GLib, GstBase, GstVideo

from gstreamer.utils import gst_buffer_with_pad_to_ndarray

try:
    import numpy as np
except ImportError:
    Gst.error("pyspinsrc requires numpy")
    raise

try:
    import PySpin
except ImportError:
    Gst.error("pyspinsrc requires PySpin")
    raise

DEFAULT_WIDTH = 720
DEFAULT_HEIGHT = 540
DEFAULT_FRAME_RATE = 10

DEFAULT_H_BINNING = 1
DEFAULT_V_BINNING = 1
DEFAULT_AUTO_EXPOSURE = True
DEFAULT_EXPOSURE_TIME = 15000
DEFAULT_AUTO_GAIN = True
DEFAULT_GAIN = 0.0
DEFAULT_NUM_BUFFERS = 10

MILLIESCONDS_PER_NANOSECOND = 1000000
TIMEOUT_MS = 2000

OCAPS = Gst.Caps(
    Gst.Structure(
        "video/x-raw",
        format="BGR",
        width=Gst.IntRange(range(1, GLib.MAXINT)),
        height=Gst.IntRange(range(1, GLib.MAXINT)),
        framerate=Gst.FractionRange(Gst.Fraction(1, 1), Gst.Fraction(GLib.MAXINT, 1)),
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
            GObject.ParamFlags.READWRITE,
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
            GObject.ParamFlags.READWRITE,
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
        "h-binning": (
            int,
            "horizontal binning",
            "Horizontal average binning (applied before width and offset x)",
            1,
            GLib.MAXINT,
            DEFAULT_H_BINNING,
            GObject.ParamFlags.READWRITE,
        ),
        "v-binning": (
            int,
            "vertical binning",
            "Vertical average binning (applied before height and offset y)",
            1,
            GLib.MAXINT,
            DEFAULT_V_BINNING,
            GObject.ParamFlags.READWRITE,
        ),
        "num-buffers": (
            int,
            "number of image buffers",
            "Number of buffers for Spinnaker to allocate for buffer handling",
            1,
            GLib.MAXINT,
            DEFAULT_EXPOSURE_TIME,
            GObject.ParamFlags.READWRITE,
        ),
        "serial": (
            str,
            "serial number",
            "The camera serial number",
            None,
            GObject.ParamFlags.READWRITE,
        ),
    }

    # GST function
    def __init__(self):
        super(PySpinSrc, self).__init__()

        # Initialize properties before Base Class initialization
        self.info = GstVideo.VideoInfo()

        # Properties
        self.auto_exposure: bool = DEFAULT_AUTO_EXPOSURE
        self.exposure_time: int = DEFAULT_EXPOSURE_TIME
        self.auto_gain: bool = DEFAULT_AUTO_GAIN
        self.gain: float = DEFAULT_GAIN
        self.h_binning: int = DEFAULT_H_BINNING
        self.v_binning: int = DEFAULT_V_BINNING
        self.serial: str = None
        self.num_cam_buffers: int = DEFAULT_NUM_BUFFERS

        # Spinnaker objects
        self.system: PySpin.System = None
        self.cam_list: PySpin.CameraList = None
        self.cam: PySpin.Camera = None

        # Buffer timing
        self.timestamp_offset: long = 0
        self.previous_timestamp: long = 0

        # Base class proprties
        self.set_live(True)
        self.set_format(Gst.Format.TIME)

    # Camera helper function
    def apply_default_settings(self) -> bool:
        try:
            self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
            self.cam.UserSetLoad()
        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False

        return True

    # Camera helper function
    # ret val= Height: int, OffsetY: int, Width: int, OffsetX int
    def get_roi(self) -> (int, int, int, int):
        return (
            self.cam.Height.GetValue(),
            self.cam.OffsetY.GetValue(),
            self.cam.Width.GetValue(),
            self.cam.OffsetX.GetValue(),
        )

    # Camera helper function
    def apply_caps_to_cam(self) -> bool:
        Gst.info("Applying caps.")
        try:

            # Apply Caps
            self.cam.Width.SetValue(self.info.width)
            Gst.info(f"Width: {self.cam.Width.GetValue()}")

            self.cam.Height.SetValue(self.info.height)
            Gst.info(f"Height: {self.cam.Height.GetValue()}")

            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
            Gst.info(
                f"Pixel format: {self.cam.PixelFormat.GetCurrentEntry().GetSymbolic()}"
            )

            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.AcquisitionFrameRate.SetValue(self.info.fps_n / self.info.fps_d)
            Gst.info(f"Frame rate: {self.cam.AcquisitionFrameRate.GetValue()}")

        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False
        return True

    # Camera helper funtion
    def apply_properties_to_transport_layer(self) -> bool:
        try:
            # Configure Transport Layer Properties
            self.cam.TLStream.StreamBufferHandlingMode.SetValue(
                PySpin.StreamBufferHandlingMode_OldestFirst
            )
            self.cam.TLStream.StreamBufferCountMode.SetValue(
                PySpin.StreamBufferCountMode_Manual
            )
            self.cam.TLStream.StreamBufferCountManual.SetValue(self.num_cam_buffers)

            Gst.info(
                f"Buffer Handling Mode: {self.cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetSymbolic()}"
            )
            Gst.info(
                f"Buffer Count Mode: {self.cam.TLStream.StreamBufferCountMode.GetCurrentEntry().GetSymbolic()}"
            )
            Gst.info(
                f"Buffer Count: {self.cam.TLStream.StreamBufferCountManual.GetValue()}"
            )
            Gst.info(
                f"Max Buffer Count: {self.cam.TLStream.StreamBufferCountManual.GetMax()}"
            )

        except PySpin.SpinnakerException as ex:

            Gst.error(f"Error: {ex}")
            return False
        return True

    # Camera helper function
    def apply_properties_to_cam(self) -> bool:
        Gst.info("Applying properties")
        try:
            # Configure Camera Properties

            # Apply binning before caps
            if self.h_binning > 1:
                self.cam.BinningHorizontal.SetValue(self.h_binning)
                self.cam.BinningHorizontalMode.SetValue(
                    PySpin.BinningHorizontalMode_Average
                )
                Gst.info(f"Horizontal Binning: {self.cam.BinningHorizontal.GetValue()}")

            if self.v_binning > 1:
                self.cam.BinningVertical.SetValue(self.v_binning)
                self.cam.BinningVerticalMode.SetValue(
                    PySpin.BinningVerticalMode_Average
                )
                Gst.info(f"Vertical Binning: {self.cam.BinningVertical.GetValue()}")

            if self.auto_exposure:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
                Gst.info(
                    f"Auto Exposure: {self.cam.ExposureAuto.GetCurrentEntry().GetSymbolic()}"
                )
            else:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                self.cam.ExposureTime.SetValue(self.exposure_time)
                Gst.info(
                    f"Auto Exposure: {self.cam.ExposureAuto.GetCurrentEntry().GetSymbolic()}"
                )
                Gst.info(f"Exposure Time: {self.cam.ExposureTime.GetValue()}us")

            if self.auto_gain:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
                Gst.info(
                    f"Auto Gain: {self.cam.GainAuto.GetCurrentEntry().GetSymbolic()}"
                )
            else:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                self.cam.GainAuto.SetValue(self.gain)
                Gst.info(
                    f"Auto Gain: {self.cam.GainAuto.GetCurrentEntry().GetSymbolic()}"
                )
                Gst.info(f"Gain: {self.cam.Gain.GetValue()}db")

        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False

        return True

    # Camera helper function
    def start_streaming(self) -> bool:

        if not self.apply_caps_to_cam():
            return False

        try:
            self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.cam.BeginAcquisition()
        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False

        Gst.info("Acquisition Started")

        return True

    # Camera helper function
    def init_cam(self) -> bool:
        try:
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            # Finish if there are no cameras
            if self.cam_list.GetSize() == 0:
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                Gst.error("No cameras detected")
                return False

            if self.serial is None:
                # No serial provided retrieve the first available camera
                self.cam = self.cam_list.GetByIndex(0)
                Gst.info(
                    f"No serial number provided"
                    f"Using camera: {self.cam.TLDevice.DeviceSerialNumber.GetValue()}"
                )
            else:
                self.cam = self.cam_list.GetBySerial(self.serial)
                Gst.info(
                    f"Using camera: {self.cam.TLDevice.DeviceSerialNumber.GetValue()}"
                )

            if not self.cam or self.cam is None:
                Gst.error("Could not retrieve camera from camera list.")
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                return False

            self.cam.Init()

            # Ensure that acquisition is stopped before applying settings
            try:
                self.cam.AcquisitionStop()
            except PySpin.SpinnakerException as ex:
                Gst.info("Acquisition stopped to apply settings")

            if not self.apply_default_settings():
                return False

            if not self.apply_properties_to_cam():
                return False

            if not self.apply_properties_to_transport_layer():
                return False

        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False

        return True

    # Camera helper function
    def deinit_cam(self) -> bool:
        try:
            self.cam.EndAcquisition()
            Gst.info("Acquisition Ended")
            self.cam.DeInit()
            del self.cam
            self.cam_list.Clear()
            self.system.ReleaseInstance()

        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False

        return True

    # GST function
    def do_set_caps(self, caps: Gst.Caps) -> bool:
        self.info.from_caps(caps)
        self.set_blocksize(self.info.size)

        return self.start_streaming()

    # GST function
    def do_fixate(self, caps: Gst.Caps):
        Gst.info("Fixating caps")

        try:
            current_cam_height, _, current_cam_width, _ = self.get_roi()
        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            # Error reading camera roi settings, just use default values
            current_cam_width = DEFAULT_WIDTH
            current_cam_height = DEFAULT_HEIGHT

        try:
            frame_rate = self.cam.AcquisitionFrameRate.GetValue()
        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            # Error reading camera framerate, just use default values
            frame_rate = DEFAULT_FRAME_RATE

        structure = caps.get_structure(0).copy()
        structure.fixate_field_nearest_int("width", current_cam_width)
        structure.fixate_field_nearest_int("height", current_cam_height)
        # structure.fixate_field_nearest_fraction("framerate", frame_rate + 0.5, 1)
        structure.fixate_field_nearest_fraction("framerate", frame_rate, 1)

        new_caps = Gst.Caps.new_empty()
        new_caps.append_structure(structure)
        return new_caps.fixate()

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
        elif prop.name == "h-binning":
            return self.h_binning
        elif prop.name == "v-binning":
            return self.v_binning
        elif prop.name == "num-buffers":
            return self.num_cam_buffers
        elif prop.name == "serial":
            return self.serial
        else:
            raise AttributeError("unknown property %s" % prop.name)

    # GST function
    def do_set_property(self, prop: GObject.GParamSpec, value):
        Gst.info(f"Setting {prop.name} = {value}")
        if prop.name == "exposure-auto":
            self.auto_exposure = value
        elif prop.name == "exposure-time":
            self.exposure_time = value
        elif prop.name == "gain-auto":
            self.auto_gain = value
        elif prop.name == "gain":
            self.gain = value
        elif prop.name == "h-binning":
            self.h_binning = value
        elif prop.name == "v-binning":
            self.v_binning = value
        elif prop.name == "num-buffers":
            self.num_cam_buffers = value
        elif prop.name == "serial":
            self.serial = value
        else:
            raise AttributeError("unknown property %s" % prop.name)

    # GST function
    def do_start(self):
        Gst.info("Starting")
        return self.init_cam()

    # GST function
    def do_stop(self):
        Gst.info("Stopping")
        return self.deinit_cam()

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

        spinnaker_image = None

        while spinnaker_image is None or spinnaker_image.IsIncomplete():

            # Grab a buffered image from the camera
            try:
                spinnaker_image = self.cam.GetNextImage(TIMEOUT_MS)
            except PySpin.SpinnakerException as ex:
                Gst.error(f"Error: {ex}")
                return Gst.FlowReturn.ERROR

            if spinnaker_image.IsIncomplete():
                Gst.warning(
                    f"Image incomplete with image status {image_result.GetImageStatus()}"
                )
                spinnaker_image.Release()

        image = gst_buffer_with_pad_to_ndarray(buffer, self.srcpad)
        image[:] = spinnaker_image.GetNDArray()

        image_timestamp = spinnaker_image.GetTimeStamp()

        if self.timestamp_offset == 0:
            self.timestamp_offset = image_timestamp
            self.previous_timestamp = image_timestamp

        buffer.pts = image_timestamp - self.timestamp_offset
        buffer.duration = image_timestamp - self.previous_timestamp

        self.previous_timestamp = image_timestamp

        Gst.log(
            f"Sending buffer of size: {image.shape} "
            f"frame id: {spinnaker_image.GetFrameID()} "
            f"timestamp offset: {buffer.pts // MILLIESCONDS_PER_NANOSECOND}ms"
        )
        spinnaker_image.Release()

        return Gst.FlowReturn.OK


# Register plugin to use it from command line
GObject.type_register(PySpinSrc)
__gstelementfactory__ = (PySpinSrc.GST_PLUGIN_NAME, Gst.Rank.NONE, PySpinSrc)
