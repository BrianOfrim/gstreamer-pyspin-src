from dataclasses import dataclass
from typing import List

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

DEFAULT_EXPOSURE_TIME = -1
DEFAULT_GAIN = -1
DEFAULT_H_BINNING = 1
DEFAULT_V_BINNING = 1
DEFAULT_OFFSET_X = 0
DEFAULT_OFFSET_Y = 0
DEFAULT_NUM_BUFFERS = 10
DEFAULT_SERIAL_NUMBER = None
DEFAULT_LOAD_DEFAULT = True

MILLIESCONDS_PER_NANOSECOND = 1000000
TIMEOUT_MS = 2000


@dataclass
class PixelFormatType:
    cap_type: str
    gst: str
    genicam: str


RAW_CAP_TYPE = "video/x-raw"
BAYER_CAP_TYPE = "video/x-bayer"

SUPPORTED_PIXEL_FORMATS = [
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="GRAY8", genicam="Mono8"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="GRAY16_LE", genicam="Mono16"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="UYVY", genicam="YUV422Packed"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="YUY2", genicam="YCbCr422_8"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="RGB", genicam="RGB8"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="BGR", genicam="BGR8"),
    PixelFormatType(cap_type=BAYER_CAP_TYPE, gst="rggb", genicam="BayerRG8"),
    PixelFormatType(cap_type=BAYER_CAP_TYPE, gst="gbrg", genicam="BayerGB8"),
    PixelFormatType(cap_type=BAYER_CAP_TYPE, gst="bggr", genicam="BayerBG8"),
    PixelFormatType(cap_type=BAYER_CAP_TYPE, gst="grbg", genicam="BayerGR8"),
]


class PySpinSrc(GstBase.PushSrc):

    GST_PLUGIN_NAME = "pyspinsrc"

    __gstmetadata__ = ("pyspinsrc", "Src", "PySpin src element", "Brian Ofrim")

    __gsttemplates__ = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any(),
    )

    __gproperties__ = {
        "exposure-time": (
            float,
            "exposure time",
            "Exposure time in microsecods (if not specified auto exposure is used)zz",
            -1,
            100000000.0,
            DEFAULT_EXPOSURE_TIME,
            GObject.ParamFlags.READWRITE,
        ),
        "gain": (
            float,
            "gain",
            "Gain in decibels (if not specified auto gain is used)",
            -1,
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
        "offset-x": (
            int,
            "offset x",
            "Horizontal offset for the region of interest",
            0,
            GLib.MAXINT,
            DEFAULT_OFFSET_X,
            GObject.ParamFlags.READWRITE,
        ),
        "offset-y": (
            int,
            "offset y",
            "Vertical offset for the region of interest",
            0,
            GLib.MAXINT,
            DEFAULT_OFFSET_Y,
            GObject.ParamFlags.READWRITE,
        ),
        "num-image-buffers": (
            int,
            "number of image buffers",
            "Number of buffers for Spinnaker to allocate for buffer handling",
            1,
            GLib.MAXINT,
            DEFAULT_NUM_BUFFERS,
            GObject.ParamFlags.READWRITE,
        ),
        "serial": (
            str,
            "serial number",
            "The camera serial number",
            DEFAULT_SERIAL_NUMBER,
            GObject.ParamFlags.READWRITE,
        ),
        "load-defaults": (
            bool,
            "load default user set",
            "Apply properties on top of the default settings or on top of current settings",
            DEFAULT_LOAD_DEFAULT,
            GObject.ParamFlags.READWRITE,
        ),
    }

    # GST function
    def __init__(self):
        super(PySpinSrc, self).__init__()

        # Initialize properties before Base Class initialization
        self.info = GstVideo.VideoInfo()

        # Properties
        self.exposure_time: float = DEFAULT_EXPOSURE_TIME
        self.gain: float = DEFAULT_GAIN
        self.h_binning: int = DEFAULT_H_BINNING
        self.v_binning: int = DEFAULT_V_BINNING
        self.offset_x: int = DEFAULT_OFFSET_X
        self.offset_y: int = DEFAULT_OFFSET_Y
        self.num_cam_buffers: int = DEFAULT_NUM_BUFFERS
        self.serial: str = DEFAULT_SERIAL_NUMBER
        self.load_defaults: bool = DEFAULT_LOAD_DEFAULT

        self.camera_caps = None

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
            Gst.info("Default settings loaded")
        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False

        return True

    # Camera helper function
    # ret val = Height: int, Width: int, OffsetY: int, OffsetX int
    def get_roi(self) -> (int, int, int, int):
        return (
            self.cam.Height.GetValue(),
            self.cam.Width.GetValue(),
            self.cam.OffsetY.GetValue(),
            self.cam.OffsetX.GetValue(),
        )

    # Camera helper function
    def set_roi(self, height: int, width: int, offset_y: int = 0, offset_x: int = 0):

        self.cam.Height.SetValue(height)
        Gst.info(f"Height: {self.cam.Height.GetValue()}")

        self.cam.Width.SetValue(width)
        Gst.info(f"Width: {self.cam.Width.GetValue()}")

        self.cam.OffsetY.SetValue(offset_y)
        Gst.info(f"OffsetY: {self.cam.OffsetY.GetValue()}")

        self.cam.OffsetX.SetValue(offset_x)
        Gst.info(f"OffsetX: {self.cam.OffsetX.GetValue()}")

    def get_format_type_from_genicam(self, genicam_format: str) -> PixelFormatType:
        return next(
            (
                f
                for f in SUPPORTED_PIXEL_FORMATS
                if f.genicam.lower() == genicam_format.lower()
            ),
            None,
        )

    def get_format_type_from_gst(self, gst_format: str) -> PixelFormatType:
        return next(
            (f for f in SUPPORTED_PIXEL_FORMATS if f.gst.lower() == gst_format.lower()),
            None,
        )

    # Camera helper function
    def apply_caps_to_cam(self) -> bool:
        Gst.info("Applying caps.")
        try:

            self.cam.PixelFormat.SetIntValue(
                self.cam.PixelFormat.GetEntryByName(
                    self.get_format_type_from_gst(self.info.finfo.name).genicam
                ).GetValue()
            )

            Gst.info(
                f"Pixel format: {self.cam.PixelFormat.GetCurrentEntry().GetSymbolic()}"
            )

            self.set_roi(
                self.info.height, self.info.width, self.offset_y, self.offset_x
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

            if self.exposure_time < 0:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
                Gst.info(
                    f"Auto Exposure: {self.cam.ExposureAuto.GetCurrentEntry().GetSymbolic()}"
                )
            else:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                self.cam.ExposureTime.SetValue(self.exposure_time)
                Gst.info(f"Exposure Time: {self.cam.ExposureTime.GetValue()}us")

            if self.gain < 0:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
                Gst.info(
                    f"Auto Gain: {self.cam.GainAuto.GetCurrentEntry().GetSymbolic()}"
                )
            else:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                self.cam.Gain.SetValue(self.gain)
                Gst.info(f"Gain: {self.cam.Gain.GetValue()}db")

        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False

        return True

    def get_camera_caps(self) -> Gst.Caps:

        # Get current pixel format
        starting_pixel_format_name = (
            self.cam.PixelFormat.GetCurrentEntry().GetDisplayName()
        )

        # Get Pixel Formats
        supported_pixel_formats = [
            self.get_format_type_from_genicam(pf.GetDisplayName())
            for pf in self.cam.PixelFormat.GetEntries()
            if PySpin.IsAvailable(pf)
            and self.get_format_type_from_genicam(pf.GetDisplayName()) is not None
        ]

        camera_caps = Gst.Caps.new_empty()

        for pixel_format_type in supported_pixel_formats:

            # set the pixel format
            self.cam.PixelFormat.SetIntValue(
                self.cam.PixelFormat.GetEntryByName(
                    pixel_format_type.genicam
                ).GetValue()
            )

            camera_caps.append_structure(
                Gst.Structure(
                    pixel_format_type.cap_type,
                    format=pixel_format_type.gst,
                    width=Gst.IntRange(
                        range(self.cam.Width.GetMin(), self.cam.Width.GetMax())
                    ),
                    height=Gst.IntRange(
                        range(self.cam.Height.GetMin(), self.cam.Height.GetMax())
                    ),
                    framerate=Gst.FractionRange(
                        Gst.Fraction(
                            *Gst.util_double_to_fraction(
                                self.cam.AcquisitionFrameRate.GetMin()
                            )
                        ),
                        Gst.Fraction(
                            *Gst.util_double_to_fraction(
                                self.cam.AcquisitionFrameRate.GetMax()
                            )
                        ),
                    ),
                )
            )

        # Set the pixel format back to the starting format
        self.cam.PixelFormat.SetIntValue(
            self.cam.PixelFormat.GetEntryByName(starting_pixel_format_name).GetValue()
        )

        return camera_caps

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

            if self.load_defaults and not self.apply_default_settings():
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
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()
                Gst.info("Acquisition Ended")
            if self.cam.IsInitialized():
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
        Gst.info("Setting caps")
        self.info.from_caps(caps)
        self.set_blocksize(self.info.size)
        return self.start_streaming()

    def do_get_caps(self, filter: Gst.Caps) -> Gst.Caps:
        Gst.info("Get Caps")
        caps = None

        if self.camera_caps is not None:
            caps = Gst.Caps.copy(self.camera_caps)
        else:
            caps = Gst.Caps.new_any()

        Gst.info(f"Avaliable caps: {caps.to_string()}")
        return caps

    # GST function
    def do_fixate(self, caps: Gst.Caps) -> Gst.Caps:
        Gst.info("Fixating caps")

        current_cam_height, current_cam_width, _, _ = self.get_roi()
        frame_rate = self.cam.AcquisitionFrameRate.GetValue()

        structure = caps.get_structure(0).copy()
        structure.fixate_field_nearest_int("width", current_cam_width)
        structure.fixate_field_nearest_int("height", current_cam_height)
        structure.fixate_field_nearest_fraction("framerate", frame_rate, 1)

        new_caps = Gst.Caps.new_empty()
        new_caps.append_structure(structure)
        return new_caps.fixate()

    # GST function
    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "exposure-time":
            return self.exposure_time
        elif prop.name == "gain":
            return self.gain
        elif prop.name == "h-binning":
            return self.h_binning
        elif prop.name == "v-binning":
            return self.v_binning
        elif prop.name == "offset-x":
            return self.offset_x
        elif prop.name == "offset-y":
            return self.offset_y
        elif prop.name == "num-image-buffers":
            return self.num_cam_buffers
        elif prop.name == "serial":
            return self.serial
        elif prop.name == "load-defaults":
            return self.load_defaults
        else:
            raise AttributeError("unknown property %s" % prop.name)

    # GST function
    def do_set_property(self, prop: GObject.GParamSpec, value):
        Gst.info(f"Setting {prop.name} = {value}")
        if prop.name == "exposure-time":
            self.exposure_time = value
        elif prop.name == "gain":
            self.gain = value
        elif prop.name == "h-binning":
            self.h_binning = value
        elif prop.name == "v-binning":
            self.v_binning = value
        elif prop.name == "offset-x":
            self.offset_x = value
        elif prop.name == "offset-y":
            self.offset_y = value
        elif prop.name == "num-image-buffers":
            self.num_cam_buffers = value
        elif prop.name == "serial":
            self.serial = value
        elif prop.name == "load-defaults":
            self.load_defaults = value
        else:
            raise AttributeError("unknown property %s" % prop.name)

    # GST function
    def do_start(self) -> bool:
        Gst.info("Starting")
        if not self.init_cam():
            return False

        self.camera_caps = self.get_camera_caps()

        return True

    # GST function
    def do_stop(self) -> bool:
        Gst.info("Stopping")
        return self.deinit_cam()

    # GST function
    def do_get_times(self, buffer: Gst.Buffer) -> (int, int):
        end = 0
        start = 0
        if self.is_live:
            ts = buffer.pts
            if ts != Gst.CLOCK_TIME_NONE:
                duration = buffer.duration
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
                    f"Image incomplete with image status {spinnaker_image.GetImageStatus()}"
                )
                spinnaker_image.Release()

        image_buffer = gst_buffer_with_pad_to_ndarray(buffer, self.srcpad)

        image_array = spinnaker_image.GetNDArray()

        image_buffer[:] = (
            image_array if image_array.ndim > 2 else np.expand_dims(image_array, axis=2)
        )

        image_timestamp = spinnaker_image.GetTimeStamp()

        if self.timestamp_offset == 0:
            self.timestamp_offset = image_timestamp
            self.previous_timestamp = image_timestamp

        buffer.pts = image_timestamp - self.timestamp_offset
        buffer.duration = image_timestamp - self.previous_timestamp

        self.previous_timestamp = image_timestamp

        Gst.log(
            f"Sending buffer of size: {image_buffer.shape} "
            f"frame id: {spinnaker_image.GetFrameID()} "
            f"timestamp offset: {buffer.pts // MILLIESCONDS_PER_NANOSECOND}ms"
        )
        spinnaker_image.Release()

        return Gst.FlowReturn.OK


# Register plugin to use it from command line
GObject.type_register(PySpinSrc)
__gstelementfactory__ = (PySpinSrc.GST_PLUGIN_NAME, Gst.Rank.NONE, PySpinSrc)
