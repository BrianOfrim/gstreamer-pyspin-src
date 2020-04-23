from dataclasses import dataclass
import math
from typing import List, Callable

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")
gi.require_version("GstVideo", "1.0")

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
DEFAULT_WB_BLUE = -1
DEFAULT_WB_RED = -1
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
    #    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="GRAY16_LE", genicam="Mono16"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="UYVY", genicam="YUV422Packed"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="YUY2", genicam="YCbCr422_8"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="RGB", genicam="RGB8"),
    PixelFormatType(cap_type=RAW_CAP_TYPE, gst="BGR", genicam="BGR8"),
    PixelFormatType(cap_type=BAYER_CAP_TYPE, gst="rggb", genicam="BayerRG8"),
    PixelFormatType(cap_type=BAYER_CAP_TYPE, gst="gbrg", genicam="BayerGB8"),
    PixelFormatType(cap_type=BAYER_CAP_TYPE, gst="bggr", genicam="BayerBG8"),
    PixelFormatType(cap_type=BAYER_CAP_TYPE, gst="grbg", genicam="BayerGR8"),
]


class ImageAcquirer:
    def __init__(self):
        self._system = PySpin.System.GetInstance()
        self._device_list = self._system.GetCameras()
        self._current_device = None
        self._node_map = None

    def __del__(self):

        self._reset_cam()
        self._device_list.Clear()
        self._system.ReleaseInstance()

    def _reset_cam(self):
        if self._current_device is not None and self._current_device.IsValid():
            if self._current_device.IsStreaming():
                self._current_device.EndAcquisition()
            if self._current_device.IsInitialized():
                self._current_device.DeInit()

        del self._current_device
        self._node_map = None
        self._current_device = None

    def update_device_list(self):
        self._device_list = self._system.GetCameras()

    def get_device_count(self, update_list: bool = True) -> int:
        if update_list:
            self.update_device_list()
        return self._device_list.GetSize()

    def _get_node_map(self) -> PySpin.NodeMap:
        if self._current_device is None or not self._current_device.IsValid():
            raise ValueError("No device has been selected an initialied")
        if self._node_map is None:
            self._node_map = self._current_device.GetNodeMap()
        return self._node_map

    def _get_device_id(self) -> str:
        if self._current_device is None or not self._current_device.IsValid():
            return None
        return self._current_device.TLDevice.DeviceSerialNumber.GetValue()

    # Camera helper function
    def init_device(self, device_serial: str = None, device_index: int = None,) -> bool:
        # reset cam
        self._reset_cam()

        self.update_device_list()

        candidate_devices: List[PySpin.Camera] = []

        if device_serial is not None:
            candidate_devices.append(self._device_list.GetBySerial(device_serial))

        if device_index is not None and device_index < self.get_device_count(
            update_list=False
        ):
            candidate_devices.append(self._device_list.GetByIndex(device_index))

        self._current_device = next(
            (dev for dev in candidate_devices if dev and dev.IsValid()), None
        )

        if self._current_device is None:
            raise ValueError(
                f"Error: No device with serial number '{device_serial}' or index '{device_index}' is available"
            )

        self._current_device.Init()

        # Ensure that acquisition is stopped
        try:
            self.end_acquisition()
        except PySpin.SpinnakerException:
            pass

        return True

    def start_acquisition(self):
        self.set_enum_node_val("AcquisitionMode", "Continuous")
        self._current_device.BeginAcquisition()
        return True

    def end_acquisition(self):
        self._current_device.EndAcquisition()
        return True

    # Convenience method
    def restore_default_settings(self):
        self.set_enum_node_val("UserSetSelector", "Default")
        self.execute_command_node("UserSetLoad")
        return True

    # Convenience method
    def set_frame_rate(self, frame_rate: float, logger: Callable[[str], None] = None):
        if PySpin.IsAvailable(
            PySpin.CBooleanPtr(
                self._get_node_map().GetNode("AcquisitionFrameRateEnable")
            )
        ):
            self.set_bool_node_val("AcquisitionFrameRateEnable", True, logger)

        else:
            # Camera is an older model
            self.set_enum_node_val("AcquisitionFrameRateAuto", "Off", logger)
            self.set_bool_node_val("AcquisitionFrameRateEnabled", False, logger)

        self.set_float_node_val("AcquisitionFrameRate", frame_rate, logger)

    def configure_buffer_handling(self, num_device_buffers: int = 10):
        # Configure Transport Layer Properties
        self._current_device.TLStream.StreamBufferHandlingMode.SetValue(
            PySpin.StreamBufferHandlingMode_OldestFirst
        )
        self._current_device.StreamBufferCountMode.SetValue(
            PySpin.StreamBufferCountMode_Manual
        )
        self._current_device.StreamBufferCountManual.SetValue(num_device_buffers)

    def get_int_node_val(self, node_name: str) -> int:

        int_node = PySpin.CIntegerPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(int_node):
            raise ValueError(f"Error: Integer node '{node_name}' is not available")
        if not PySpin.IsReadable(int_node):
            raise ValueError(f"Error: Integer node '{node_name}' is not readable")
        return int_node.GetValue()

    def set_int_node_val(
        self, node_name: str, value: int, logger: Callable[[str], None] = None
    ):

        int_node = PySpin.CIntegerPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(int_node):
            raise ValueError(f"Error: Integer node '{node_name}' is not available")
        if not PySpin.IsWritable(int_node):
            raise ValueError(f"Error: Integer node '{node_name}' is not writable")

        value = max(value, int_node.GetMin())
        value = min(value, int_node.GetMax())

        int_node.SetValue(int(value))

        if logger:
            logger(f"{node_name} = {self.get_int_node_val(node_name)}")

    def get_float_node_val(self, node_name: str) -> float:

        float_node = PySpin.CFloatPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(float_node):
            raise ValueError(f"Error: Float node '{node_name}' is not available")
        if not PySpin.IsReadable(float_node):
            raise ValueError(f"Error: Float node '{node_name}' is not readable")
        return float_node.GetValue()

    def set_float_node_val(
        self, node_name: str, value: float, logger: Callable[[str], None] = None
    ):

        float_node = PySpin.CFloatPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(float_node):
            raise ValueError(f"Error: Float node '{node_name}' is not available")
        if not PySpin.IsWritable(float_node):
            raise ValueError(f"Error: Float node '{node_name}' is not writable")

        value = max(value, float_node.GetMin())
        value = min(value, float_node.GetMax())

        float_node.SetValue(float(value))

        if logger:
            logger(f"{node_name} = {self.get_float_node_val(node_name)}")

    def get_bool_node_val(self, node_name: str) -> bool:

        bool_node = PySpin.CBooleanPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(bool_node):
            raise ValueError(f"Error: Boolean node '{node_name}' is not available")
        if not PySpin.IsReadable(bool_node):
            raise ValueError(f"Error: Boolean node '{node_name}' is not readable")
        return bool_node.GetValue()

    def set_bool_node_val(
        self, node_name: str, value: bool, logger: Callable[[str], None] = None
    ):

        bool_node = PySpin.CBooleanPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(bool_node):
            raise ValueError(f"Error: Boolean node '{node_name}' is not available")
        if not PySpin.IsWritable(bool_node):
            raise ValueError(f"Error: Boolean node '{node_name}' is not writable")

        bool_node.SetValue(bool(value))

        if logger:
            logger(f"{node_name} = {self.get_bool_node_val(node_name)}")

    def get_enum_node_val(self, node_name: str) -> str:

        enum_node = PySpin.CEnumerationPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(enum_node):
            raise ValueError(f"Error: Enumeration node '{node_name}' is not available")
        if not PySpin.IsReadable(enum_node):
            raise ValueError(f"Error: Enumeration node '{node_name}' is not readable")
        return enum_node.GetCurrentEntry().GetSymbolic()

    def set_enum_node_val(
        self, node_name: str, value: str, logger: Callable[[str], None] = None
    ) -> bool:

        enum_node = PySpin.CEnumerationPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(enum_node):
            raise ValueError(f"Error: Enumeration node '{node_name}' is not available")
        if not PySpin.IsWritable(enum_node):
            raise ValueError(f"Error: Enumeration node '{node_name}' is not writable")

        enum_entry_node = enum_node.GetEntryByName(value)
        if not PySpin.IsAvailable(enum_entry_node):
            raise ValueError(
                f"Error: Entry '{value}' for enumeration node '{node_name}' is not available"
            )
        if not PySpin.IsReadable(enum_entry_node):
            raise ValueError(
                f"Error: Entry '{value}' for enumeration node '{node_name}' is not readable"
            )

        enum_node.SetIntValue(enum_entry_node)

        if logger:
            logger(f"{node_name} = {self.get_enum_node_val(node_name)}")

        return True

        # return True

    def execute_command_node(self, node_name: str) -> bool:
        command_node = PySpin.CCommandPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(command_node):
            raise ValueError(f"Error: Command node '{node_name}' is not available")
        if not PySpin.IsWritable(command_node):
            raise ValueError(f"Error: Command node '{node_name}' is not writable")
        command_node.Execute()
        return True


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
        "wb-blue-ratio": (
            float,
            "White balance blue ratio",
            "White balance blue/green ratio (If neither wb-blue or wb-red are specified, auto wb is used)",
            -1,
            10.0,
            DEFAULT_WB_BLUE,
            GObject.ParamFlags.READWRITE,
        ),
        "wb-red-ratio": (
            float,
            "White balance red ratio",
            "White balance red/green ratio (If neither wb-blue or wb-red are specified, auto wb is used)",
            -1,
            10.0,
            DEFAULT_WB_BLUE,
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
        self.wb_blue: float = DEFAULT_WB_BLUE
        self.wb_red: float = DEFAULT_WB_RED
        self.h_binning: int = DEFAULT_H_BINNING
        self.v_binning: int = DEFAULT_V_BINNING
        self.offset_x: int = DEFAULT_OFFSET_X
        self.offset_y: int = DEFAULT_OFFSET_Y
        self.num_cam_buffers: int = DEFAULT_NUM_BUFFERS
        self.serial: str = DEFAULT_SERIAL_NUMBER
        self.load_defaults: bool = DEFAULT_LOAD_DEFAULT

        self.camera_caps = None

        self.image_acquirer: ImageAcquirer = None

        # Buffer timing
        self.timestamp_offset: int = 0
        self.previous_timestamp: int = 0

        # Base class proprties
        self.set_live(True)
        self.set_format(Gst.Format.TIME)

    def get_format_from_genicam(self, genicam_format: str) -> PixelFormatType:
        return next(
            (
                f
                for f in SUPPORTED_PIXEL_FORMATS
                if f.genicam.lower() == genicam_format.lower()
            ),
            None,
        )

    def get_format_from_gst(self, gst_format: str) -> PixelFormatType:
        return next(
            (f for f in SUPPORTED_PIXEL_FORMATS if f.gst.lower() == gst_format.lower()),
            None,
        )

    # ret val = Height: int, Width: int, OffsetY: int, OffsetX int
    def get_roi(self) -> (int, int, int, int):
        return (
            self.image_acquirer.get_int_node_val("Height"),
            self.image_acquirer.get_int_node_val("Width"),
            self.image_acquirer.get_int_node_val("OffsetY"),
            self.image_acquirer.get_int_node_val("OffsetX"),
        )

    def set_roi(self, height: int, width: int, offset_y: int = 0, offset_x: int = 0):

        self.image_acquirer.set_int_node_val("Height", height, Gst.info)
        self.image_acquirer.set_int_node_val("Width", width, Gst.info)
        self.image_acquirer.set_int_node_val("OffsetY", offset_y, Gst.info)
        self.image_acquirer.set_int_node_val("OffsetX", offset_x, Gst.info)

    def set_pixel_format(self, gst_pixel_format: str):
        genicam_format = self.get_format_from_gst(gst_pixel_format).genicam
        self.image_acquirer.set_enum_node_val("PixelFormat", genicam_format, Gst.info)

    def apply_caps_to_cam(self) -> bool:
        Gst.info("Applying caps.")
        try:

            self.set_pixel_format(self.info.finfo.name)

            self.set_roi(
                self.info.height, self.info.width, self.offset_y, self.offset_x
            )

            self.image_acquirer.set_frame_rate(
                self.info.fps_n / self.info.fps_d, Gst.info
            )

        except ValueError as ex:
            Gst.error(f"Error: {ex}")
            return False
        return True

    # # Camera helper function
    # def apply_properties_to_transport_layer(self) -> bool:
    #     try:
    #         # Configure Transport Layer Properties
    #         self.cam.TLStream.StreamBufferHandlingMode.SetValue(
    #             PySpin.StreamBufferHandlingMode_OldestFirst
    #         )
    #         self.cam.TLStream.StreamBufferCountMode.SetValue(
    #             PySpin.StreamBufferCountMode_Manual
    #         )
    #         self.cam.TLStream.StreamBufferCountManual.SetValue(self.num_cam_buffers)

    #         Gst.info(
    #             f"Buffer Handling Mode: {self.cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetSymbolic()}"
    #         )
    #         Gst.info(
    #             f"Buffer Count Mode: {self.cam.TLStream.StreamBufferCountMode.GetCurrentEntry().GetSymbolic()}"
    #         )
    #         Gst.info(
    #             f"Buffer Count: {self.cam.TLStream.StreamBufferCountManual.GetValue()}"
    #         )
    #         Gst.info(
    #             f"Max Buffer Count: {self.cam.TLStream.StreamBufferCountManual.GetMax()}"
    #         )

    #     except PySpin.SpinnakerException as ex:

    #         Gst.error(f"Error: {ex}")
    #         return False
    #     return True

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
                self.exposure_time = max(
                    self.exposure_time, self.cam.ExposureTime.GetMin()
                )
                self.exposure_time = min(
                    self.exposure_time, self.cam.ExposureTime.GetMax()
                )
                self.cam.ExposureTime.SetValue(self.exposure_time)
                Gst.info(f"Exposure Time: {self.cam.ExposureTime.GetValue()}us")

            if self.gain < 0:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
                Gst.info(
                    f"Auto Gain: {self.cam.GainAuto.GetCurrentEntry().GetSymbolic()}"
                )
            else:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                self.gain = max(self.gain, self.cam.Gain.GetMin())
                self.gain = min(self.gain, self.cam.Gain.GetMax())
                self.cam.Gain.SetValue(self.gain)
                Gst.info(f"Gain: {self.cam.Gain.GetValue()}db")

            if self.cam.BalanceWhiteAuto.GetAccessMode() != PySpin.RW:
                Gst.info("White balance not applicable")
            elif self.wb_blue < 0 and self.wb_red < 0:
                self.cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Continuous)
                Gst.info(
                    f"Auto White Balance: {self.cam.BalanceWhiteAuto.GetCurrentEntry().GetSymbolic()}"
                )
            else:
                self.cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)

                self.cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Blue)
                if self.wb_blue >= 0:
                    self.wb_blue = max(self.wb_blue, self.cam.BalanceRatio.GetMin())
                    self.wb_blue = min(self.wb_blue, self.cam.BalanceRatio.GetMax())
                    self.cam.BalanceRatio.SetValue(self.wb_blue)
                Gst.info(
                    f"White balance blue/green ratio: {self.cam.BalanceRatio.GetValue()}"
                )

                self.cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Red)
                if self.wb_red >= 0:
                    self.wb_red = max(self.wb_red, self.cam.BalanceRatio.GetMin())
                    self.wb_red = min(self.wb_red, self.cam.BalanceRatio.GetMax())
                    self.cam.BalanceRatio.SetValue(self.wb_red)
                Gst.info(
                    f"White balance red/green ratio: {self.cam.BalanceRatio.GetValue()}"
                )

        except PySpin.SpinnakerException as ex:
            Gst.error(f"Error: {ex}")
            return False

        return True

    # Camera helper function
    def get_camera_caps(self) -> Gst.Caps:

        # Get current pixel format
        starting_pixel_format_name = (
            self.cam.PixelFormat.GetCurrentEntry().GetSymbolic()
        )

        # Get Pixel Formats
        supported_pixel_formats = [
            self.get_format_type_from_genicam(pf.GetName().split("_")[-1])
            for pf in self.cam.PixelFormat.GetEntries()
            if PySpin.IsAvailable(pf)
            and self.get_format_type_from_genicam(pf.GetName().split("_")[-1])
            is not None
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
        elif prop.name == "wb-blue-ratio":
            return self.wb_blue
        elif prop.name == "wb-red-ratio":
            return self.wb_red
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
        elif prop.name == "wb-blue-ratio":
            self.wb_blue = value
        elif prop.name == "wb-red-ratio":
            self.wb_red = value
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
