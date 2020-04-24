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

DEFAULT_AUTO_EXPOSURE = True
DEFAULT_AUTO_GAIN = True
DEFAULT_EXPOSURE_TIME = -1
DEFAULT_GAIN = -1
DEFAULT_AUTO_WB = True
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
                f"No device with serial number '{device_serial}' or index '{device_index}' is available"
            )

        self._current_device.Init()

        # Ensure that acquisition is stopped
        try:
            self.end_acquisition()
        except ValueError:
            pass

        return True

    def start_acquisition(self):
        self.set_enum_node_val("AcquisitionMode", "Continuous")
        try:
            self._current_device.BeginAcquisition()
        except PySpin.SpinnakerException as ex:
            raise ValueError(f"Error: {ex}")

    def end_acquisition(self):
        try:
            self._current_device.EndAcquisition()
        except PySpin.SpinnakerException as ex:
            raise ValueError(f"Error: {ex}")

    # Convenience method
    def restore_default_settings(self):
        self.set_enum_node_val("UserSetSelector", "Default")
        self.execute_command_node("UserSetLoad")

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
            self.set_bool_node_val("AcquisitionFrameRateEnabled", True, logger)

        self.set_float_node_val("AcquisitionFrameRate", frame_rate, logger)

    def configure_buffer_handling(
        self, num_device_buffers: int = 10, logger: Callable[[str], None] = None
    ):
        # Configure Transport Layer Properties
        self._current_device.TLStream.StreamBufferHandlingMode.SetValue(
            PySpin.StreamBufferHandlingMode_OldestFirst
        )
        self._current_device.TLStream.StreamBufferCountMode.SetValue(
            PySpin.StreamBufferCountMode_Manual
        )
        self._current_device.TLStream.StreamBufferCountManual.SetValue(
            num_device_buffers
        )
        if logger:
            logger(
                f"Buffer Handling Mode: {self._current_device.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetSymbolic()}"
            )
            logger(
                f"Buffer Count Mode: {self._current_device.TLStream.StreamBufferCountMode.GetCurrentEntry().GetSymbolic()}"
            )
            logger(
                f"Buffer Count: {self._current_device.TLStream.StreamBufferCountManual.GetValue()}"
            )
            logger(
                f"Max Buffer Count: {self._current_device.TLStream.StreamBufferCountManual.GetMax()}"
            )

    def get_int_node_val(self, node_name: str) -> int:

        int_node = PySpin.CIntegerPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(int_node) or not PySpin.IsReadable(int_node):
            raise ValueError(f"Integer node '{node_name}' is not readable")
        return int_node.GetValue()

    def get_int_node_range(self, node_name: str) -> (int, int):
        int_node = PySpin.CIntegerPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(int_node) or not PySpin.IsReadable(int_node):
            raise ValueError(f"Integer node '{node_name}' is not writable")

        return (int_node.GetMin(), int_node.GetMax())

    def set_int_node_val(
        self, node_name: str, value: int, logger: Callable[[str], None] = None
    ):

        int_node = PySpin.CIntegerPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(int_node) or not PySpin.IsWritable(int_node):
            raise ValueError(f"Integer node '{node_name}' is not writable")

        value = max(value, int_node.GetMin())
        value = min(value, int_node.GetMax())

        int_node.SetValue(int(value))

        if logger:
            logger(f"{node_name} = {self.get_int_node_val(node_name)}")

    def get_float_node_val(self, node_name: str) -> (float, float):
        float_node = PySpin.CFloatPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(float_node) or not PySpin.IsReadable(float_node):
            raise ValueError(f"Float node '{node_name}' is not readable")
        return float_node.GetValue()

    def get_float_node_range(self, node_name: str) -> (int, int):
        float_node = PySpin.CFloatPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(float_node) or not PySpin.IsReadable(float_node):
            raise ValueError(f"Float node '{node_name}' is not readable")
        return (float_node.GetMin(), float_node.GetMax())

    def set_float_node_val(
        self, node_name: str, value: float, logger: Callable[[str], None] = None
    ):
        float_node = PySpin.CFloatPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(float_node) or not PySpin.IsWritable(float_node):
            raise ValueError(f"Float node '{node_name}' is not writable")

        value = max(value, float_node.GetMin())
        value = min(value, float_node.GetMax())
        float_node.SetValue(float(value))

        if logger:
            logger(f"{node_name} = {self.get_float_node_val(node_name)}")

    def get_bool_node_val(self, node_name: str) -> bool:
        bool_node = PySpin.CBooleanPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(bool_node) or not PySpin.IsReadable(bool_node):
            raise ValueError(f"Boolean node '{node_name}' is not readable")
        return bool_node.GetValue()

    def set_bool_node_val(
        self, node_name: str, value: bool, logger: Callable[[str], None] = None
    ):
        bool_node = PySpin.CBooleanPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(bool_node) or not PySpin.IsWritable(bool_node):
            raise ValueError(f"Boolean node '{node_name}' is not writable")

        bool_node.SetValue(bool(value))

        if logger:
            logger(f"{node_name} = {self.get_bool_node_val(node_name)}")

    def enum_node_available(self, node_name: str) -> bool:
        return PySpin.IsAvailable(
            PySpin.CEnumerationPtr(self._get_node_map().GetNode(node_name))
        )

    def get_available_enum_entries(self, node_name: str) -> List[str]:
        enum_node = PySpin.CEnumerationPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(enum_node) or not PySpin.IsReadable(enum_node):
            raise ValueError(f"Enumeration node '{node_name}' is not readable")

        available_entries = [
            pf.GetName().split("_")[-1]
            for pf in enum_node.GetEntries()
            if PySpin.IsAvailable(pf)
        ]

        return available_entries

    def get_enum_node_val(self, node_name: str) -> str:

        enum_node = PySpin.CEnumerationPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(enum_node) or not PySpin.IsReadable(enum_node):
            raise ValueError(f"Enumeration node '{node_name}' is not readable")
        return enum_node.GetCurrentEntry().GetSymbolic()

    def set_enum_node_val(
        self, node_name: str, value: str, logger: Callable[[str], None] = None
    ):

        enum_node = PySpin.CEnumerationPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(enum_node) or not PySpin.IsWritable(enum_node):
            raise ValueError(f"Enumeration node '{node_name}' is not writable")

        enum_entry = enum_node.GetEntryByName(value)
        if not PySpin.IsAvailable(enum_entry) or not PySpin.IsReadable(enum_entry):
            raise ValueError(
                f"Entry '{value}' for enumeration node '{node_name}' is not available"
            )

        enum_node.SetIntValue(enum_entry.GetValue())

        if logger:
            logger(f"{node_name} = {self.get_enum_node_val(node_name)}")

    def execute_command_node(
        self, node_name: str, logger: Callable[[str], None] = None
    ):
        command_node = PySpin.CCommandPtr(self._get_node_map().GetNode(node_name))
        if not PySpin.IsAvailable(command_node) or not PySpin.IsWritable(command_node):
            raise ValueError(f"Error: Command node '{node_name}' is not writable")

        command_node.Execute()

        if logger:
            logger(f"{node_name} executed")

    def get_next_image(self, logger: Callable[[str], None] = None) -> (np.ndarray, int):
        spinnaker_image = None

        while spinnaker_image is None or spinnaker_image.IsIncomplete():

            # Grab a buffered image from the camera
            try:
                spinnaker_image = self._current_device.GetNextImage(TIMEOUT_MS)
            except PySpin.SpinnakerException as ex:
                if logger:
                    logger(f"Error: {ex}")
                return None, None

            if spinnaker_image.IsIncomplete():
                logger(
                    f"Image incomplete with image status {spinnaker_image.GetImageStatus()}"
                )
                spinnaker_image.Release()

        image_array = spinnaker_image.GetNDArray()
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=2)
        image_timestamp = spinnaker_image.GetTimeStamp()

        spinnaker_image.Release()

        return (image_array, image_timestamp)


class PySpinSrc(GstBase.PushSrc):

    GST_PLUGIN_NAME = "pyspinsrc"

    __gstmetadata__ = ("pyspinsrc", "Src", "PySpin src element", "Brian Ofrim")

    __gsttemplates__ = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any(),
    )

    __gproperties__ = {
        "auto-exposure": (
            bool,
            "automatic exposure timing",
            "Enable the automatic exposure time algorithm",
            DEFAULT_AUTO_EXPOSURE,
            GObject.ParamFlags.READWRITE,
        ),
        "auto-gain": (
            bool,
            "automatic gain",
            "Enable the automatic gain algorithm",
            DEFAULT_AUTO_GAIN,
            GObject.ParamFlags.READWRITE,
        ),
        "exposure": (
            float,
            "exposure time",
            "Exposure time in microsecods",
            -1,
            100000000.0,
            DEFAULT_EXPOSURE_TIME,
            GObject.ParamFlags.READWRITE,
        ),
        "gain": (
            float,
            "gain",
            "Gain in decibels",
            -1,
            100.0,
            DEFAULT_GAIN,
            GObject.ParamFlags.READWRITE,
        ),
        "auto-wb": (
            bool,
            "automatic white balance",
            "Enable the automatic white balance algorithm",
            DEFAULT_AUTO_WB,
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
        self.auto_exposure: bool = DEFAULT_AUTO_EXPOSURE
        self.auto_gain: bool = DEFAULT_AUTO_GAIN
        self.exposure_time: float = DEFAULT_EXPOSURE_TIME
        self.gain: float = DEFAULT_GAIN
        self.auto_wb: bool = DEFAULT_AUTO_WB
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

        self.image_acquirer: ImageAcquirer = ImageAcquirer()

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

    # Camera helper function
    def apply_buffer_handling_properties(self) -> bool:
        try:
            self.image_acquirer.configure_buffer_handling(self.num_cam_buffers)
        except ValueError as ex:
            Gst.error(f"Error: {ex}")
            return False
        return True

    # Camera helper function
    def apply_properties_to_cam(self) -> bool:
        Gst.info("Applying properties")
        try:
            # Configure Camera Properties
            if self.h_binning > 1:
                self.image_acquirer.set_int_node_val(
                    "BinningHorizontal", self.h_binning, Gst.info
                )

            if self.v_binning > 1:
                self.image_acquirer.set_int_node_val(
                    "BinningVertical", self.v_binning, Gst.info
                )

            if self.exposure_time >= 0:
                self.image_acquirer.set_enum_node_val("ExposureAuto", "Off", Gst.info)
                self.image_acquirer.set_float_node_val(
                    "ExposureTime", self.exposure_time, Gst.info
                )

            if self.auto_exposure:
                self.image_acquirer.set_enum_node_val(
                    "ExposureAuto", "Continuous", Gst.info
                )

            if self.gain >= 0:
                self.image_acquirer.set_enum_node_val("GainAuto", "Off", Gst.info)
                self.image_acquirer.set_float_node_val("Gain", self.gain, Gst.info)

            if self.auto_gain:
                self.image_acquirer.set_enum_node_val(
                    "GainAuto", "Continuous", Gst.info
                )

            if (
                self.image_acquirer.enum_node_available("BalanceWhiteAuto")
                and self.wb_blue >= 0
            ):
                self.image_acquirer.set_enum_node_val(
                    "BalanceWhiteAuto", "Off", Gst.info
                )
                self.image_acquirer.set_enum_node_val(
                    "BalanceRatioSelector", "Blue", Gst.info
                )
                self.image_acquirer.set_float_node_val(
                    "BalanceRatio", self.wb_blue, Gst.info
                )

            if (
                self.image_acquirer.enum_node_available("BalanceWhiteAuto")
                and self.wb_red >= 0
            ):
                self.image_acquirer.set_enum_node_val(
                    "BalanceWhiteAuto", "Off", Gst.info
                )
                self.image_acquirer.set_enum_node_val(
                    "BalanceRatioSelector", "Red", Gst.info
                )
                self.image_acquirer.set_float_node_val(
                    "BalanceRatio", self.wb_red, Gst.info
                )

            if (
                self.image_acquirer.enum_node_available("BalanceWhiteAuto")
                and self.auto_wb
            ):
                self.image_acquirer.set_enum_node_val(
                    "BalanceWhiteAuto", "Continuous", Gst.info
                )

        except ValueError as ex:
            Gst.error(f"Error: {ex}")
            return False

        return True

    # Camera helper function
    def get_camera_caps(self) -> Gst.Caps:

        # Get current pixel format
        starting_pixel_format = self.image_acquirer.get_enum_node_val("PixelFormat")

        genicam_formats = self.image_acquirer.get_available_enum_entries("PixelFormat")

        supported_pixel_formats = [
            self.get_format_from_genicam(pf) for pf in genicam_formats
        ]

        supported_pixel_formats = [
            pf for pf in supported_pixel_formats if pf is not None
        ]

        camera_caps = Gst.Caps.new_empty()

        for pixel_format in supported_pixel_formats:

            self.image_acquirer.set_enum_node_val("PixelFormat", pixel_format.genicam)

            width_min, width_max = self.image_acquirer.get_int_node_range("Width")
            height_min, height_max = self.image_acquirer.get_int_node_range("Height")
            fr_min, fr_max = self.image_acquirer.get_float_node_range(
                "AcquisitionFrameRate"
            )

            camera_caps.append_structure(
                Gst.Structure(
                    pixel_format.cap_type,
                    format=pixel_format.gst,
                    width=Gst.IntRange(range(width_min, width_max)),
                    height=Gst.IntRange(range(height_min, height_max)),
                    framerate=Gst.FractionRange(
                        Gst.Fraction(*Gst.util_double_to_fraction(fr_min)),
                        Gst.Fraction(*Gst.util_double_to_fraction(fr_max)),
                    ),
                )
            )

        # Set the pixel format back to the starting format
        self.image_acquirer.set_enum_node_val("PixelFormat", starting_pixel_format)

        return camera_caps

    # Camera helper function
    def start_streaming(self) -> bool:

        if not self.apply_caps_to_cam():
            return False

        try:
            self.image_acquirer.start_acquisition()
        except ValueError as ex:
            Gst.error(f"Error: {ex}")
            return False

        Gst.info("Acquisition Started")
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
        frame_rate = self.image_acquirer.get_float_node_val("AcquisitionFrameRate")

        structure = caps.get_structure(0).copy()
        structure.fixate_field_nearest_int("width", current_cam_width)
        structure.fixate_field_nearest_int("height", current_cam_height)
        structure.fixate_field_nearest_fraction("framerate", frame_rate, 1)

        new_caps = Gst.Caps.new_empty()
        new_caps.append_structure(structure)
        return new_caps.fixate()

    # GST function
    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "auto-exposure":
            return self.auto_exposure
        elif prop.name == "auto-gain":
            return self.auto_gain
        elif prop.name == "exposure":
            return self.exposure_time
        elif prop.name == "gain":
            return self.gain
        elif prop.name == "auto-wb":
            return self.auto_wb
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
        if prop.name == "auto-exposure":
            self.auto_exposure = value
        elif prop.name == "auto-gain":
            self.auto_gain = value
        elif prop.name == "exposure":
            self.exposure_time = value
        elif prop.name == "gain":
            self.gain = value
        elif prop.name == "auto-wb":
            self.auto_wb = value
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
        try:
            if not self.image_acquirer.init_device(
                device_serial=self.serial,
                device_index=(0 if self.serial is None else None),
            ):
                return False

            if self.load_defaults:
                self.image_acquirer.restore_default_settings()

            if not self.apply_properties_to_cam():
                return False

            if not self.apply_buffer_handling_properties():
                return False

            self.camera_caps = self.get_camera_caps()
        except ValueError as ex:
            Gst.error(f"Error: {ex}")
            return False
        return True

    # GST function
    def do_stop(self) -> bool:
        Gst.info("Stopping")
        try:
            del self.image_acquirer
        except Exception as ex:
            Gst.error(f"Error: {ex}")
        return True

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

        image_buffer = gst_buffer_with_pad_to_ndarray(buffer, self.srcpad)

        image_array, image_timestamp = self.image_acquirer.get_next_image(
            logger=Gst.warning
        )

        if image_array is None:
            return Gst.FlowReturn.ERROR

        image_buffer[:] = image_array

        if self.timestamp_offset == 0:
            self.timestamp_offset = image_timestamp
            self.previous_timestamp = image_timestamp

        buffer.pts = image_timestamp - self.timestamp_offset
        buffer.duration = image_timestamp - self.previous_timestamp

        self.previous_timestamp = image_timestamp

        Gst.log(
            f"Sending buffer of size: {image_buffer.shape} "
            f"timestamp offset: {buffer.pts // MILLIESCONDS_PER_NANOSECOND}ms"
        )

        return Gst.FlowReturn.OK


# Register plugin to use it from command line
GObject.type_register(PySpinSrc)
__gstelementfactory__ = (PySpinSrc.GST_PLUGIN_NAME, Gst.Rank.NONE, PySpinSrc)
