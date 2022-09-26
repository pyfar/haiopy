from haiopy import devices
from . import utils
from unittest.mock import patch


@patch('sounddevice.query_devices', new=utils.query_devices)
def test_audio_device():
    devices.AudioDevice(0)


@patch('sounddevice.query_devices', new=utils.query_devices)
@patch('sounddevice.check_input_settings', new=utils.check_input_settings)
def test_check_input_settings():
    in_device = devices.AudioInputDevice()
    # in_device.check_settings()


@patch('sounddevice.query_devices', new=utils.query_devices)
@patch('sounddevice.check_output_settings', new=utils.check_output_settings)
def test_check_output_settings():
    in_device = devices.AudioOutputDevice()
    # in_device.check_settings()
