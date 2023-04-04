from haiopy import devices
from . import utils
from . import sounddevice_mocks as sdm
from unittest.mock import patch
import sounddevice as sd
import pytest
import numpy as np
from numpy import testing as npt


@patch('sounddevice.query_devices', new=utils.query_devices)
def test_audio_device():
    devices.AudioDevice(0)


@patch('sounddevice.query_devices', new=utils.query_devices)
@patch('sounddevice.check_output_settings', new=utils.check_output_settings)
@patch('sounddevice.OutputStream', new=sdm.output_stream_mock())
def test_check_output_settings(empty_buffer_stub):
    out_device = devices.OutputAudioDevice(
        output_buffer=empty_buffer_stub[0])
    out_device.check_settings()


@patch('sounddevice.query_devices', new=utils.query_devices)
@patch('sounddevice.check_output_settings', new=utils.check_output_settings)
@patch('sounddevice.OutputStream', new=sdm.output_stream_mock())
def test_check_init(empty_buffer_stub):
    buffer = empty_buffer_stub[0]
    out_device = devices.OutputAudioDevice(
        output_buffer=empty_buffer_stub[0])
    out_device.check_settings()

    out_device.output_buffer = buffer
    out_device._output_buffer == buffer
    out_device.output_buffer == buffer

    # set a buffer with non matching block size
    buffer.block_size = 256
    with pytest.raises(ValueError, match='block size does not match'):
        out_device.output_buffer = buffer

    # change the block size of the buffer and check if buffers block size is
    # set accordingly
    new_block_size = 256
    out_device.block_size = new_block_size
    out_device._block_size == new_block_size
    out_device.output_buffer.block_size == new_block_size

    # set and get sampling rate
    out_device.sampling_rate = 44100
    out_device._sampling_rate == 44100


@patch('sounddevice.query_devices', new=utils.query_devices)
@patch('sounddevice.check_output_settings', new=utils.check_output_settings)
@patch('sounddevice.outputstream', new=sdm.output_stream_mock())
def test_sine_playback(sine_buffer_stub):
    buffer = sine_buffer_stub[0]

    config = {'default_samplerate': 44100}
    sampling_rate = config['default_samplerate']

    channels = [1]

    out_device = devices.OutputAudioDevice(
        identifier=0,
        output_buffer=buffer,
        channels=channels,
        sampling_rate=sampling_rate)
    out_device.check_settings()

    out_device.start()
    assert out_device.output_buffer.is_active is True

    # manually call the callback function with an arbitrary outdata array
    outdata = np.zeros((512, 2), dtype=np.float32)
    # unset all callback flags
    status = sd.CallbackFlags()

    # call callback, this would happen in a separate thread controlled by
    # portaudio. Once the buffer is empty, the callback should raise an
    # sd.CallbackStop exception.
    with pytest.raises(sd.CallbackStop, match='Buffer empty'):
        bdx = 0
        while True:
            out_device.output_callback(outdata, 512, None, status)
            npt.assert_allclose(
                outdata[:, 1],
                buffer._strided_data[0, bdx, :])
            bdx += 1

    out_device._finished_callback()
    assert out_device.output_buffer.is_active is False


@patch('sounddevice.query_devices', new=utils.query_devices)
@patch('sounddevice.check_output_settings', new=utils.check_output_settings)
@patch('sounddevice.OutputStream', new=sdm.output_stream_mock())
def test_callback_errors(sine_buffer_stub):
    buffer = sine_buffer_stub[0]

    config = {'default_samplerate': 44100}
    sampling_rate = config['default_samplerate']

    channels = [1]

    out_device = devices.OutputAudioDevice(
        identifier=0,
        output_buffer=buffer,
        channels=channels,
        sampling_rate=sampling_rate)
    out_device.check_settings()

    out_device.start()
    assert out_device.output_buffer.is_active is True

    # manually call the callback function with an arbitrary outdata array
    outdata = np.zeros((512, 2), dtype=np.float32)
    # unset all callback flags
    status = sd.CallbackFlags()

    # No error
    out_device.output_callback(outdata, 512, None, status)

    # Buffer underflow
    with pytest.raises(sd.CallbackAbort, match='Buffer underflow'):
        status = sd.CallbackFlags()
        status.output_underflow = True
        out_device.output_callback(outdata, 512, None, status)
