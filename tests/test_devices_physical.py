from haiopy import devices
import sounddevice as sd
import pytest
import os


def default_device_multiface_fireface(kind='both'):
    device_list = sd.query_devices()
    found = False

    valid_devices = [
        'Multiface',
        'Fireface',
        'Scarlett 2i4',
        'MADIface',
        'Focusrite USB ASIO']

    for valid_device in valid_devices:
        for identifier, device in enumerate(device_list):
            if valid_device in device['name']:
                found = True
                break
    if not found:
        raise ValueError(
            "Please connect Fireface or Multiface, or specify test device.")

    return identifier, device


@pytest.mark.skipif(os.environ.get('CI') == 'true',
                    reason="CI does not have a soundcard")
def test_default_device_helper():
    identifier, device = default_device_multiface_fireface()
    fireface = 'Fireface' in sd.query_devices(identifier)['name']
    multiface = 'Multiface' in sd.query_devices(identifier)['name']
    scarlett = 'Scarlett 2i4' in sd.query_devices(identifier)['name']
    madiface = 'MADIface' in sd.query_devices(identifier)['name']
    focusrite = 'Focusrite USB ASIO' in sd.query_devices(identifier)['name']

    assert fireface or multiface or scarlett or madiface or focusrite

    if fireface:
        assert device['max_input_channels'] == 18
        assert device['max_output_channels'] == 18

    if scarlett:
        assert device['max_input_channels'] == 2
        assert device['max_output_channels'] == 4

    if madiface:
        assert device['max_input_channels'] == 196
        assert device['max_output_channels'] == 198

    if focusrite:
        assert device['max_input_channels'] == 2
        assert device['max_output_channels'] == 2

# -----------------------------------------------------------------------------
# Output Device Tests
# -----------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get('CI') == 'true',
                    reason="CI does not have a soundcard")
def test_check_output_settings(empty_buffer_stub):
    identifier, config = default_device_multiface_fireface()
    channels = [0]
    block_size = 512

    buffer = empty_buffer_stub[0]
    sampling_rate = config['default_samplerate']

    out_device = devices.OutputAudioDevice(identifier=identifier,
                                           sampling_rate=sampling_rate,
                                           block_size=block_size,
                                           channels=channels,
                                           dtype='float32',
                                           output_buffer=buffer)

    # Check sampling rate
    out_device.check_settings(sampling_rate=sampling_rate)
    with pytest.raises(sd.PortAudioError, match="Invalid"):
        out_device.check_settings(sampling_rate=10)

    # Check the dtype, apparently this raises a ValueError if invalid
    out_device.check_settings(dtype='float32')
    with pytest.raises(ValueError, match="Invalid"):
        out_device.check_settings(dtype=float)

    # Check number of channels
    out_device.check_settings(n_channels=config['max_output_channels'])
    with pytest.raises(sd.PortAudioError, match="Invalid"):
        out_device.check_settings(config['max_output_channels']+10)

    # Close Output Stream for next Tests
    with pytest.raises(StopIteration, match="iteration stopped"):
        out_device.close()


@pytest.mark.skipif(os.environ.get('CI') == 'true',
                    reason="CI does not have a soundcard")
def test_sine_playback(sine_buffer_stub):

    buffer = sine_buffer_stub[0]
    identifier, config = default_device_multiface_fireface()

    sampling_rate = config['default_samplerate']

    out_device = devices.OutputAudioDevice(
        identifier=identifier,
        output_buffer=buffer,
        channels=[1],
        sampling_rate=sampling_rate)
    out_device.check_settings()

    out_device.start()
    assert out_device.output_buffer.is_active is True
    out_device.wait()
    assert out_device.output_buffer.is_active is False

    # Close Output Stream for next Tests
    with pytest.raises(StopIteration, match="iteration stopped"):
        out_device.close()


@pytest.mark.skipif(os.environ.get('CI') == 'true',
                    reason="CI does not have a soundcard")
def test_check_init(empty_buffer_stub, sine_buffer_stub):
    buffer = sine_buffer_stub[0]
    identifier, config = default_device_multiface_fireface()

    sampling_rate = config['default_samplerate']

    out_device = devices.OutputAudioDevice(
        identifier=identifier,
        output_buffer=empty_buffer_stub[0],
        channels=[1],
        sampling_rate=sampling_rate)
    out_device.check_settings()
    assert out_device.output_buffer == empty_buffer_stub[0]
    out_device.output_buffer = buffer
    assert out_device._output_buffer == buffer
    assert out_device.output_buffer == buffer
    # test playback with new buffer
    out_device.start()
    out_device.wait()

    # set new block_size
    new_block_size = 1024
    out_device.block_size = new_block_size
    assert out_device._block_size == new_block_size
    assert out_device.output_buffer.block_size == new_block_size
    # test playback with new block_size
    out_device.start()
    out_device.wait()

    # set new sampling_rate
    new_sampling_rate = 88200 if sampling_rate == 44100 else 44100
    out_device.sampling_rate = new_sampling_rate
    assert out_device._sampling_rate == new_sampling_rate
    assert out_device.sampling_rate == new_sampling_rate
    assert out_device.output_buffer.sampling_rate == new_sampling_rate
    # test playback with new sampling_rate
    out_device.start()
    out_device.wait()

    # set new channels
    new_channels = [0, 1]
    out_device.channels = new_channels
    out_device._output_buffer.reset_index()
    assert out_device.channels == new_channels
    # test playback with new channels
    out_device.start()
    out_device.wait()

    # Close Output Stream for next Tests
    with pytest.raises(StopIteration, match="iteration stopped"):
        out_device.close()
