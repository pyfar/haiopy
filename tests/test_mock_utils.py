import haiopy.mock_utils as mock_utils
import haiopy.devices as devices


def test__Device_mock_properties():
    """ Test to check _Device mock initialization.
    """
    mock_dir = dir(mock_utils._Device)
    device_dir = dir(devices._Device)

    assert mock_dir.sort() == device_dir.sort()


def test_AudioDevice_mock_properties():
    """ Test to check AudioDevice mock initialization.
    """
    mock_dir = dir(mock_utils.AudioDevice)
    device_dir = dir(devices.AudioDevice)

    assert mock_dir.sort() == device_dir.sort()


