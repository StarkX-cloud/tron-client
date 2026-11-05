"""Pure unit tests for hardware detection — no model loading, always fast."""

from tron_efficiency.hardware.detect import HardwareInfo, detect_hardware


def test_detect_hardware_returns_populated_info():
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.platform
    assert hw.python_version
    assert isinstance(hw.cuda_available, bool)
    assert isinstance(hw.gpus, list)


def test_detect_hardware_without_cuda_warns():
    hw = detect_hardware()
    if not hw.cuda_available:
        assert any("no CUDA GPU" in w for w in hw.warnings)
        assert hw.gpus == []


def test_hardware_info_to_dict_round_trips_shape():
    hw = detect_hardware()
    d = hw.to_dict()
    assert d["platform"] == hw.platform
    assert d["cuda_available"] == hw.cuda_available
    assert "gpus" in d and "warnings" in d
