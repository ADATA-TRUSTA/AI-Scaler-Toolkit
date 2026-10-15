"""
DRAM modules in the system information, on every platform and without root.

Linux used to read them only through dmidecode, which needs root: without a
passwordless sudo rule for it, the modules were simply missing. udev has already
decoded the same SMBIOS table at boot and keeps it readable by anyone. Windows read
only the first module and reported the total as if it were one.
"""

import subprocess

import pytest

from service.utils import system_monitor as sm

pytestmark = pytest.mark.inference

# udevadm info --query=property --path=/sys/devices/virtual/dmi/id, from a laptop with
# soldered LPDDR5 (four channels shown as four modules) and one empty slot added.
UDEV_LAPTOP = """\
MEMORY_ARRAY_LOCATION=System Board Or Motherboard
MEMORY_ARRAY_MAX_CAPACITY=34359738368
MEMORY_DEVICE_0_SIZE=4294967296
MEMORY_DEVICE_0_LOCATOR=Controller0-ChannelA
MEMORY_DEVICE_0_TYPE=LPDDR5
MEMORY_DEVICE_0_SPEED_MTS=8533
MEMORY_DEVICE_0_MANUFACTURER=SK Hynix
MEMORY_DEVICE_0_PART_NUMBER=H58G66CK8BX147
MEMORY_DEVICE_0_CONFIGURED_SPEED_MTS=7467
MEMORY_DEVICE_1_SIZE=4294967296
MEMORY_DEVICE_1_TYPE=LPDDR5
MEMORY_DEVICE_1_SPEED_MTS=8533
MEMORY_DEVICE_1_MANUFACTURER=SK Hynix
MEMORY_DEVICE_2_PRESENT=0
MEMORY_DEVICE_2_LOCATOR=DIMM 2
MEMORY_DEVICE_10_SIZE=17179869184
MEMORY_DEVICE_10_TYPE=DDR5
MEMORY_DEVICE_10_CONFIGURED_SPEED_MTS=5600
MEMORY_DEVICE_10_MANUFACTURER=Not Specified
MEMORY_ARRAY_NUM_DEVICES=4
"""

# A Hyper-V guest: no type or speed, the hypervisor as manufacturer.
UDEV_VM = """\
E: MEMORY_DEVICE_0_SIZE=3221225472
E: MEMORY_DEVICE_0_TYPE=Unknown
E: MEMORY_DEVICE_0_MANUFACTURER=Microsoft Corporation
E: MEMORY_DEVICE_1_SIZE=31138512896
E: MEMORY_DEVICE_1_TYPE=Unknown
"""


def test_udev_modules_read_per_device_in_slot_order():
    modules = sm._parse_udev_memory_modules(UDEV_LAPTOP)
    assert modules == [
        {"size": "4 GB", "type": "LPDDR5", "speed_mhz": 8533, "manufacturer": "SK Hynix"},
        {"size": "4 GB", "type": "LPDDR5", "speed_mhz": 8533, "manufacturer": "SK Hynix"},
        # Slot 10 sorts after 1, not between 1 and 2; only a configured speed is
        # known; "Not Specified" is no manufacturer at all.
        {"size": "16 GB", "type": "DDR5", "speed_mhz": 5600},
    ]


def test_udev_modules_leave_out_what_firmware_does_not_state():
    modules = sm._parse_udev_memory_modules(UDEV_VM)
    assert modules == [
        {"size": "3 GB", "manufacturer": "Microsoft Corporation"},
        {"size": "29 GB"},
    ]


def test_udev_without_memory_devices_gives_nothing():
    assert sm._parse_udev_memory_modules("MEMORY_ARRAY_NUM_DEVICES=0\n") == []


def test_udev_reader_survives_a_missing_or_failing_udevadm(monkeypatch):
    def missing(*_a, **_k):
        raise FileNotFoundError("udevadm")

    monkeypatch.setattr(subprocess, "run", missing)
    assert sm._read_udev_memory_modules() == []

    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 1, "", "no device")
    )
    assert sm._read_udev_memory_modules() == []


def test_windows_lists_every_module():
    rows = [
        {"Capacity": 17179869184, "Speed": 5600, "SMBIOS": 34, "Type": 0, "Mfg": "Kingston"},
        {"Capacity": 17179869184, "Speed": 5600, "SMBIOS": 34, "Type": 0, "Mfg": "Kingston"},
    ]
    assert sm._parse_windows_memory_modules(rows) == [
        {"size": "16 GB", "type": "DDR5", "speed_mhz": 5600, "manufacturer": "Kingston"},
        {"size": "16 GB", "type": "DDR5", "speed_mhz": 5600, "manufacturer": "Kingston"},
    ]


def test_windows_single_module_and_unknown_fields():
    # ConvertTo-Json writes one module as an object, not a one-item list.
    row = {"Capacity": 8589934592, "Speed": 0, "SMBIOS": 0, "Type": 0, "Mfg": "Unknown"}
    assert sm._parse_windows_memory_modules(row) == [{"size": "8 GB"}]
    assert sm._parse_windows_memory_modules(None) == []
