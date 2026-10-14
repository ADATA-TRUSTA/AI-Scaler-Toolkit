"""Integrated-vs-discrete GPU detection, against a synthetic DRM tree.

This is what decides whether an adapter counts toward the budget a model is sized
against. Getting it wrong in either direction is expensive: a missed iGPU inflates
the budget by ~100 GB of borrowed system RAM, and a misflagged discrete Arc would
hide the only real load target on that machine.

The Linux branch is covered here rather than on hardware because WSL exposes no
DRM cards at all (/sys/class/drm holds only `version`), so nothing on a Windows
dev box can exercise it for real.
"""

import os
from pathlib import Path

import pytest

from service.utils.system_monitor import SystemMonitor


def _card(root: Path, n: int, vendor: str, vram_node: str | None = None,
          vram_bytes: int = 0) -> None:
    device = root / f"card{n}" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor + "\n")
    if vram_node:
        node = device / vram_node
        node.parent.mkdir(parents=True, exist_ok=True)
        node.write_text(f"{vram_bytes}\n")


@pytest.fixture
def monitor():
    return SystemMonitor()


def test_linux_intel_igpu_has_no_vram_node(monitor, tmp_path, monkeypatch):
    """
    An Intel iGPU publishes no local-memory size, because it has none.

    sysfs cannot close the question on its own, though: a discrete card whose driver
    spells the node differently looks identical here. So this branch abstains and the
    verdict is the RAM-share fallback's to make -- see the _adapter_is_integrated test
    below, which checks it still lands on True for an iGPU.
    """
    _card(tmp_path, 0, "0x8086")
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("Intel(R) Graphics") is None


def test_linux_discrete_arc_is_not_integrated(monitor, tmp_path, monkeypatch):
    """A discrete Arc reports its framebuffer through the xe node -- keep it."""
    _card(tmp_path, 0, "0x8086", "tile0/physical_vram_size_bytes", 16 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("Intel(R) Arc(TM) Pro B70") is False


def test_linux_i915_discrete_lmem_is_not_integrated(monitor, tmp_path, monkeypatch):
    _card(tmp_path, 0, "0x8086", "lmem_total_bytes", 8 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("Intel(R) Arc(TM) A770") is False


def test_linux_amdgpu_discrete_is_not_integrated(monitor, tmp_path, monkeypatch):
    _card(tmp_path, 0, "0x1002", "mem_info_vram_total", 24 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("AMD Radeon RX 7900 XTX") is False


def test_linux_ignores_other_vendors_cards(monitor, tmp_path, monkeypatch):
    """An NVIDIA card next door must not make the Intel iGPU look discrete."""
    _card(tmp_path, 0, "0x8086")
    _card(tmp_path, 1, "0x10de", "mem_info_vram_total", 16 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    # The NVIDIA card is filtered out by vendor, leaving nothing that published a
    # size, so this abstains rather than reading the neighbour's 16 GiB as Intel's.
    assert monitor._linux_device_is_integrated("Intel(R) Graphics") is None


def test_linux_returns_none_without_sysfs(monitor, tmp_path, monkeypatch):
    """WSL: /sys/class/drm has no cards, so this branch must abstain, not guess."""
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("Intel(R) Graphics") is None


def test_ram_share_fallback_flags_borrowed_memory(monitor):
    """Measured shape of the bug: a 102 GB "GPU" on a 128 GB box is system RAM."""
    total_ram = __import__("psutil").virtual_memory().total

    assert monitor._memory_share_suggests_integrated(int(total_ram * 0.8)) is True
    assert monitor._memory_share_suggests_integrated(int(total_ram * 0.12)) is False
    assert monitor._memory_share_suggests_integrated(None) is False
    assert monitor._memory_share_suggests_integrated(0) is False


# --------------------------------------------------------------------------
# The DXGI enumeration path
#
# Which loop emits a given adapter depends on the torch build: with torch.xpu
# present the Intel adapter comes from the XPU loop, and without it from the DXGI
# loop. Only one runs on any install, so the DXGI verdict is tested directly
# rather than by arranging for that loop to be the one that fires.
# --------------------------------------------------------------------------

GIB = 1024**3


@pytest.mark.parametrize(
    "adapter, expected",
    [
        # Discrete cards report a real framebuffer.
        ({"vendor_id": 0x10DE, "dedicated_video_bytes": 16 * GIB,
          "shared_system_bytes": 64 * GIB}, ("nvidia", False)),
        ({"vendor_id": 0x1002, "dedicated_video_bytes": 24 * GIB,
          "shared_system_bytes": 64 * GIB}, ("amd", False)),
        # A discrete Intel Arc must stay a load target.
        ({"vendor_id": 0x8086, "dedicated_video_bytes": 16 * GIB,
          "shared_system_bytes": 64 * GIB}, ("intel", False)),
        # An iGPU: no dedicated memory to speak of, a huge shared pool.
        ({"vendor_id": 0x8086, "dedicated_video_bytes": 128 * 1024**2,
          "shared_system_bytes": 64 * GIB}, ("intel", True)),
        ({"vendor_id": 0x8086, "dedicated_video_bytes": 0,
          "shared_system_bytes": 100 * GIB}, ("intel", True)),
        # Unknown vendor is reported as such, not guessed.
        ({"vendor_id": 0x1414, "dedicated_video_bytes": 0,
          "shared_system_bytes": 8 * GIB}, ("unknown", True)),
        # Missing keys must not raise.
        ({}, ("unknown", False)),
    ],
)
def test_describe_dxgi_adapter(monitor, adapter, expected):
    assert monitor._describe_dxgi_adapter(adapter) == expected


def test_dxgi_verdict_agrees_with_the_name_based_lookup(monitor, monkeypatch):
    """The two Windows paths must not disagree about the same adapter."""
    adapters = [
        {"name": "Intel(R) Graphics", "vendor_id": 0x8086,
         "dedicated_video_bytes": 0, "shared_system_bytes": 100 * GIB},
        {"name": "NVIDIA GeForce RTX 5060 Ti", "vendor_id": 0x10DE,
         "dedicated_video_bytes": 16 * GIB, "shared_system_bytes": 64 * GIB},
    ]
    monkeypatch.setattr(SystemMonitor, "_get_windows_dxgi_adapters",
                        lambda self: adapters)
    monkeypatch.setattr(os, "name", "nt")

    for adapter in adapters:
        _, integrated = monitor._describe_dxgi_adapter(adapter)
        assert integrated == monitor._windows_adapter_is_integrated(adapter["name"])


# --------------------------------------------------------------------------
# Two cards of the same vendor
#
# A discrete Arc and an Intel iGPU are both 0x8086, so the vendor stops
# identifying the adapter. Answering from the first card that happened to have
# VRAM marked the iGPU discrete as well, and because the budget takes the
# largest free device, that adapter's ~100 GB of borrowed RAM then won.
# --------------------------------------------------------------------------


def test_linux_mixed_intel_box_matches_the_adapter_by_its_reported_size(
    monitor, tmp_path, monkeypatch
):
    """Only the real card reports a size sysfs also publishes."""
    _card(tmp_path, 0, "0x8086")  # iGPU: no local memory
    _card(tmp_path, 1, "0x8086", "lmem_total_bytes", 16 * 1024**3)  # discrete Arc
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    arc = monitor._linux_device_is_integrated("Intel(R) Arc(TM) A770", 16 * 1024**3)
    igpu = monitor._linux_device_is_integrated("Intel(R) Graphics", 100 * 1024**3)

    assert arc is False
    assert igpu is True


def test_linux_mixed_box_tolerates_the_driver_reserving_vram(monitor, tmp_path, monkeypatch):
    """torch never sees the whole framebuffer, so a few percent short is still that card."""
    _card(tmp_path, 0, "0x8086")
    _card(tmp_path, 1, "0x8086", "tile0/physical_vram_size_bytes", 16 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    reported = int(16 * 1024**3 * 0.93)

    assert monitor._linux_device_is_integrated("Intel(R) Arc(TM) B580", reported) is False


def test_linux_mixed_box_abstains_without_a_size(monitor, tmp_path, monkeypatch):
    """Nothing to match on: fall through to the caller's last resort rather than guess."""
    _card(tmp_path, 0, "0x8086")
    _card(tmp_path, 1, "0x8086", "lmem_total_bytes", 16 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("Intel(R) Graphics") is None


def test_linux_two_discrete_cards_stay_discrete(monitor, tmp_path, monkeypatch):
    """Agreeing cards need no size at all -- the vendor answer is already right."""
    _card(tmp_path, 0, "0x10de", "mem_info_vram_total", 16 * 1024**3)
    _card(tmp_path, 1, "0x10de", "mem_info_vram_total", 24 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("NVIDIA GeForce RTX 5060 Ti") is False


# --------------------------------------------------------------------------
# A node name this code has never seen
#
# _LINUX_VRAM_NODES is an allowlist of three spellings, and every discrete case
# above builds its synthetic card from a name on that list -- so the allowlist is
# only ever asked to recognise itself. Treating "no name I recognise" as "no local
# memory" made an unlisted discrete card read as an iGPU, and as a hard True, which
# skipped the RAM-share fallback that would have caught it. These cases publish
# names deliberately absent from the list, so they fail if the lookup goes back to
# matching on names alone.
# --------------------------------------------------------------------------


def test_linux_discrete_card_with_an_unlisted_node_name_is_not_integrated(
    monitor, tmp_path, monkeypatch
):
    """A framebuffer is a framebuffer whatever the driver called the file."""
    assert "tile0/vram_size_bytes" not in SystemMonitor._LINUX_VRAM_NODES
    _card(tmp_path, 0, "0x8086", "tile0/vram_size_bytes", 16 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("Intel(R) Arc(TM) Pro B70") is False


def test_linux_unlisted_node_at_the_device_root_is_found(monitor, tmp_path, monkeypatch):
    """Not every driver puts the size under a tile."""
    assert "local_mem_total_bytes" not in SystemMonitor._LINUX_VRAM_NODES
    _card(tmp_path, 0, "0x1002", "local_mem_total_bytes", 24 * 1024**3)
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("AMD Radeon RX 7900 XTX") is False


def test_linux_card_vram_prefers_the_total_over_a_usage_sibling(monitor, tmp_path):
    """
    The mixed-vendor branch matches a reported size against this number, so a usage
    figure winning the glob would match no card and flip that verdict.
    """
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "mem_info_vram_used").write_text(f"{3 * 1024**3}\n")
    (device / "mem_info_vram_grand_total").write_text(f"{24 * 1024**3}\n")

    assert monitor._linux_card_vram_bytes(device) == 24 * 1024**3


def test_linux_card_vram_ignores_a_node_it_cannot_parse(monitor, tmp_path):
    """A vendor string or a directory under the same glob must not read as a size."""
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x8086\n")
    (device / "mem_info_vram_vendor").write_text("intel\n")
    (device / "vram_stats").mkdir()

    assert monitor._linux_card_vram_bytes(device) == 0


def test_adapter_verdict_falls_back_to_ram_share_when_sysfs_abstains(
    monitor, tmp_path, monkeypatch
):
    """
    _linux_device_is_integrated answering None is only safe if the layer above it
    still reaches the right answer, so pin both directions through the public entry
    point rather than through the branch alone.
    """
    _card(tmp_path, 0, "0x8086")  # publishes no size either way
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)
    monkeypatch.setattr(os, "name", "posix")
    total_ram = __import__("psutil").virtual_memory().total

    igpu = monitor._adapter_is_integrated("Intel(R) Graphics", int(total_ram * 0.8))
    discrete = monitor._adapter_is_integrated("Intel(R) Arc(TM) Pro B70", int(total_ram * 0.12))

    assert igpu is True
    assert discrete is False


def test_linux_card_vram_ignores_the_system_ram_aperture(monitor, tmp_path):
    """
    amdgpu's mem_info_gtt_total sizes the system-RAM aperture, not the card. Counting
    it would report local memory for an adapter that has none -- calling an iGPU
    discrete, which is the direction this whole lookup exists to prevent.
    """
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "mem_info_gtt_total").write_text(f"{64 * 1024**3}\n")

    assert monitor._linux_card_vram_bytes(device) == 0


# --------------------------------------------------------------------------
# Which "total" is the framebuffer
#
# A driver publishes more than one total, and only the largest is local memory:
# the others are windows onto it. amdgpu's mem_info_vis_vram_total is the
# CPU-visible aperture -- 256 MiB on a card without resizable BAR, against a 16
# GiB framebuffer. Picking by name gave the aperture the answer, because
# "vis_vram" precedes "vram", and the size the mixed-vendor branch matches
# against was then wrong by two orders of magnitude.
# --------------------------------------------------------------------------


def test_linux_card_vram_picks_the_framebuffer_over_the_visible_aperture(monitor, tmp_path):
    """The aperture sorts first by name, so name order must not decide this."""
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    # Neither spelling is on the allowlist, so the shape lookup has to choose.
    (device / "mem_info_vis_vram_total").write_text(f"{256 * 1024**2}\n")
    (device / "mem_info_vram_size_total").write_text(f"{16 * 1024**3}\n")

    assert monitor._linux_card_vram_bytes(device) == 16 * 1024**3


def test_linux_mixed_box_matches_the_framebuffer_not_the_aperture(monitor, tmp_path, monkeypatch):
    """
    What the wrong pick costs: the reported size is matched against this number, so
    an aperture in its place leaves a discrete card matching nothing and reading as
    the integrated one.
    """
    _card(tmp_path, 0, "0x8086")  # iGPU: no local memory
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x8086\n")
    (device / "vis_vram_total_bytes").write_text(f"{256 * 1024**2}\n")
    (device / "vram_size_total_bytes").write_text(f"{16 * 1024**3}\n")
    monkeypatch.setattr(SystemMonitor, "_DRM_ROOT", tmp_path)

    assert monitor._linux_device_is_integrated("Intel(R) Arc(TM) Pro B70", 16 * 1024**3) is False


def test_linux_card_vram_keeps_the_total_tier_ahead_of_a_larger_usage_node(monitor, tmp_path):
    """Largest only breaks ties inside a tier -- a usage figure never outranks a total."""
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "vram_size_total_bytes").write_text(f"{8 * 1024**3}\n")
    (device / "vram_reserved_bytes").write_text(f"{20 * 1024**3}\n")  # bigger, but not a total

    assert monitor._linux_card_vram_bytes(device) == 8 * 1024**3


def test_linux_card_vram_reports_one_tile_not_the_sum(monitor, tmp_path):
    """
    A multi-tile card publishes a size per tile, and the listed tile0 node yields one
    of them. The shape lookup must agree, or the same card reads at double the size
    depending on which spelling its driver happens to use.
    """
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x8086\n")
    for tile in ("tile0", "tile1"):
        (device / tile).mkdir()
        (device / tile / "vram_size_bytes").write_text(f"{16 * 1024**3}\n")

    assert monitor._linux_card_vram_bytes(device) == 16 * 1024**3
