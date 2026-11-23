"""
scripts/install_flashinfer_jit_cache.py: which prebuilt FlashInfer kernels a machine gets, and that a
failure leaves the environment as it found it.

The pair is a shim plus a per-GPU provider with identical versions. A provider without the shim
crashes `import flashinfer`, and the shim as a plain dependency drags in every provider, so the
install command and the rollback are what these tests pin. Nothing here touches the network, uv
or a GPU: the module's own seams (installed_version, gpu_capabilities, install, verify) are
replaced.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.inference]

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "install_flashinfer_jit_cache.py"


@pytest.fixture
def jit():
    spec = importlib.util.spec_from_file_location("install_flashinfer_jit_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def machine(jit, monkeypatch):
    """A CUDA host with flashinfer 0.7.0.post1 / torch +cu130, an sm120 GPU and nothing installed."""
    state = {"installed": {}, "calls": [], "verify": None, "install_ok": True}
    versions = {"flashinfer-python": "0.7.0.post1", "torch": "2.13.0+cu130"}

    monkeypatch.delenv("TRUSTA_FLASHINFER_JIT_CACHE", raising=False)
    monkeypatch.delenv("TRUSTA_FLASHINFER_JIT_CACHE_INDEX", raising=False)
    monkeypatch.delenv("TRUSTA_FLASHINFER_JIT_ARCH", raising=False)
    monkeypatch.setattr(jit, "installed_version", lambda name: {**versions, **state["installed"]}.get(name))
    monkeypatch.setattr(jit, "gpu_capabilities", lambda: ["12.0"])
    monkeypatch.setattr(jit, "find_uv", lambda: "/usr/bin/uv")

    def fake_install(uv, index, version, providers):
        state["calls"].append(("install", index, version, list(providers)))
        if state["install_ok"]:
            for name in ["flashinfer-jit-cache", *(f"flashinfer-jit-cache-{p}" for p in providers)]:
                state["installed"][name] = version
        return state["install_ok"]

    def fake_uninstall(uv, providers):
        state["calls"].append(("uninstall", list(providers)))
        state["installed"].clear()

    monkeypatch.setattr(jit, "install", fake_install)
    monkeypatch.setattr(jit, "uninstall", fake_uninstall)
    monkeypatch.setattr(jit, "verify", lambda providers, version: state["verify"])
    return state


def test_each_capability_maps_to_its_provider_and_unknown_ones_have_none(jit):
    assert jit.providers_for(["12.0", "8.9", "12.0"]) == (["sm120f", "sm89"], [])
    assert jit.providers_for(["8.6", "12.0"]) == (["sm120f"], ["8.6"])
    assert jit.providers_for([]) == ([], [])


def test_the_provider_table_is_what_the_shim_requires(jit):
    assert jit.PROVIDERS == {"sm80", "sm89", "sm90a", "sm100a", "sm103a", "sm120f"}


def test_arch_override_rejects_unknown_names(jit):
    assert jit.parse_arch_list("sm120f, SM89,sm120f") == ["sm120f", "sm89"]
    with pytest.raises(ValueError, match="sm86"):
        jit.parse_arch_list("sm86")


def test_cuda_tag_comes_from_torchs_local_label(jit, monkeypatch):
    monkeypatch.setattr(jit, "installed_version", lambda name: "2.13.0+cu130")
    assert jit.cuda_tag() == "cu130"
    monkeypatch.setattr(jit, "installed_version", lambda name: "2.13.0+xpu")
    assert jit.cuda_tag() is None
    monkeypatch.setattr(jit, "installed_version", lambda name: None)
    assert jit.cuda_tag() is None


def test_installs_the_shim_and_the_gpus_provider_at_flashinferos_version(jit, machine):
    assert jit.main([]) == jit.EXIT_OK
    assert machine["calls"] == [("install", "https://flashinfer.ai/whl/cu130", "0.7.0.post1+cu130", ["sm120f"])]


def test_install_command_is_no_deps_with_the_shim_first(jit, monkeypatch):
    seen = {}

    class Done:
        returncode = 0

    monkeypatch.setattr(jit.subprocess, "run", lambda command, check: seen.update(command=command) or Done())
    assert jit.install("uv", "https://idx", "0.7.0.post1+cu130", ["sm120f", "sm89"])
    command = seen["command"]
    assert "--no-deps" in command
    assert command[command.index("--index-url") + 1] == "https://idx"
    assert command[-3:] == [
        "flashinfer-jit-cache==0.7.0.post1+cu130",
        "flashinfer-jit-cache-sm120f==0.7.0.post1+cu130",
        "flashinfer-jit-cache-sm89==0.7.0.post1+cu130",
    ]


def test_second_run_installs_nothing(jit, machine):
    assert jit.main([]) == jit.EXIT_OK
    machine["calls"].clear()
    assert jit.main([]) == jit.EXIT_OK
    assert machine["calls"] == []


def test_a_version_mismatch_is_reinstalled(jit, machine):
    machine["installed"].update({"flashinfer-jit-cache": "0.6.0+cu130", "flashinfer-jit-cache-sm120f": "0.6.0+cu130"})
    assert jit.main([]) == jit.EXIT_OK
    assert machine["calls"][0][0] == "install"


def test_a_pair_that_fails_verification_is_removed_again(jit, machine):
    machine["verify"] = "FlashInfer did not discover flashinfer-jit-cache-sm120f"
    assert jit.main([]) == jit.EXIT_COMPILES
    assert [call[0] for call in machine["calls"]] == ["install", "uninstall"]
    assert machine["installed"] == {}


def test_a_failed_download_leaves_things_as_they_were(jit, machine):
    machine["install_ok"] = False
    assert jit.main([]) == jit.EXIT_COMPILES
    assert [call[0] for call in machine["calls"]] == ["install"]


def test_a_gpu_without_a_provider_installs_nothing(jit, machine, monkeypatch):
    monkeypatch.setattr(jit, "gpu_capabilities", lambda: ["8.6"])
    assert jit.main([]) == jit.EXIT_COMPILES
    assert machine["calls"] == []


def test_no_visible_gpu_installs_nothing_unless_told_which(jit, machine, monkeypatch):
    monkeypatch.setattr(jit, "gpu_capabilities", lambda: [])
    assert jit.main([]) == jit.EXIT_COMPILES
    assert machine["calls"] == []
    monkeypatch.setenv("TRUSTA_FLASHINFER_JIT_ARCH", "sm89")
    assert jit.main([]) == jit.EXIT_OK
    assert machine["calls"][0][3] == ["sm89"]


def test_several_gpu_families_each_get_a_provider(jit, machine, monkeypatch):
    monkeypatch.setattr(jit, "gpu_capabilities", lambda: ["12.0", "8.9"])
    assert jit.main([]) == jit.EXIT_OK
    assert machine["calls"][0][3] == ["sm120f", "sm89"]


def test_check_reports_without_installing(jit, machine):
    assert jit.main(["--check"]) == 1
    assert machine["calls"] == []
    jit.main([])
    machine["calls"].clear()
    assert jit.main(["--check"]) == jit.EXIT_OK


def test_the_index_can_be_pointed_at_a_mirror(jit, machine, monkeypatch):
    monkeypatch.setenv("TRUSTA_FLASHINFER_JIT_CACHE_INDEX", "https://mirror.example/whl")
    jit.main([])
    assert machine["calls"][0][1] == "https://mirror.example/whl"


def test_switched_off_or_not_applicable_does_nothing(jit, machine, monkeypatch):
    monkeypatch.setenv("TRUSTA_FLASHINFER_JIT_CACHE", "0")
    assert jit.main([]) == jit.EXIT_NOT_APPLICABLE
    monkeypatch.delenv("TRUSTA_FLASHINFER_JIT_CACHE")
    machine["installed"]["torch"] = "2.13.0+xpu"
    assert jit.main([]) == jit.EXIT_NOT_APPLICABLE
    machine["installed"]["torch"] = "2.13.0+cu130"
    machine["installed"]["flashinfer-python"] = None
    assert jit.main([]) == jit.EXIT_NOT_APPLICABLE
    assert machine["calls"] == []
