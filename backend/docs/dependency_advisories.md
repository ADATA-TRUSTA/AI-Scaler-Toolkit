# Known dependency advisories

Advisories that a scan (osv-scanner 2.7.0, 2026-10-27) reports against packages this project installs,
reviewed and left in place on purpose: no fixed release exists yet, or one exists but another dependency
does not allow it yet. Each entry says how the package is used here, why the advisory is not reachable from
that use, and when to look again.

Recheck an entry when a fixed release becomes installable (then upgrade), or when the usage described here
changes.

繁體中文版：[dependency_advisories.zh-TW.md](dependency_advisories.zh-TW.md)

---

## chromadb 1.5.9 — optional, the `chroma` extra

**Not installed by default.** RAG keeps documents in SQLite and searches them with FTS5 (falling back to
`LIKE`), optionally ranked with sentence-transformers embeddings. Chroma is an alternative vector store
for the same `/rag/*` routes.

To use it, add the extra to the ones `setup_env` picked, then turn it on in `.env`:

```bash
uv sync --extra cuda --extra vllm --extra chroma   # keep the extras this host already uses
```

```bash
RAG_BACKEND=chroma        # or RAG_USE_CHROMA=true
```

With either setting and no `chroma` extra, the service logs a warning at startup and keeps using SQLite.

### Open advisories

| Advisory | Summary |
|---|---|
| GHSA-f4j7-r4q5-qw2c / PYSEC-2026-311 | Pre-authentication code injection through the collections endpoint of Chroma's server |
| GHSA-36p7-vc44-83pf / PYSEC-2026-3814 | Authenticated code injection through the collection update endpoint |
| GHSA-2wm9-hf6c-p5cr / PYSEC-2026-3813 | Any authenticated user can read and write any tenant's collections |
| GHSA-xph7-9rjv-w5fr / PYSEC-2026-3815 | `SimpleRBACAuthorizationProvider` does not check which tenant a permission applies to |

### Why they are not reachable here

All four are in Chroma's own HTTP server: its `/api/v2/...` endpoints and its authentication and
authorization providers. This project never starts that server. `service/rag_manager.py` uses
`chromadb.PersistentClient`, which runs in the service's own process against a local folder; no Chroma
port is opened. The project's `/rag/*` routes are local-only unless `SERVICE_REMOTE_ACCESS=all`.

This stops holding if someone runs `chroma run` (Chroma's standalone server) next to the service, or
replaces `PersistentClient` with `HttpClient` pointed at a shared Chroma server.

---

## accelerate 1.14.0 — always installed

| Advisory | Summary |
|---|---|
| GHSA-4j2p-28q2-5m79 / PYSEC-2026-3804 | Path traversal and denial of service through `weight_map` entries of a sharded checkpoint index |

The flaw is in `load_checkpoint_in_model` and `load_checkpoint_and_dispatch`: they read shard paths from
the checkpoint's index file without checking them, so a crafted checkpoint can point at any file (`../`,
absolute paths) or at a named pipe that blocks the load forever.

### How it is used here

- `infer_auto_device_map` — splits a model's layers across GPU and CPU
  (`service/inference/engines/transformers_engine.py`, `service/inference/model_utils.py`).
- `init_empty_weights` — builds a model without allocating its weights, for the memory estimate
  (`service/inference/memory_estimator.py`).
- `AlignDevicesHook`, `SequentialHook` — to find where an offloaded base model lives when a LoRA adapter
  is attached (`service/inference/load_model/peft_loader.py`).

### Why it is not reachable here

Neither function is called by this project, nor by transformers, peft, trl, sentence-transformers or
bitsandbytes at the versions in `uv.lock` (checked by searching their installed sources on 2026-10-27).
transformers loads sharded checkpoints with its own code. Even through those functions, the attack needs
the user to load a checkpoint someone tampered with on this machine.

This stops holding if code here starts calling either function, or if an upgrade of transformers or peft
starts routing loads through them — search the installed packages again when upgrading those.

---

## setuptools 80.10.2 — fixed upstream, held back by vllm

| Advisory | Summary | Fixed in |
|---|---|---|
| GHSA-h35f-9h28-mq5c / CVE-2026-59890 / PYSEC-2026-3447 | `MANIFEST.in` exclusions miss files whose names differ only in Unicode normalization (NFC/NFD) on macOS APFS/HFS+, so an sdist can ship a file meant to be left out | 83.0.0 |
| GHSA-xmj3-9gf7-h52w | On Windows, a wheel's `namespace_packages.txt` can make setuptools' legacy wheel install (`Wheel.install_as_egg`) write an `__init__.py` outside its target directory | 84.1.0 |

Unlike the entries above, fixed releases exist. They cannot be installed yet: setuptools is not pinned here
but comes in through torch and vllm, and vllm 0.32.0 asks for `setuptools<81`. uv therefore resolves
80.10.2 for every platform, Windows included (through torch).

### How it is used here

No code in `service/` or `scripts/` imports `setuptools` or `pkg_resources`. It is only used when a package
is built from source, and only two in `uv.lock` are, both Linux-only: `deepspeed` (in uv's isolated build
environment, with its own setuptools rather than this one) and `causal-conv1d` (listed in
`no-build-isolation-package`, so it does build with the installed 80.10.2).

### Why it is not reachable here

- The sdist advisory is about *making* an sdist for publishing, on macOS. This project does not build or
  publish an sdist, and macOS is not a supported host.
- The Windows advisory needs setuptools itself to install a wheel. In the installed setuptools the only path
  there is `fetch_build_eggs`, for a legacy `setup_requires` during a source build. uv installs wheels with
  its own installer, and on Windows nothing here is built from source against the installed setuptools
  (checked in the Windows venv on 2026-10-27). The one build that uses it, `causal-conv1d`, happens only on
  Linux.

This stops holding if vllm allows `setuptools>=81` (then upgrade to 84.1.0 or later at once — check this on
every vllm upgrade), if a Windows install starts building a package from source without build isolation,
or if this project starts building and publishing an sdist.
