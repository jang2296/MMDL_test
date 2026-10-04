#!/usr/bin/env bash
# Private Pod-only preparation; no GPU/model loading, cloud calls, or Qwen changes.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
setup_root=/workspace/gemma-verifier
min_free_gib=64
execute=false
deadline=2026-10-03T04:22:38Z
while (($#)); do
  case "$1" in
    --root) setup_root=${2:?missing --root value}; shift 2 ;;
    --min-free-gib) min_free_gib=${2:?missing --min-free-gib value}; shift 2 ;;
    --deadline-utc) deadline=${2:?missing --deadline-utc value}; shift 2 ;;
    --execute) execute=true; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
remaining=$(python3 - "$deadline" <<'PY'
from datetime import datetime, timezone
import math, sys
print(max(0, math.floor((datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00")) - datetime.now(timezone.utc)).total_seconds())))
PY
)
if [[ "$execute" == true ]]; then
  if ((remaining <= 0)); then printf 'Global deadline expired; no installation/download started.\n' >&2; exit 124; fi
  prefix=(timeout --signal=TERM --kill-after=10s "${remaining}s")
else
  prefix=()
fi
"${prefix[@]}" python3 - "$script_dir/deployment.json" "$setup_root" "$min_free_gib" "$execute" "$deadline" <<'PY'
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request

spec_path, root_arg, free_arg, execute, deadline_arg = sys.argv[1:]
spec_path = Path(spec_path)
spec = json.loads(spec_path.read_text())
root = Path(root_arg).resolve()
minimum = int(free_arg)
if (not Path(root_arg).is_absolute() or not root.is_relative_to(Path("/workspace"))
        or root == Path("/workspace") or minimum < 32):
    raise SystemExit("Use a dedicated absolute /workspace subtree and --min-free-gib >=32")
deadline = datetime.datetime.fromisoformat(deadline_arg.replace("Z", "+00:00")).timestamp()
venv, model, state = root / "venv", root / "model", root / "state"
python = venv / "bin/python"
wheel = spec["vllm"]["wheel_url"] + "#sha256=" + spec["vllm"]["wheel_sha256"]
packages = [wheel] + [f"{name}=={version}" for name, version in spec["required_distribution_versions"].items()]
install = [str(python), "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
           "--index-url", "https://pypi.org/simple", "--extra-index-url",
           "https://download.pytorch.org/whl/cu129", *packages]
print(json.dumps({"mode": "EXECUTE" if execute == "true" else "DRY_RUN_NO_INSTALL_NO_DOWNLOAD",
                  "root": str(root), "install_command": install,
                  "model_id": spec["model_id"], "revision": spec["model_revision"],
                  "model_files": list(spec["expected_model_files"]),
                  "expected_model_bytes": sum(x["bytes"] for x in spec["expected_model_files"].values()),
                  "minimum_free_gib": minimum, "preserved_headroom_gib": 10,
                  "deadline_utc": deadline_arg, "server_started": False}, sort_keys=True), flush=True)
if execute != "true":
    sys.exit(0)
if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("Pod setup requires Linux x86_64 Python3.12")

def remaining():
    seconds = deadline - datetime.datetime.now(datetime.timezone.utc).timestamp()
    if seconds <= 0:
        raise SystemExit("BUDGET_STOP: global deadline expired")
    return seconds

def run(args, capture=False):
    return subprocess.run(args, check=True, timeout=remaining(), text=True,
                          stdout=subprocess.PIPE if capture else None).stdout

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def freeze(path, value):
    if path.exists():
        if path.read_bytes() != value:
            raise SystemExit(f"Refusing changed setup receipt: {path}")
        return
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())

def free_check(required):
    available = shutil.disk_usage(root).free
    if available < required:
        raise SystemExit(f"Insufficient private workspace disk: free={available}, required={required}; no deletion/fallback")

if root.exists() and any(root.iterdir()) and not (state / "setup_identity.json").is_file():
    raise SystemExit("Existing nonempty root has no Gemma setup ownership receipt; refusing changes")
root.mkdir(parents=True, exist_ok=True)
owner_lock = (root / ".setup.lock").open("a")
try:
    fcntl.flock(owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("Another Gemma setup owns this root")
if any(path.is_symlink() for path in (venv, model, state)):
    raise SystemExit("Private setup subdirectories must not be symlinks")
state.mkdir(exist_ok=True)
model.mkdir(exist_ok=True)
freeze(state / "setup_identity.json", (json.dumps({"deployment_sha256": sha(spec_path),
       "setup_script_sha256": sha(spec_path.with_name("runpod_setup.sh")),
       "root": str(root), "deadline_utc": deadline_arg}, sort_keys=True) + "\n").encode())
environment_receipt = state / "setup_environment.json"
if not environment_receipt.exists():
    free_check(minimum * 1024**3)
    if not venv.exists():
        run([sys.executable, "-m", "venv", str(venv)])
    elif not (venv / "pyvenv.cfg").is_file():
        raise SystemExit("Existing private venv is malformed; refusing replacement")
    run(install)
run([str(python), "-m", "pip", "check"])
# Native imports are install/ABI evidence only, not GPU or full-context validation.
probe = run([str(python), "-c", '''
import importlib.metadata as m, json, sys
import torch, torchaudio, torchvision, transformers, huggingface_hub, vllm
names = ["torch", "torchaudio", "torchvision", "transformers", "huggingface-hub", "vllm"]
distribution = m.distribution("vllm")
print(json.dumps({"versions": {n: m.version(n) for n in names}, "torch_cuda": torch.version.cuda,
 "prefix": sys.prefix, "vllm_direct_url": json.loads(distribution.read_text("direct_url.json") or "{}"),
 "native_imports_passed": True, "gpu_tested": False, "server_started": False}, sort_keys=True))
'''], capture=True)
observed = json.loads(probe)
expected = spec["required_distribution_versions"] | {"vllm": spec["vllm"]["version"]}
if observed["versions"] != expected or observed["torch_cuda"] != spec["torch_cuda_runtime"] or Path(observed["prefix"]).resolve() != venv:
    raise SystemExit("Installed versions/runtime/prefix differ from the pinned isolated candidate")
archive = observed["vllm_direct_url"].get("archive_info", {})
digest = archive.get("hashes", {}).get("sha256", archive.get("hash", "").removeprefix("sha256="))
if digest != spec["vllm"]["wheel_sha256"]:
    raise SystemExit("Installed vLLM direct wheel digest differs")
lock = run([str(python), "-m", "pip", "freeze", "--all"], capture=True).encode()
observed["resolved_lock_sha256"] = hashlib.sha256(lock).hexdigest()
freeze(state / "requirements.resolved.txt", lock)
freeze(environment_receipt, (json.dumps(observed, sort_keys=True) + "\n").encode())
inventory = []
for name, expected_file in sorted(spec["expected_model_files"].items()):
    remaining()
    target = model / name
    if not target.exists():
        free_check(expected_file["bytes"] + 10 * 1024**3)
        url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['model_revision']}/{name}?download=true"
        # Interrupted partial downloads are preserved; a new attempt never overwrites them.
        with tempfile.NamedTemporaryFile(prefix=name + ".partial-", dir=model, delete=False) as stream:
            partial = Path(stream.name)
            checksum, count = hashlib.sha256(), 0
            request = urllib.request.Request(url, headers={"User-Agent": "private-gemma-verifier/1"})
            with urllib.request.urlopen(request, timeout=min(30, remaining())) as response:
                while True:
                    remaining()
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > expected_file["bytes"]:
                        raise SystemExit(f"Pinned download exceeds expected size: {name}")
                    stream.write(block)
                    checksum.update(block)
            stream.flush()
            os.fsync(stream.fileno())
        if count != expected_file["bytes"] or checksum.hexdigest() != expected_file["sha256"]:
            raise SystemExit(f"Pinned download size/SHA mismatch; preserved partial {partial}")
        partial.rename(target)
    actual = {"path": name, "bytes": target.stat().st_size, "sha256": sha(target)}
    if actual["bytes"] != expected_file["bytes"] or actual["sha256"] != expected_file["sha256"]:
        raise SystemExit(f"Existing pinned model file mismatch: {name}; refusing overwrite")
    inventory.append(actual)
    free_check(10 * 1024**3)
freeze(state / "downloaded_model_files.json", (json.dumps(inventory, sort_keys=True) + "\n").encode())
receipt = {"status": "INSTALL_NATIVE_IMPORT_AND_DOWNLOAD_VERIFIED_NOT_GPU_READY",
           "deployment_sha256": sha(spec_path), "revision": spec["model_revision"],
           "resolved_lock_sha256": hashlib.sha256(lock).hexdigest(),
           "model_inventory_sha256": sha(state / "downloaded_model_files.json"),
           "model_path": str(model), "venv": str(venv), "deadline_utc": deadline_arg,
           "server_started": False, "full_context_gpu_tested": False}
freeze(state / "setup_receipt.json", (json.dumps(receipt, sort_keys=True) + "\n").encode())
print(f"Verified setup only. Activate {venv}/bin/activate; then use start_server.sh --model-path {model} --output {root}/server-state --execute after Qwen releases the GPU.", flush=True)
PY
