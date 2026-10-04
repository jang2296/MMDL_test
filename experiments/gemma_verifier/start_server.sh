#!/usr/bin/env bash
# Private Pod-only launcher. No installer, downloader, GPU/context fallback, or cloud API.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model_path=""
output_path=""
port=8000
execute=false
while (($#)); do
  case "$1" in
    --model-path) model_path=${2:?missing --model-path value}; shift 2 ;;
    --output) output_path=${2:?missing --output value}; shift 2 ;;
    --port) port=${2:?missing --port value}; shift 2 ;;
    --execute) execute=true; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ -z "$model_path" || -z "$output_path" || ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  printf 'Usage: bash start_server.sh --model-path PINNED_SNAPSHOT --output PRIVATE_STATE_DIR [--port 8000] [--execute]\n' >&2
  exit 2
fi
command=(python -m vllm.entrypoints.cli.main serve "$model_path"
  --served-model-name google/gemma-4-E4B-it
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2
  --tokenizer "$model_path" --tokenizer-revision ee0ef6023621cff504d758262d4e04895a5af4a2
  --dtype bfloat16 --max-model-len 131072 --max-num-seqs 1
  --max-num-batched-tokens 4096 --enable-chunked-prefill --enforce-eager
  --gpu-memory-utilization 0.90 --language-model-only
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
  --generation-config vllm --override-generation-config '{"temperature":0,"top_p":1,"top_k":-1}'
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --chat-template-content-format string
  --host 127.0.0.1 --port "$port")
if [[ "$execute" != true ]]; then
  printf 'DRY RUN only: '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
python - "$script_dir/deployment.json" "$model_path" "$output_path" "$port" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

deployment_file, snapshot_arg, output_arg, port = sys.argv[1:]
spec = json.loads(Path(deployment_file).read_text())
snapshot, output = Path(snapshot_arg).resolve(), Path(output_arg).resolve()
if not snapshot.is_dir() or output == snapshot or snapshot.is_relative_to(output):
    raise SystemExit("Use an existing pinned snapshot and a separate private output directory")
if sys.version_info[:2] != (3, 12) or platform.machine() != "x86_64" or platform.system() != "Linux":
    raise SystemExit("Pinned deployment requires Linux x86_64 Python3.12")
versions = {name: importlib.metadata.version(name) for name in spec["required_distribution_versions"]}
if versions != spec["required_distribution_versions"]:
    raise SystemExit(f"Dependency versions differ from frozen candidate: {versions}")
dist = importlib.metadata.distribution("vllm")
if dist.version != spec["vllm"]["version"]:
    raise SystemExit("Installed vLLM is not the selected CUDA12.9 release")
direct = json.loads(dist.read_text("direct_url.json") or "{}")
archive = direct.get("archive_info", {})
wheel_hash = archive.get("hashes", {}).get("sha256")
if wheel_hash is None and archive.get("hash", "").startswith("sha256="):
    wheel_hash = archive["hash"].split("=", 1)[1]
if wheel_hash != spec["vllm"]["wheel_sha256"]:
    raise SystemExit("Install the exact digest-pinned direct wheel URL; index installs do not prove its identity")

def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

for name, expected in spec["package_source_sha256"].items():
    if sha(Path(dist.locate_file(name))) != expected:
        raise SystemExit(f"Installed vLLM source differs from pinned commit: {name}")
inventory = []
for name, expected in sorted(spec["expected_model_files"].items()):
    path = snapshot / name
    observed = {"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}
    if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
        raise SystemExit(f"Downloaded model differs from pinned revision: {name}")
    inventory.append(observed)
config = json.loads((snapshot / "config.json").read_text())
if config.get("quantization_config") or config.get("dtype") != "bfloat16":
    raise SystemExit("Only the pinned BF16 nonquantized checkpoint is allowed")
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
import torch
import torchaudio  # Required package/native import checked despite the text-only workload.
import torchvision
import vllm
if torch.version.cuda != spec["torch_cuda_runtime"]:
    raise SystemExit("Torch CUDA runtime differs from12.9; no runtime downgrade/fallback")
if torch.cuda.device_count() != 1 or "RTX 3090" not in torch.cuda.get_device_name(0):
    raise SystemExit("This untested candidate is scoped to one RTX3090, after Qwen has released it")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 is unsupported; no precision fallback")
probe = torch.ones((2, 2), dtype=torch.bfloat16, device="cuda")
assert (probe @ probe).dtype == torch.bfloat16
torch.cuda.synchronize()
del probe
torch.cuda.empty_cache()
output.mkdir(parents=True, exist_ok=True)
inventory_bytes = (json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode()
lock_bytes = freeze.encode()
identity = {
    "model_id": spec["model_id"], "revision": spec["model_revision"],
    "tokenizer_revision": spec["model_revision"], "frozen_weights": True,
    "weights_manifest_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
    "chat_template_sha256": sha(snapshot / "chat_template.jinja"),
    "environment_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
    "dtype": "bfloat16", "max_model_len": 131072, "vllm_version": vllm.__version__,
    "deployment_sha256": sha(Path(deployment_file)), "snapshot": str(snapshot),
    "endpoint": f"http://127.0.0.1:{port}/v1", "wheel_sha256": wheel_hash,
    "chat_template_content_format": "string",
    "server_started": False, "full_context_gpu_validated": False,
}
environment = {"versions": versions, "vllm_version": vllm.__version__,
               "python": platform.python_version(), "platform": platform.platform(),
               "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
               "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True),
               "native_imports_and_small_bf16_passed": True, "full_context_memory_passed": False}
artifacts = {"model_files.json": inventory_bytes, "requirements.resolved.txt": lock_bytes,
             "model_identity.json": (json.dumps(identity, sort_keys=True) + "\n").encode(),
             "environment.json": (json.dumps(environment, sort_keys=True) + "\n").encode()}
for name, value in artifacts.items():
    target = output / name
    if target.exists():
        if target.read_bytes() != value:
            raise SystemExit(f"Preserve prior state; use a fresh output directory: {target}")
    else:
        with target.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
print(f"Verified downloaded bytes and observed identity saved to {output}; server GPU profile remains untested", flush=True)
PY
exec "${command[@]}"
