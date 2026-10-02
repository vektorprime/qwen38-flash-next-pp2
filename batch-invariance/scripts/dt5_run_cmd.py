"""docker run for an eval / candidate-prod container, built from prod's saved `docker inspect`.
usage: python3 dt5_run_cmd.py <inspect.json> <name> <host_port> [options] [--dry-run]
  --inv                      all batch-invariance fixes: PLEFP8_INV_MOE/GEMM/QSA=1 + PLEFP8_ALIGN_CHUNKS=64
  --marlin-cfg C             PLEFP8_MARLIN_CFG (default: hook default)
  --hook-dir DIR             host dir mounted at /plefp8_hook (default batchinv/hook3); omit with --image-hook
  --image-hook               hook is baked into the image (no mount, no PYTHONPATH added)
  --image IMG                image (default: prod's image ID)
  --cache-volume V           named volume replacing prod's vllm-cache at /root/.cache/vllm (fresh torch.compile cache)
  --ckpt NAME                checkpoint path under /root/.cache/huggingface (replaces the served model path)
  --env K=V                  extra env (repeatable)
  --arg A                    extra serve arg (repeatable, e.g. --arg=--max-logprobs --arg=200)
  --set-arg FLAG VALUE       replace the value of an existing serve flag (e.g. --set-arg --max-model-len 441600)
  --restart P                restart policy (default no)
Everything else identical to prod (entrypoint, serve args, non-image env, GPUs, runtime, ipc, volumes). Secrets are passed
through, never printed."""
import json, subprocess, sys
B = "/home/user/qwen3nextflash/batchinv"
a = sys.argv
inspect_path, name, host_port = a[1], a[2], a[3]
opt = lambda f, d=None: a[a.index(f) + 1] if f in a else d
multi = lambda f: [x.split("=", 1)[1] for x in a if x.startswith(f + "=")] + [a[i + 1] for i, x in enumerate(a) if x == f]
dry = "--dry-run" in a
c = json.load(open(inspect_path))[0]
h, cfg = c["HostConfig"], c["Config"]
img = json.loads(subprocess.check_output(["docker", "image", "inspect", cfg["Image"]]))[0]
assert img["Id"] == c["Image"], "image tag moved"
args = list(cfg["Cmd"]); assert cfg["Entrypoint"] == ["vllm"] and args[0] == "serve"
env = [e for e in cfg["Env"] if e not in set(img["Config"]["Env"])]
assert not any(e.startswith(("PYTHONPATH=", "PLEFP8_", "VLLM_CACHE_ROOT=")) for e in env), env
if opt("--ckpt"):
    assert args[1].startswith("/root/.cache/huggingface/"); args[1] = "/root/.cache/huggingface/" + opt("--ckpt")
for i, x in enumerate(a):
    if x == "--set-arg":
        f, v = a[i + 1], a[i + 2]; j = args.index(f); args[j + 1] = v
args += multi("--arg")
cmd = ["docker", "run", "-d", "--name", name, "--restart", opt("--restart", "no")]
if h.get("Runtime"): cmd += ["--runtime", h["Runtime"]]
for d in h.get("DeviceRequests") or []: cmd += ["--gpus", '"device=%s"' % ",".join(d["DeviceIDs"])]
if h.get("IpcMode"): cmd += ["--ipc", h["IpcMode"]]
for cport in (h.get("PortBindings") or {}): cmd += ["-p", f"{host_port}:{cport.split('/')[0]}"]
for m in c["Mounts"]:
    assert m["Type"] == "volume", m
    vol = m["Name"]
    if m["Destination"] == "/root/.cache/vllm" and opt("--cache-volume"):
        vol = opt("--cache-volume")
    cmd += ["-v", f"{vol}:{m['Destination']}" + ("" if m.get("RW", True) else ":ro")]
if "--image-hook" not in a:
    cmd += ["-v", f"{opt('--hook-dir', B + '/hook3')}:/plefp8_hook:ro", "--env", "PYTHONPATH=/plefp8_hook"]
for e in env: cmd += ["--env", e]
if "--inv" in a:
    for v in ("PLEFP8_INV_MOE=1", "PLEFP8_INV_GEMM=1", "PLEFP8_INV_QSA=1", "PLEFP8_ALIGN_CHUNKS=64"): cmd += ["--env", v]
if opt("--marlin-cfg"): cmd += ["--env", "PLEFP8_MARLIN_CFG=" + opt("--marlin-cfg")]
for e in multi("--env"): cmd += ["--env", e]
cmd += [opt("--image", cfg["Image"])] + args
redact = lambda s: s.split("=", 1)[0] + "=<redacted>" if ("TOKEN" in s or "KEY" in s or "SECRET" in s) and "=" in s else s
print(json.dumps([redact(x) for x in cmd]))
if not dry:
    print(subprocess.check_output(cmd).decode().strip())
