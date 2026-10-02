"""docker run for a batch-invariance eval container, built from prod's saved `docker inspect`.
usage: python3 dt3_run_cmd.py <inspect.json> <name> <host_port> [--inv-moe] [--inv-gemm] [--inv-qsa] [--fresh-cache HOSTDIR] [--dry-run]
Identical to prod (image by ID, entrypoint, serve args, non-image env, GPUs, runtime, ipc, named volumes) except: name, host port,
--restart no, hook3 mount + PYTHONPATH, control dir mount + PLEFP8_CTRL, the PLEFP8_INV_* flags, and with --fresh-cache an
empty VLLM_CACHE_ROOT (needed for --inv-gemm: the patched GEMM changes the traced graph). Secrets are passed through, never printed."""
import json, subprocess, sys
B = "/home/user/qwen3nextflash/batchinv"
a = sys.argv
inspect_path, name, host_port = a[1], a[2], a[3]
dry = "--dry-run" in a
fresh = a[a.index("--fresh-cache") + 1] if "--fresh-cache" in a else None
c = json.load(open(inspect_path))[0]
h, cfg = c["HostConfig"], c["Config"]
img = json.loads(subprocess.check_output(["docker", "image", "inspect", cfg["Image"]]))[0]
assert img["Id"] == c["Image"], "image tag moved"
args = list(cfg["Cmd"]); assert cfg["Entrypoint"] == ["vllm"] and args[0] == "serve"
env = [e for e in cfg["Env"] if e not in set(img["Config"]["Env"])]
assert not any(e.startswith(("PYTHONPATH=", "PLEFP8_", "VLLM_CACHE_ROOT=")) for e in env), env
cmd = ["docker", "run", "-d", "--name", name, "--restart", "no"]
if h.get("Runtime"): cmd += ["--runtime", h["Runtime"]]
for d in h.get("DeviceRequests") or []: cmd += ["--gpus", '"device=%s"' % ",".join(d["DeviceIDs"])]
if h.get("IpcMode"): cmd += ["--ipc", h["IpcMode"]]
for cport in (h.get("PortBindings") or {}): cmd += ["-p", f"{host_port}:{cport.split('/')[0]}"]
for m in c["Mounts"]:
    assert m["Type"] == "volume", m
    cmd += ["-v", f"{m['Name']}:{m['Destination']}" + ("" if m.get("RW", True) else ":ro")]
cmd += ["-v", f"{B}/hook3:/plefp8_hook:ro", "-v", f"{B}/ctrl:/plefp8_ctrl"]
for e in env: cmd += ["--env", e]
cmd += ["--env", "PYTHONPATH=/plefp8_hook", "--env", "PLEFP8_CTRL=/plefp8_ctrl"]
for flag, var in (("--inv-moe", "PLEFP8_INV_MOE"), ("--inv-gemm", "PLEFP8_INV_GEMM"), ("--inv-qsa", "PLEFP8_INV_QSA")):
    if flag in a: cmd += ["--env", f"{var}=1"]
if fresh:
    cmd += ["-v", f"{fresh}:/plefp8_vcache", "--env", "VLLM_CACHE_ROOT=/plefp8_vcache"]
cmd += [cfg["Image"]] + args
redact = lambda s: s.split("=", 1)[0] + "=<redacted>" if ("TOKEN" in s or "KEY" in s or "SECRET" in s) and "=" in s else s
print(json.dumps([redact(x) for x in cmd]))
if not dry:
    print(subprocess.check_output(cmd).decode().strip())
