"""Block until a NEW line matching REGEX is appended to LOG (or the writer PID exits), print it, exit.
usage: wait_log.py LOG REGEX [PID]   -- run as a background job to get woken on the event."""
import os, re, sys, time
log, rx = sys.argv[1], re.compile(sys.argv[2])
pid = int(sys.argv[3]) if len(sys.argv) > 3 else None
f = open(log); f.seek(0, 2)
while True:
    line = f.readline()
    if line:
        if rx.search(line):
            print(line.rstrip()); sys.exit(0)
        continue
    if pid is not None:
        try:
            os.kill(pid, 0)
        except OSError:
            print(f"writer pid {pid} exited"); sys.exit(0)
    time.sleep(1)
