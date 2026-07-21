#!/usr/bin/env python3
"""Wrapper that runs the eval with proper unbuffered output and TTY detachment."""
import subprocess
import sys
import os

cmd = [
    sys.executable, '-u',
    '/home/artem/dev/amd-hackathon/gepa_plans/run_factual_58.py'
]

# Run with stdout/stderr piped, detached from TTY
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=0,
    env={**os.environ, 'PYTHONUNBUFFERED': '1'},
)

# Stream output
for line in proc.stdout:
    sys.stdout.buffer.write(line)
    sys.stdout.buffer.flush()

proc.wait()
sys.exit(proc.returncode)
