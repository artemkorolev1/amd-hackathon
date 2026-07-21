#!/usr/bin/env bash
# Wrapper script to run math eval
set -euo pipefail
cd /home/artem/dev/amd-hackathon/gepa_plans
python3 run_math_v4.py
echo "EXIT_CODE=$?"
