#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_relax45_20260919_v1
export PYTHONUNBUFFERED=1
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
"$P" train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/relax45_smoke4096_20260919 --num-envs 4096 --updates 4 --mixed-pit --actuator-profile rs0402_transient_v4 --slope-profile slope45_v1 --resume /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_20260919/model_30345.pt
"$P" train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/relax45_inherit4096_20260919 --num-envs 4096 --updates 2 --mixed-pit --resume /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/relax45_smoke4096_20260919/model_30349.pt
"$P" /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/probe_wave_relax45.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --code /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_relax45_20260919_v1 --checkpoint /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_20260919/model_30345.pt --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/relax45_wave_v3_20260919 --profile rs0402_slew12_v3
"$P" /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/probe_wave_relax45.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --code /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_relax45_20260919_v1 --checkpoint /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_20260919/model_30345.pt --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/relax45_wave_v4_20260919 --profile rs0402_transient_v4

