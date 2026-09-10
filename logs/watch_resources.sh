#!/usr/bin/env bash
# Resource watchdog for a long training run: logs RAM/disk/GPU, and reclaims disk before a
# checkpoint save can fail on it.
#
# The failure it exists to prevent: HF Trainer writes model.safetensors AND optimizer.pt per
# checkpoint. At 1.64B params that is ~6.6 + ~13.1 = ~20 GB each, and save_total_limit is 16, so
# a full run wants ~200 GB more than it needs. Running out mid-save leaves a corrupt checkpoint
# and kills the run hours in.
#
# optimizer.pt is safe to drop for OLDER checkpoints: it exists only to resume, we do not resume
# mid-run, and the checkpoint ladder that picks what to ship reads model.safetensors alone. The two
# newest are kept so an actual resume remains possible.
set -uo pipefail
cd /home/shadeform/chain-flow
CKPT_GLOB="out/flow/ckpts/tree-vae-joint-tr27bv2instruct-*/checkpoint-*"
LOG=logs/tr_v2_tr27b_instruct/resources.log
INTERVAL="${CF_WATCH_SEC:-300}"
FLOOR_GB="${CF_DISK_FLOOR_GB:-90}"        # act below this; ~4 checkpoints of headroom
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }

say "watchdog start (every ${INTERVAL}s, disk floor ${FLOOR_GB}G)"
while :; do
  free=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  ram_avail=$(free -g | awk 'NR==2{print $7}')
  ram_used=$(free -g | awk 'NR==2{print $3}')
  gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk '$1>=4{printf "%s:%sM ",$1,$2}')
  nck=$(ls -d $CKPT_GLOB 2>/dev/null | wc -l)
  say "disk ${free}G free | ram ${ram_used}G used ${ram_avail}G avail | ckpts ${nck} | gpu ${gpu}"

  if [ "$free" -lt "$FLOOR_GB" ]; then
    say "  disk below ${FLOOR_GB}G -- dropping optimizer.pt from all but the 2 newest checkpoints"
    # shellcheck disable=SC2012
    for c in $(ls -dt $CKPT_GLOB 2>/dev/null | tail -n +3); do
      [ -f "$c/optimizer.pt" ] || continue
      sz=$(du -m "$c/optimizer.pt" | cut -f1)
      rm -f "$c/optimizer.pt"
      say "    freed ${sz}MB from $(basename "$c")"
    done
    say "  disk now $(df --output=avail -BG / | tail -1 | tr -dc '0-9')G free"
  fi

  # training gone? record it and stop -- nothing left to protect
  if ! pgrep -f "train_tree_flow.py" >/dev/null 2>&1; then
    say "no training process; watchdog exiting"
    exit 0
  fi
  sleep "$INTERVAL"
done
