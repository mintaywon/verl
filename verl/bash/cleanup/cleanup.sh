#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /abs/path/to/llama-3.2-1b_ppo_n0.01 [--yes] [--leave_last]"
  exit 1
fi

BASE_DIR="$1"
DO_DELETE="no"
LEAVE_LAST="no"

# parse optional flags starting from $2
if [[ $# -ge 2 ]]; then
  for arg in "${@:2}"; do
    case "$arg" in
      --yes)
        DO_DELETE="yes"
        ;;
      --leave_last)
        LEAVE_LAST="yes"
        ;;
      *)
        echo "Unknown option: $arg"
        exit 1
        ;;
    esac
  done
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: '$BASE_DIR' is not a directory"
  exit 1
fi

LAST_DIR=""
if [[ "$LEAVE_LAST" == "yes" ]]; then
  # find the latest global_step_* directory by numeric order
  mapfile -t STEP_DIRS < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort)
  if [[ ${#STEP_DIRS[@]} -gt 0 ]]; then
    latest_n=-1
    for d in "${STEP_DIRS[@]}"; do
      bn=$(basename "$d")
      n=${bn#global_step_}
      if [[ "$n" =~ ^[0-9]+$ ]] && (( n > latest_n )); then
        latest_n=$n
        LAST_DIR="$d"
      fi
    done
  fi
  if [[ -n "$LAST_DIR" ]]; then
    echo "Will preserve optim/extra_state under latest checkpoint: $LAST_DIR"
  else
    echo "Warning: --leave_last specified but no global_step_* directories found; nothing to preserve."
  fi
fi

echo "Scanning for optim_* and extra_state_* under: $BASE_DIR"
MAPFILE=()
if [[ -n "$LAST_DIR" ]]; then
  while IFS= read -r -d '' f; do
    MAPFILE+=("$f")
  done < <(find "$BASE_DIR" -type f \
    \( -name 'optim_world_size_*_rank_*.pt' -o -name 'extra_state_world_size_*_rank_*.pt' \) \
    -not -path "$LAST_DIR/*" -print0)
else
  while IFS= read -r -d '' f; do
    MAPFILE+=("$f")
  done < <(find "$BASE_DIR" -type f \
    \( -name 'optim_world_size_*_rank_*.pt' -o -name 'extra_state_world_size_*_rank_*.pt' \) -print0)
fi

if [[ ${#MAPFILE[@]} -eq 0 ]]; then
  echo "No matching files found."
  exit 0
fi

echo "Found ${#MAPFILE[@]} files:"
printf '%s\n' "${MAPFILE[@]}"

if [[ "$DO_DELETE" == "yes" ]]; then
  echo "Deleting..."
  # Use find -delete for atomic delete to avoid issues with spaces
  if [[ -n "$LAST_DIR" ]]; then
    find "$BASE_DIR" -type f \
      \( -name 'optim_world_size_*_rank_*.pt' -o -name 'extra_state_world_size_*_rank_*.pt' \) \
      -not -path "$LAST_DIR/*" -print -delete
  else
    find "$BASE_DIR" -type f \
      \( -name 'optim_world_size_*_rank_*.pt' -o -name 'extra_state_world_size_*_rank_*.pt' \) -print -delete
  fi
  echo "Done."
else
  echo "Dry run complete. Re-run with --yes to delete."
fi