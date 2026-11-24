#!/usr/bin/env bash
# One bounded continuation of this validation run, not a recurring watcher.
ys_until=$((SECONDS + 1200))
while [ ! -f /tmp/ys-opt-1124/m2-stop-matrix.status ]; do
  if [ "$SECONDS" -ge "$ys_until" ] || ! kill -0 360089 2>/dev/null; then
    printf 'Mission 2 did not finish normally; Mission 4 was not started.\n'
    exit 2
  fi
  sleep 5
done
ys_status=$(cat /tmp/ys-opt-1124/m2-stop-matrix.status)
if [ "$ys_status" -ne 0 ]; then
  printf 'Mission 2 requires review (exit %s); Mission 4 was not started.\n' "$ys_status"
  exit "$ys_status"
fi
exec bash /tmp/ys-opt-1124/run_m4_final.sh
