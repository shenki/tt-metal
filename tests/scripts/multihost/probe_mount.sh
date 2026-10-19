#!/usr/bin/env bash
# Probe whether a filesystem path is mounted and readable on the multihost control pod
# and on every worker host that `mpirun --pernode` reaches.
#
# Usage: probe_mount.sh [path ...]          (default: /mnt/weka)
# Env:
#   PROBE_READ_FILE   file under the path to time a read of (default: first regular file found)
#   PROBE_READ_MB     MB to read from it (default 256)
#   PROBE_WRITE_DIR   if set, create + remove a small file there to check write access
#
# Exits non-zero if any path is missing or unreadable on any host.

set -uo pipefail

paths=("$@")
[ ${#paths[@]} -eq 0 ] && paths=(/mnt/weka)

probe_host() {
    local rc=0 p
    echo "=== host $(hostname) ==="
    for p in "$@"; do
        echo "--- $p"
        if ! findmnt -T "$p" -o TARGET,SOURCE,FSTYPE,OPTIONS; then
            echo "FAIL: no mount covers $p"
            rc=1
            continue
        fi
        if [ "$(findmnt -n -T "$p" -o TARGET)" = "/" ]; then
            echo "WARN: $p resolves to the root filesystem, not a dedicated mount"
        fi
        if ! timeout 60 ls -la "$p" | head -20; then
            echo "FAIL: cannot list $p"
            rc=1
            continue
        fi
        timeout 60 df -h "$p" || true

        local f="${PROBE_READ_FILE:-}"
        [ -z "$f" ] && f=$(timeout 120 find "$p" -maxdepth 6 -type f -size +1M -print -quit 2>/dev/null)
        if [ -n "$f" ]; then
            echo "read test: $f"
            if ! timeout 300 dd if="$f" of=/dev/null bs=1M count="${PROBE_READ_MB:-256}" status=progress 2>&1 | tail -1; then
                echo "FAIL: read of $f failed"
                rc=1
            fi
        else
            echo "WARN: no regular file >1 MB found under $p in 6 levels; read throughput not measured"
        fi
    done

    if [ -n "${PROBE_WRITE_DIR:-}" ]; then
        local t="$PROBE_WRITE_DIR/.probe_$(hostname)_$$"
        if echo probe > "$t" && rm -f "$t"; then
            echo "write test: OK in $PROBE_WRITE_DIR"
        else
            echo "FAIL: cannot write to $PROBE_WRITE_DIR"
            rc=1
        fi
    fi
    echo "=== host $(hostname): $([ $rc -eq 0 ] && echo PASS || echo FAIL) ==="
    return $rc
}

rc=0
echo "##### control pod (job steps run here)"
probe_host "${paths[@]}" || rc=1

if command -v mpirun >/dev/null 2>&1; then
    echo "##### worker hosts (mpirun --pernode)"
    # The function and the PROBE_* settings travel inside the command string, so they do not
    # depend on mpirun forwarding the environment.
    remote="$(declare -f probe_host)
PROBE_READ_FILE=$(printf '%q' "${PROBE_READ_FILE:-}") PROBE_READ_MB=$(printf '%q' "${PROBE_READ_MB:-256}") \
PROBE_WRITE_DIR=$(printf '%q' "${PROBE_WRITE_DIR:-}") probe_host $(printf '%q ' "${paths[@]}")"
    mpirun --bind-to none --pernode --tag-output bash -c "$remote" || rc=1
else
    echo "WARN: mpirun not on PATH; probed the local host only"
fi

echo "##### overall: $([ $rc -eq 0 ] && echo PASS || echo FAIL)"
exit $rc
