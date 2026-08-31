#!/bin/bash
set -euo pipefail

SOURCE_ROOT=/home/msai/lius0131/AGI/repostguard-lite/outputs/community_forensics_v3
TARGET_ROOT=/home/msai/xjiang026/projects/repostguard-lite/outputs/community_forensics_v3

copy_teacher() {
    local teacher=$1
    local expected_sha256=$2
    local source="$SOURCE_ROOT/$teacher/best.pt"
    local destination="$TARGET_ROOT/$teacher/best.pt"
    local temporary="$TARGET_ROOT/$teacher/.best.pt.partial"

    if [[ ! -r "$source" ]]; then
        echo "Teacher checkpoint is not readable: $source" >&2
        return 13
    fi
    local source_sha256
    source_sha256=$(sha256sum "$source" | awk '{print $1}')
    if [[ "$source_sha256" != "$expected_sha256" ]]; then
        echo "Source SHA256 mismatch for $teacher: $source_sha256" >&2
        return 14
    fi
    mkdir -p "$TARGET_ROOT/$teacher"
    if [[ -e "$destination" ]]; then
        local destination_sha256
        destination_sha256=$(sha256sum "$destination" | awk '{print $1}')
        if [[ "$destination_sha256" == "$expected_sha256" ]]; then
            echo "$teacher already imported and verified"
            return 0
        fi
        echo "Refusing to overwrite mismatched destination: $destination" >&2
        return 15
    fi
    cp --reflink=auto -- "$source" "$temporary"
    local copied_sha256
    copied_sha256=$(sha256sum "$temporary" | awk '{print $1}')
    if [[ "$copied_sha256" != "$expected_sha256" ]]; then
        echo "Copied SHA256 mismatch for $teacher: $copied_sha256" >&2
        return 16
    fi
    chmod 0640 "$temporary"
    mv -- "$temporary" "$destination"
    echo "$teacher imported: $destination $copied_sha256"
}

copy_teacher m2 468d3a58603fdf8dfe1b234a24fd8e52a99c6e4881e921bef6bb0cea64bbac34
copy_teacher m3 c83f70641a9c8d7f6808e794cfc8c28c0e478feeca7506e489c772a512115b2f
