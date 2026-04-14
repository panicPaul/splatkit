#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patch_path="$repo_root/patches/diff-gaussian-rasterization/0001-modernize-packaging-and-add-splatkit-wrapper.patch"
target_repo="$repo_root/third_party/diff-gaussian-rasterization"

# Ensure nested third-party dependencies like glm are present even when this
# script is run independently of the top-level bootstrap flow.
git -C "$target_repo" submodule update --init --recursive

if git -C "$target_repo" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
    echo "diff-gaussian-rasterization patch already applied"
    exit 0
fi

git -C "$target_repo" apply --check "$patch_path"
git -C "$target_repo" apply "$patch_path"
echo "applied diff-gaussian-rasterization patch"
