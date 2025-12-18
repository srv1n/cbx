#!/usr/bin/env bash
# Installs the bundled "default" cbx voice profiles into the local voice cache.
# This downloads from GitHub Releases and verifies SHA256.

set -euo pipefail

REPO="${REPO:-srv1n/cbx}"
VERSION="${VERSION:-}"

print() { printf "%s\n" "$*"; }
err() { printf "error: %s\n" "$*" >&2; }

detect_hf_home() {
  if [[ -n "${HF_HOME:-}" ]]; then
    print "${HF_HOME}"
    return 0
  fi

  # Common defaults used by Hugging Face tooling:
  # - Linux: ~/.cache/huggingface
  # - macOS: ~/Library/Caches/huggingface
  case "$(uname -s 2>/dev/null || echo unknown)" in
    Darwin) print "${HOME}/Library/Caches/huggingface" ;;
    *) print "${HOME}/.cache/huggingface" ;;
  esac
}

http_get() {
  local url=$1
  local out=$2

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$out" "$url"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$out" "$url"
    return 0
  fi

  err "neither curl nor wget found"
  return 1
}

get_latest_version() {
  local api="https://api.github.com/repos/${REPO}/releases/latest"
  local tmp
  tmp="$(mktemp 2>/dev/null || mktemp -t cbx-latest)"
  http_get "$api" "$tmp"

  local tag
  # Parse `"tag_name": "vX.Y.Z"` from the GitHub API response.
  # Use `cut -d'"'` (1-byte delimiter); macOS `cut` rejects multi-byte delimiters.
  tag="$(grep -m1 '\"tag_name\"' "$tmp" | cut -d'"' -f4 || true)"
  rm -f "$tmp" || true
  if [[ -z "${tag}" ]]; then
    err "failed to determine latest version tag from GitHub API"
    return 1
  fi
  print "$tag"
}

verify_sha256() {
  local file=$1
  local sha_file=$2

  if command -v shasum >/dev/null 2>&1; then
    (cd "$(dirname "$file")" && shasum -a 256 -c "$(basename "$sha_file")")
    return 0
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$(dirname "$file")" && sha256sum -c "$(basename "$sha_file")")
    return 0
  fi

  err "no sha256 verifier found (need shasum or sha256sum)"
  return 1
}

main() {
  local hf_home voice_dir base_url tmp
  hf_home="$(detect_hf_home)"
  voice_dir="${hf_home}/cbx/voices"
  if [[ -z "${VERSION}" ]]; then
    VERSION="$(get_latest_version)"
  fi
  base_url="https://github.com/${REPO}/releases/download/${VERSION}"
  tmp="$(mktemp -d 2>/dev/null || mktemp -d -t cbx-voice)"

  print "cbx default voice installer"
  print "  repo:      ${REPO}"
  print "  version:   ${VERSION}"
  print "  hf_home:   ${hf_home}"
  print "  install:   ${voice_dir}"
  print ""
  print "This installs two profiles:"
  print "  - default.cbxvoice      (dtype=fp16)"
  print "  - default-fp32.cbxvoice (dtype=fp32)"

  mkdir -p "${voice_dir}"

  for name in cbx-voice-default-fp16.cbxvoice cbx-voice-default-fp32.cbxvoice; do
    local url="${base_url}/${name}"
    local sha="${url}.sha256"
    local out="${tmp}/${name}"
    local sha_out="${tmp}/${name}.sha256"

    http_get "${url}" "${out}"
    http_get "${sha}" "${sha_out}"
    verify_sha256 "${out}" "${sha_out}"

    if [[ "${name}" == "cbx-voice-default-fp16.cbxvoice" ]]; then
      cp -f "${out}" "${voice_dir}/default.cbxvoice"
    else
      cp -f "${out}" "${voice_dir}/default-fp32.cbxvoice"
    fi
  done

  print ""
  print "Installed:"
  print "  ${voice_dir}/default.cbxvoice"
  print "  ${voice_dir}/default-fp32.cbxvoice"
  print ""
  print "Try:"
  print "  cbx speak --text \"Hello\" --out-wav out.wav"
  print "  cbx speak --dtype fp32 --voice default-fp32 --text \"Hello\" --out-wav out.wav"
}

main "$@"
