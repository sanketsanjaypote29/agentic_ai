#!/usr/bin/env bash
set -Eeuo pipefail

# Reset cached Git credentials for the current repository's remote.
# This does not delete commits, branches, the index, or the .git directory.

git config user.name "sanketsanjaypote29"
git config user.email "sanketsanjaypote@gmail.com"

remote_name="${1:-origin}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'Error: run this script from inside a Git working tree.\n' >&2
  exit 1
fi

remote_url="$(git remote get-url "$remote_name" 2>/dev/null || true)"
if [[ -z "$remote_url" ]]; then
  printf 'Error: remote "%s" was not found.\n' "$remote_name" >&2
  exit 1
fi

protocol="https"
host=""
path=""
if [[ "$remote_url" =~ ^https?://([^/]+)(/.*)?$ ]]; then
  host="${BASH_REMATCH[1]%%:*}"
  path="${BASH_REMATCH[2]:-}"
elif [[ "$remote_url" =~ ^git@([^:]+):(.+)$ ]]; then
  protocol="ssh"
  host="${BASH_REMATCH[1]}"
  path="/${BASH_REMATCH[2]}"
else
  printf 'Warning: unsupported remote URL format; clearing only in-memory caches.\n' >&2
fi

# Stop Git's temporary credential-cache daemon, if one is running.
git credential-cache exit >/dev/null 2>&1 || true

# Reject the remote credential from configured helpers, including Git Credential Manager.
if [[ -n "$host" ]]; then
  printf 'protocol=%s\nhost=%s\npath=%s\n\n' "$protocol" "$host" "$path" |
    git credential reject >/dev/null 2>&1 || true
fi

# Remove repository-local credential helper overrides so the next push prompts cleanly.
git config --local --unset-all credential.helper >/dev/null 2>&1 || true

printf 'Git credential cache reset for remote "%s" (%s).\n' "$remote_name" "$remote_url"
printf 'Git identity set to: %s <%s>\n' "$(git config user.name)" "$(git config user.email)"
printf 'The next authenticated operation will prompt for credentials again.\n'