#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="${SCRIPT_DIR}/../cloud_terraform"

cd "$INFRA_DIR"

if [[ ! -f inventory.ini ]]; then
  echo "File inventory.ini not found!"
  exit 1
fi

echo "Runining inventory:"
cat inventory.ini
echo

echo "Ansible playbook is running"
ansible-playbook -i inventory.ini site.yml
