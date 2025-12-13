#!/usr/bin/env bash
set -euo pipefail

# Путь к каталогу с Terraform-конфигурацией
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="${SCRIPT_DIR}/../cloud_terraform"

cd "$INFRA_DIR"

echo "RUN terraform init"
terraform init

echo "RUN terraform plan"
terraform plan

echo "RUN terraform apply"
terraform apply -auto-approve

echo "Получение IP-адреса ВМ из output..."
PUBLIC_IP=$(terraform output -raw externalipaddressvm1 2>/dev/null || terraform output -raw external_ip_address_vm1 2>/dev/null || true)

if [[ -z "$PUBLIC_IP" ]]; then
  echo "Не удалось получить публичный IP из output. Проверь названия output в main.tf."
  exit 1
fi

echo "RUN Публичный IP ВМ: $PUBLIC_IP"

# Обновление inventory для Ansible
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVENTORY_FILE="$DEPLOY_DIR/inventory.ini"

cat > "$INVENTORY_FILE" <<EOF
[app]
$PUBLIC_IP ansible_user=winter ansible_ssh_private_key_file=~/.ssh/id_ed25519
EOF

echo "RUN inventory.ini обновлён:"
cat "$INVENTORY_FILE"
