#!/usr/bin/env bash
# Build and start the lab and console, then publish them over HTTPS. Run on
# the server after setup.sh and push-env.sh; run again to deploy an update.
#
#   /opt/epicenter/deploy/start.sh [SITE_ADDRESS]
#
# Without SITE_ADDRESS the site is served at <ip-with-dashes>.sslip.io, a
# hostname that resolves to this server, until a domain points here.
set -euo pipefail
cd /opt/epicenter
git pull --ff-only
[ -f .env ] || { echo "no .env yet: run deploy/push-env.sh from your machine first"; exit 1; }

site="${1:-$(curl -fsS -4 https://ifconfig.me | tr . -).sslip.io}"
docker compose -f lab/docker-compose.yml up -d --build

mkdir -p /etc/systemd/system/caddy.service.d
printf '[Service]\nEnvironment=SITE_ADDRESS=%s\n' "$site" > /etc/systemd/system/caddy.service.d/site.conf
cp deploy/Caddyfile /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl restart caddy

for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break; sleep 2; done
echo "Live at https://$site"
