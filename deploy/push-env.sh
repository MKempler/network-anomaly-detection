#!/usr/bin/env bash
# Copy only the keys the live demo needs from your local .env to the server.
#
#   deploy/push-env.sh SERVER [SITE_ADDRESS]
#
# The Meta Model API key stays on your machine: the live lab investigates with
# Muse Spark Open on Baseten, and optionally fine-tuned Glimmer on Modal. The
# file is written straight to the server (owner-only) and never to local disk.
set -euo pipefail
server="$1"; site="${2:-}"
here="$(cd "$(dirname "$0")/.." && pwd)"
keys='^(BASETEN_API_KEY|LIVE_GLIMMER|GLIMMER_BASE_URL|GLIMMER_API_KEY|MODAL_PROXY_KEY|MODAL_PROXY_SECRET|SLACK_WEBHOOK_URL|STUDY_CODES)='
{
  grep -E "$keys" "$here/.env" || true
  [ -n "$site" ] && echo "CONSOLE_URL=https://$site"
} | ssh "root@$server" 'umask 077; cat > /opt/epicenter/.env && echo "wrote $(wc -l < /opt/epicenter/.env) settings to /opt/epicenter/.env"'
