#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server for the Epicenter demo.
#
#   ssh root@SERVER 'bash -s' < deploy/setup.sh
#
# Installs Docker and Caddy, opens only SSH, HTTP and HTTPS, turns on
# security updates and SSH keys only, adds 2 GB of swap and clones the repo
# to /opt/epicenter. Safe to re-run.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y upgrade
apt-get install -y ca-certificates curl git ufw unattended-upgrades docker.io docker-compose-v2 caddy
systemctl enable --now docker

# Firewall: only SSH and the website. The lab publishes nothing but
# 127.0.0.1:8000, so Docker's own port rules expose nothing either.
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Security updates install themselves; logins are by SSH key only.
dpkg-reconfigure -f noninteractive unattended-upgrades
sed -i -E 's/^#?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl try-reload-or-restart ssh   # 24.04 starts sshd on demand, so it may not be running yet

# A little swap, so building images can't run the server out of memory.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

if [ ! -d /opt/epicenter ]; then
  git clone https://github.com/MKempler/network-anomaly-detection.git /opt/epicenter
fi
echo "Setup done. Next, from your machine: deploy/push-env.sh SERVER, then deploy/start.sh on the server."
