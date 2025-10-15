#!/usr/bin/env bash
set -euo pipefail
mkdir -p certs && cd certs
openssl req -x509 -newkey rsa:4096 -keyout ca.key -out ca.crt -days 365 -nodes -subj "/CN=FL-Dev-CA"
# Coordinator
openssl req -newkey rsa:2048 -keyout coord.key -out coord.csr -nodes -subj "/CN=coordinator"
openssl x509 -req -in coord.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out coord.crt -days 365
# 10 clients
for i in $(seq 1 10); do
  openssl req -newkey rsa:2048 -keyout client$i.key -out client$i.csr -nodes -subj "/CN=client$i"
  openssl x509 -req -in client$i.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client$i.crt -days 365
done
echo "Certs generated in $(pwd)"
