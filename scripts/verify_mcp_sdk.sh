#!/bin/sh
# Independent MCP interoperability check.
#
# Installs the official MCP SDK into a throwaway venv and uses it as a client
# against this server. The SDK is NOT a dependency of Filmautomator — it only
# exists here to prove a third-party client can connect.
set -e
cd "$(dirname "$0")/.."

if [ ! -x venv/bin/python ]; then
  /opt/homebrew/bin/python3 -m venv venv
fi

venv/bin/python -m pip install --quiet --upgrade pip
venv/bin/python -m pip install --quiet mcp
echo "official mcp SDK installed"
venv/bin/python tests/verify_mcp_sdk.py
