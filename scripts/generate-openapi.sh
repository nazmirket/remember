#!/bin/bash
set -e
cd "$(dirname "$0")/../remember-dev"
uv run generate-openapi
