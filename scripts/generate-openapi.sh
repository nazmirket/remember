#!/bin/bash
set -e
cd "$(dirname "$0")/../hindsight-dev"
uv run generate-openapi
