#!/bin/bash
set -e
cd "$(dirname "$0")/../dev"
uv run generate-openapi
