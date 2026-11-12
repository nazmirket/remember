#!/bin/bash
set -e

# Regenerate schemas/bank-template-schema.json from the
# BankTemplateManifest Pydantic model. This file is the source of truth
# for local validation.

cd "$(dirname "$0")/.."

echo "Generating bank template JSON Schema..."
cd remember-dev
uv run generate-bank-template-schema

echo ""
echo "Bank template schema generated successfully!"
