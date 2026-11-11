#!/bin/bash
set -e

# Regenerate schemas/bank-template-schema.json from the
# BankTemplateManifest Pydantic model. This file is the source of truth
# for local validation and
# is served verbatim at /bank-template-schema.json on the docs site.

cd "$(dirname "$0")/.."

echo "Generating bank template JSON Schema..."
cd hindsight-dev
uv run generate-bank-template-schema

echo ""
echo "Bank template schema generated successfully!"
