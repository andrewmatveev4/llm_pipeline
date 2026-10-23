#!/usr/bin/env bash
# Прогон демо-сценариев. Результаты в demo/outputs/*.json
set -u
cd "$(dirname "$0")/.."
mkdir -p demo/outputs

for name in complaint support_en sales injection; do
    echo "== $name"
    llm-pipeline --file "demo/inputs/$name.txt" --json > "demo/outputs/$name.json"
    echo "код возврата: $?"
done

echo "== empty_input"
llm-pipeline "   " --json > demo/outputs/empty_input.json
echo "код возврата: $?"

echo "== api_down"
OPENAI_BASE_URL=http://127.0.0.1:9 llm-pipeline --file demo/inputs/complaint.txt --json \
    > demo/outputs/api_down.json
echo "код возврата: $?"