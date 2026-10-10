#!/bin/bash
# Generate reasoning+solution datasets from KodCode-V1 using DeepSeek-R1.
#
# Steps:
#   1. Filter 10% of KodCode-V1 for security relevance (splits into relevant/irrelevant)
#   2. Create subsets of the relevant dataset: 11%, 22%, 33%, 67% fractions
#      and injection-only (no memory), memory-only (no injection)
#   3. Generate traces on the security-relevant dataset with secure, secure+, and cwe prompts
#   4. Generate traces on the security-irrelevant dataset with secure+ prompt (random 1324 samples)

set -euo pipefail

# --- Configuration ---
BASE_URL="" # Base URL
API_KEY="" # API Key
CHECKER_MODEL="openai/gpt-4o"
GEN_MODEL="vertex_ai/deepseek-ai/deepseek-r1-0528-maas"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_BASE_URL="$BASE_URL"

DATA_DIR="data/dataset_security"
RELEVANT="${DATA_DIR}/KodCode-KodCode-V1_security_relevant.json"
IRRELEVANT="${DATA_DIR}/KodCode-KodCode-V1_security_irrelevant.json"
SUBSET_SCRIPT="scripts/subset_dataset.py"

# --- Step 1: Filter 10% of KodCode-V1 ---
echo "=== Step 1: Filtering 10% of KodCode-V1 ==="
python -m src --mode filter-dataset \
    --checker_model "$CHECKER_MODEL" \
    --checker_provider openai \
    --sample_fraction 0.1

echo ""
echo "Relevant samples:   $(python -c "import json; print(len(json.load(open('$RELEVANT'))))")"
echo "Irrelevant samples: $(python -c "import json; print(len(json.load(open('$IRRELEVANT'))))")"

# --- Step 2: Create subsets ---
echo ""
echo "=== Step 2: Creating subsets of security-relevant dataset ==="

# Fraction-based subsets
for FRAC in 0.11 0.22 0.33 0.67; do
    OUTFILE="${DATA_DIR}/relevant_frac_${FRAC}.json"
    python "$SUBSET_SCRIPT" fraction "$RELEVANT" "$OUTFILE" --frac "$FRAC"
done

# CWE-category subsets: injection-only (no memory) and memory-only (no injection)
python "$SUBSET_SCRIPT" category "$RELEVANT" "${DATA_DIR}/relevant_injection_only.json" \
    --include injection --exclude memory

python "$SUBSET_SCRIPT" category "$RELEVANT" "${DATA_DIR}/relevant_memory_only.json" \
    --include memory --exclude injection

# --- Step 3: Generate on security-relevant dataset (secure, secure+, cwe) ---
echo ""
echo "=== Step 3: Generating traces on security-relevant dataset ==="

for PROMPT in secure "secure+" cwe; do
    echo "--- Prompt: $PROMPT ---"
    python -m src --mode generate-dataset \
        --model "$GEN_MODEL" \
        --provider openai \
        --gen_input "$RELEVANT" \
        --gen_prompt "$PROMPT"
done

# --- Step 4: Generate on security-irrelevant dataset (secure+, 1324 random samples) ---
echo ""
echo "=== Step 4: Generating secure+ traces on security-irrelevant dataset (1324 random samples) ==="
python -m src --mode generate-dataset \
    --model "$GEN_MODEL" \
    --provider openai \
    --gen_input "$IRRELEVANT" \
    --gen_prompt "secure+" \
    --gen_max_samples 1324 \
    --gen_output "data/dataset_generated/irrelevant_secure+"
