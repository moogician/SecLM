#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_eval.sh — Generate, test, and summarize across CWEval, LiveCodeBench,
#               and BaxBench for a given model.
#
# Usage:
#   ./run_eval.sh --model <model_name> --provider <provider> [options]
#
# Examples:
#   # Run all benchmarks with vLLM (server must be running)
#   ./run_eval.sh --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --provider vllm
#
#   # Run only CWEval with OpenAI
#   ./run_eval.sh --model gpt-4o --provider openai --benchmarks cweval
#
#   # Run with multiple prompts and custom results dir
#   ./run_eval.sh --model Qwen/QwQ-32B --provider vllm \
#       --prompts "direct secure secureplus" --results_dir my-results
#
#   # Skip generation (only test + summary on existing results)
#   ./run_eval.sh --model Qwen/QwQ-32B --provider vllm --skip_generate
#
#   # Skip testing (only summary on existing test results)
#   ./run_eval.sh --model Qwen/QwQ-32B --provider vllm --skip_generate --skip_test
# ============================================================================

# --- Defaults ---
MODEL=""
PROVIDER="vllm"
BENCHMARKS="cweval livecodebench baxbench"
PROMPTS="direct"
TEMPERATURE=0.6
N_SAMPLES=5
MAX_TOKENS=24000
MAX_CONCURRENT=32
RESULTS_DIR="eval-traces"
VLLM_PORT=8000
EVAL_SUBSET="valid"
TIMEOUT=300
SUFFIX=""
SKIP_GENERATE=false
SKIP_TEST=false
FORCE=false

# --- BaxBench-specific ---
SPEC_TYPE="openapi"
SCENARIOS=""
ENVS=""

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# \?//'
    echo ""
    echo "Options:"
    echo "  --model            Model name (required, e.g. deepseek-ai/DeepSeek-R1)"
    echo "  --provider         Inference provider: vllm|openai|together|openrouter|anthropic (default: vllm)"
    echo "  --benchmarks       Space-separated list: cweval livecodebench baxbench (default: all three)"
    echo "  --prompts          Space-separated prompt types: direct secure secureplus (default: direct)"
    echo "  --temperature      Sampling temperature (default: 0.6)"
    echo "  --n_samples        Samples per problem (default: 5)"
    echo "  --max_tokens       Max completion tokens (default: 24000)"
    echo "  --max_concurrent   Max parallel workers (default: 32)"
    echo "  --results_dir      Results directory (default: eval-traces)"
    echo "  --vllm_port        vLLM server port (default: 8000)"
    echo "  --eval_subset      Evaluation subset for CWEval (default: valid)"
    echo "  --timeout          Test timeout in seconds for BaxBench (default: 300)"
    echo "  --suffix           Suffix for BaxBench result dir"
    echo "  --spec_type        BaxBench spec type: openapi|text (default: openapi)"
    echo "  --scenarios        BaxBench scenarios (space-separated)"
    echo "  --envs             BaxBench environments (space-separated)"
    echo "  --skip_generate    Skip the generation phase"
    echo "  --skip_test        Skip the test phase"
    echo "  --force            Force re-generation and re-testing"
    exit 1
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2 ;;
        --provider)       PROVIDER="$2"; shift 2 ;;
        --benchmarks)     BENCHMARKS="$2"; shift 2 ;;
        --prompts)        PROMPTS="$2"; shift 2 ;;
        --temperature)    TEMPERATURE="$2"; shift 2 ;;
        --n_samples)      N_SAMPLES="$2"; shift 2 ;;
        --max_tokens)     MAX_TOKENS="$2"; shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --results_dir)    RESULTS_DIR="$2"; shift 2 ;;
        --vllm_port)      VLLM_PORT="$2"; shift 2 ;;
        --eval_subset)    EVAL_SUBSET="$2"; shift 2 ;;
        --timeout)        TIMEOUT="$2"; shift 2 ;;
        --suffix)         SUFFIX="$2"; shift 2 ;;
        --spec_type)      SPEC_TYPE="$2"; shift 2 ;;
        --scenarios)      SCENARIOS="$2"; shift 2 ;;
        --envs)           ENVS="$2"; shift 2 ;;
        --skip_generate)  SKIP_GENERATE=true; shift ;;
        --skip_test)      SKIP_TEST=true; shift ;;
        --force)          FORCE=true; shift ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    usage
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Helpers ---
log() { echo -e "\n========== $1 ==========\n"; }
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_phase() {
    local benchmark="$1"
    local mode="$2"
    local prompt="$3"

    local cmd=(python -m src.main
        --mode "$mode"
        --benchmark "$benchmark"
        --model "$MODEL"
        --provider "$PROVIDER"
        --prompt "$prompt"
        --temperature "$TEMPERATURE"
        --n_samples "$N_SAMPLES"
        --max_tokens "$MAX_TOKENS"
        --max_concurrent_runs "$MAX_CONCURRENT"
        --results_dir "$RESULTS_DIR"
        --vllm_port "$VLLM_PORT"
        --eval_subset "$EVAL_SUBSET"
    )

    # BaxBench-specific args
    if [[ "$benchmark" == "baxbench" ]]; then
        cmd+=(--spec_type "$SPEC_TYPE" --timeout "$TIMEOUT")
        [[ -n "$SUFFIX" ]] && cmd+=(--suffix "$SUFFIX")
        [[ -n "$SCENARIOS" ]] && cmd+=(--scenarios $SCENARIOS)
        [[ -n "$ENVS" ]] && cmd+=(--envs $ENVS)
        [[ "$FORCE" == true ]] && cmd+=(--force)
    fi

    echo "[$(timestamp)] Running: ${cmd[*]}"
    "${cmd[@]}"
}

# --- Main pipeline ---
for benchmark in $BENCHMARKS; do
    log "$benchmark"

    for prompt in $PROMPTS; do
        log "$benchmark / prompt=$prompt"

        # 1. Generate
        if [[ "$SKIP_GENERATE" == false ]]; then
            log "GENERATE: $benchmark (prompt=$prompt)"
            run_phase "$benchmark" generate "$prompt"
        fi

        # 2. Test
        if [[ "$SKIP_TEST" == false ]]; then
            log "TEST: $benchmark (prompt=$prompt)"
            run_phase "$benchmark" test "$prompt"
        fi
    done

    # 3. Summary (runs once per benchmark, reads all prompts)
    log "SUMMARY: $benchmark"
    run_phase "$benchmark" summary "${PROMPTS%% *}"
done

log "ALL DONE"
echo "[$(timestamp)] Results in: $RESULTS_DIR/"
