#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-Cora CiteSeer PubMed Reddit ogbn-arxiv ogbn-products}"
MODES="${MODES:-fast_full gcn_full fast_mini}"
RUN_PLOTS="${RUN_PLOTS:-1}"
TASK_TIMER_INTERVAL="${TASK_TIMER_INTERVAL:-30}"

# Optional Reddit tuning knobs.
# Defaults favor faster turnaround for presentation/demo comparisons.
REDDIT_INIT_BATCH="${REDDIT_INIT_BATCH:-1024}"
REDDIT_SAMPLE_SIZE="${REDDIT_SAMPLE_SIZE:-5120}"
REDDIT_INF_INIT_BATCH="${REDDIT_INF_INIT_BATCH:-1024}"
REDDIT_INF_SAMPLE_SIZE="${REDDIT_INF_SAMPLE_SIZE:-15360}"
REDDIT_REPORT="${REDDIT_REPORT:-20}"

# Optional ogbn-arxiv tuning knobs.
# Defaults are adjusted for faster wall-clock runs during demos.
ARXIV_EPOCHS="${ARXIV_EPOCHS:-200}"
ARXIV_REPORT="${ARXIV_REPORT:-50}"
ARXIV_INF_INIT_BATCH="${ARXIV_INF_INIT_BATCH:-8192}"
ARXIV_INF_SAMPLE_SIZE="${ARXIV_INF_SAMPLE_SIZE:-32768}"

# Optional ogbn-products tuning knobs.
# Defaults are adjusted for faster wall-clock runs during demos.
PRODUCTS_EPOCHS="${PRODUCTS_EPOCHS:-300}"
PRODUCTS_REPORT="${PRODUCTS_REPORT:-200}"
PRODUCTS_INF_INIT_BATCH="${PRODUCTS_INF_INIT_BATCH:-32768}"
PRODUCTS_INF_SAMPLE_SIZE="${PRODUCTS_INF_SAMPLE_SIZE:-131072}"

mkdir -p "$OUT_DIR/logs"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/.mpl-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OUT_DIR/.cache}"
export PYTHONUNBUFFERED=1
export OGB_AUTO_INPUT="${OGB_AUTO_INPUT:-1}"
export OGB_UPDATE_RESPONSE="${OGB_UPDATE_RESPONSE:-n}"
export OGB_DOWNLOAD_RESPONSE="${OGB_DOWNLOAD_RESPONSE:-y}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"
RESULT_CSV="$OUT_DIR/results.csv"
printf "dataset,mode,trial,status,acc,batch_time,total_time,log_path,command\n" > "$RESULT_CSV"

dataset_trials() {
    case "$1" in
        Reddit) echo 1 ;;
        ogbn-arxiv|ogbn-products) echo 1 ;;
        *) echo 3 ;;
    esac
}

dataset_base_cmd() {
    case "$1" in
        Cora)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='Cora' --norm_feat='false' --fast='true' --hidden_dim=16 --init_batch=256 --sample_size=400 --early_stop=10 --wd=5e-4"
            ;;
        CiteSeer)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='CiteSeer' --norm_feat='false' --fast='true' --hidden_dim=16 --init_batch=256 --sample_size=400 --early_stop=10 --wd=5e-4"
            ;;
        PubMed)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='PubMed' --norm_feat='true' --fast='true' --hidden_dim=16 --init_batch=256 --sample_size=400 --early_stop=10 --wd=5e-4"
            ;;
        Reddit)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='Reddit' --norm_feat='false' --fast='true' --hidden_dim=128 --init_batch=${REDDIT_INIT_BATCH} --sample_size=${REDDIT_SAMPLE_SIZE} --early_stop=20 --report=${REDDIT_REPORT} --wd=1e-4"
            ;;
        ogbn-arxiv)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='ogbn-arxiv' --norm_feat='false' --fast='true' --init_batch=1024 --sample_size=10240 --early_stop=-1 --wd=0.0 --batch_norm='true' --hidden_dim=256 --num_layers=2 --drop=0.5 --scale_factor=5 --epochs=${ARXIV_EPOCHS} --lr=0.001 --report=${ARXIV_REPORT}"
            ;;
        ogbn-products)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='ogbn-products' --norm_feat='false' --fast='true' --init_batch=1024 --sample_size=15360 --early_stop=-1 --wd=0.0 --batch_norm='false' --hidden_dim=256 --num_layers=2 --drop=0.5 --scale_factor=1 --epochs=${PRODUCTS_EPOCHS} --lr=0.01 --report=${PRODUCTS_REPORT}"
            ;;
        *)
            echo "Unknown dataset: $1" >&2
            return 1
            ;;
    esac
}

dataset_mini_extra() {
    case "$1" in
        Cora)
            echo "--samp_inference='true' --inference_init_batch=256 --inference_sample_size=1280"
            ;;
        CiteSeer)
            echo "--samp_inference='true' --inference_init_batch=256 --inference_sample_size=2560"
            ;;
        PubMed)
            echo "--samp_inference='true' --inference_init_batch=256 --inference_sample_size=2560"
            ;;
        Reddit)
            echo "--samp_inference='true' --inference_init_batch=${REDDIT_INF_INIT_BATCH} --inference_sample_size=${REDDIT_INF_SAMPLE_SIZE}"
            ;;
        ogbn-arxiv)
            echo "--samp_inference='true' --inference_init_batch=${ARXIV_INF_INIT_BATCH} --inference_sample_size=${ARXIV_INF_SAMPLE_SIZE}"
            ;;
        ogbn-products)
            echo "--samp_inference='true' --inference_init_batch=${PRODUCTS_INF_INIT_BATCH} --inference_sample_size=${PRODUCTS_INF_SAMPLE_SIZE}"
            ;;
        *)
            echo "Unknown dataset: $1" >&2
            return 1
            ;;
    esac
}

build_cmd() {
    local dataset="$1"
    local mode="$2"
    local base_cmd
    local mini_extra

    base_cmd="$(dataset_base_cmd "$dataset")"
    mini_extra="$(dataset_mini_extra "$dataset")"

    case "$mode" in
        fast_full)
            echo "$base_cmd"
            ;;
        gcn_full)
            # Use sed to robustly flip the flag regardless of quote style.
            echo "$base_cmd" | sed -e "s/--fast='true'/--fast='false'/" -e "s/--fast=true/--fast=false/"
            ;;
        fast_mini)
            echo "$base_cmd $mini_extra"
            ;;
        *)
            echo "Unknown mode: $mode" >&2
            return 1
            ;;
    esac
}

extract_metric() {
    local pattern="$1"
    local file="$2"
    local value
    value="$(grep -E "$pattern" "$file" | tail -n 1 || true)"
    printf "%s" "$value"
}

render_progress() {
    local done="$1"
    local total="$2"
    local start_ts="$3"
    local width=30
    local filled=0
    local pct=0
    local now_ts
    local elapsed
    local eta=0
    local bar_fill
    local bar_empty

    if (( total > 0 )); then
        filled=$(( done * width / total ))
        pct=$(( done * 100 / total ))
    fi

    bar_fill="$(printf "%${filled}s" "" | tr ' ' '#')"
    bar_empty="$(printf "%$((width - filled))s" "" | tr ' ' '-')"
    now_ts="$(date +%s)"
    elapsed=$(( now_ts - start_ts ))

    if (( done > 0 )); then
        eta=$(( elapsed * (total - done) / done ))
    fi

    echo "[PROGRESS] [${bar_fill}${bar_empty}] ${done}/${total} (${pct}%) | elapsed=${elapsed}s | eta=${eta}s"
}

run_one() {
    local dataset="$1"
    local mode="$2"
    local trial="$3"
    local run_idx="$4"
    local run_total="$5"
    local run_start_ts="$6"
    local cmd
    local log_file
    local status
    local cmd_pid
    local timer_pid
    local timer_interval="$TASK_TIMER_INTERVAL"
    local task_start_ts
    local run_status
    local acc
    local batch_time
    local total_time

    cmd="$(build_cmd "$dataset" "$mode")"
    log_file="$OUT_DIR/logs/${dataset}_${mode}_trial${trial}.log"

    render_progress "$((run_idx - 1))" "$run_total" "$run_start_ts"
    echo ">>> [${run_idx}/${run_total}] [$dataset][$mode][trial $trial] $cmd"

    task_start_ts="$(date +%s)"
    set +e
    bash -lc "$cmd" > >(tee "$log_file") 2>&1 &
    cmd_pid=$!

    (
        while kill -0 "$cmd_pid" 2>/dev/null; do
            sleep "$timer_interval"
            if ! kill -0 "$cmd_pid" 2>/dev/null; then
                break
            fi
            printf "[TASK] [%s][%s][trial %s] elapsed=%ss\n" "$dataset" "$mode" "$trial" "$(( $(date +%s) - task_start_ts ))"
        done
    ) &
    timer_pid=$!

    wait "$cmd_pid"
    status=$?
    kill "$timer_pid" 2>/dev/null || true
    wait "$timer_pid" 2>/dev/null || true
    set -e

    run_status="ok"
    acc="NA"
    batch_time="NA"
    total_time="NA"

    if [[ "$status" -ne 0 ]]; then
        run_status="failed($status)"
    fi

    if grep -qiE "OOM|out of memory" "$log_file"; then
        run_status="oom"
    fi

    if [[ "$run_status" == "ok" ]]; then
        acc="$(extract_metric "^\[ACC\]" "$log_file" | sed -E 's/.*accuracy: ([0-9.]+) %.*/\1/' || true)"
        batch_time="$(extract_metric "^\[BATCH TIME\]" "$log_file" | sed -E 's/.*\] ([0-9.]+) seconds.*/\1/' || true)"
        total_time="$(extract_metric "^\[TOTAL TIME\]" "$log_file" | sed -E 's/.*\] ([0-9.]+) seconds.*/\1/' || true)"

        [[ -n "$acc" ]] || acc="NA"
        [[ -n "$batch_time" ]] || batch_time="NA"
        [[ -n "$total_time" ]] || total_time="NA"
    fi

    printf '"%s","%s",%s,"%s","%s","%s","%s","%s","%s"\n' \
        "$dataset" "$mode" "$trial" "$run_status" "$acc" "$batch_time" "$total_time" "$log_file" "$cmd" >> "$RESULT_CSV"

    render_progress "$run_idx" "$run_total" "$run_start_ts"
}

read -r -a dataset_arr <<< "$DATASETS"
read -r -a mode_arr <<< "$MODES"

total_runs=0
for dataset in "${dataset_arr[@]}"; do
    trials="$(dataset_trials "$dataset")"
    for mode in "${mode_arr[@]}"; do
        total_runs=$((total_runs + trials))
    done
done

run_idx=0
run_start_ts="$(date +%s)"

for dataset in "${dataset_arr[@]}"; do
    trials="$(dataset_trials "$dataset")"
    for mode in "${mode_arr[@]}"; do
        for ((trial = 1; trial <= trials; trial++)); do
            run_idx=$((run_idx + 1))
            run_one "$dataset" "$mode" "$trial" "$run_idx" "$total_runs" "$run_start_ts"
        done
    done
done

echo "Saved raw results to: $RESULT_CSV"

if [[ "$RUN_PLOTS" == "1" ]]; then
    if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/plot_benchmark_results.py" --input "$RESULT_CSV" --outdir "$OUT_DIR"; then
        echo "Plotting failed. Raw CSV and logs are still available at: $OUT_DIR" >&2
    fi
fi

echo "Done. Output directory: $OUT_DIR"
