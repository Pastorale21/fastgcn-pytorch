#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/param_sweeps/$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-PubMed Reddit}"
EXPERIMENTS="${EXPERIMENTS:-sample_size init_batch samp_dist}"
RUN_PLOTS="${RUN_PLOTS:-1}"
TASK_TIMER_INTERVAL="${TASK_TIMER_INTERVAL:-30}"
DRY_RUN="${DRY_RUN:-0}"

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
printf "dataset,experiment,param_value,trial,status,acc,batch_time,total_time,log_path,command\n" > "$RESULT_CSV"

dataset_trials() {
    local dataset="$1"
    case "$dataset" in
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
            echo "$PYTHON_BIN fastgcn_test.py --dataset='Reddit' --norm_feat='false' --fast='true' --hidden_dim=128 --init_batch=1024 --sample_size=5120 --early_stop=20 --report=20 --wd=1e-4"
            ;;
        ogbn-arxiv)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='ogbn-arxiv' --norm_feat='false' --fast='true' --init_batch=1024 --sample_size=10240 --early_stop=-1 --wd=0.0 --batch_norm='true' --hidden_dim=256 --num_layers=2 --drop=0.5 --scale_factor=5 --epochs=200 --lr=0.001 --report=50"
            ;;
        ogbn-products)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='ogbn-products' --norm_feat='false' --fast='true' --init_batch=1024 --sample_size=15360 --early_stop=-1 --wd=0.0 --batch_norm='false' --hidden_dim=256 --num_layers=2 --drop=0.5 --scale_factor=1 --epochs=300 --lr=0.01 --report=200"
            ;;
        *)
            echo "Unknown dataset: $1" >&2
            return 1
            ;;
    esac
}

experiment_values() {
    local dataset="$1"
    local experiment="$2"
    case "$experiment" in
        sample_size)
            case "$dataset" in
                Cora|CiteSeer) echo "100 200 400 800" ;;
                PubMed) echo "100 200 400 800 1600" ;;
                Reddit) echo "1024 2048 5120 10240" ;;
                ogbn-arxiv) echo "2048 4096 10240 20480" ;;
                ogbn-products) echo "4096 8192 15360 30720" ;;
                *) return 1 ;;
            esac
            ;;
        init_batch)
            case "$dataset" in
                Cora|CiteSeer|PubMed) echo "64 128 256 512" ;;
                Reddit) echo "256 512 1024 2048" ;;
                ogbn-arxiv) echo "256 512 1024 2048" ;;
                ogbn-products) echo "256 512 1024 2048" ;;
                *) return 1 ;;
            esac
            ;;
        samp_dist)
            echo "uniform importance"
            ;;
        *)
            echo "Unknown experiment: $experiment" >&2
            return 1
            ;;
    esac
}

build_cmd() {
    local dataset="$1"
    local experiment="$2"
    local value="$3"
    local base_cmd

    base_cmd="$(dataset_base_cmd "$dataset")"

    case "$experiment" in
        sample_size)
            echo "$base_cmd" | sed -E "s/--sample_size=[^ ]+/--sample_size=${value}/"
            ;;
        init_batch)
            echo "$base_cmd" \
                | sed -E "s/--init_batch=[^ ]+/--init_batch=${value}/" \
                | sed -E "s/--early_stop=[^ ]+/--early_stop=-1/"
            ;;
        samp_dist)
            echo "$base_cmd --samp_dist='${value}'"
            ;;
        *)
            echo "Unknown experiment: $experiment" >&2
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

has_complete_results() {
    local file="$1"
    grep -qE "^\[ACC\]" "$file" \
        && grep -qE "^\[BATCH TIME\]" "$file" \
        && grep -qE "^\[TOTAL TIME\]" "$file"
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
    local experiment="$2"
    local value="$3"
    local trial="$4"
    local run_idx="$5"
    local run_total="$6"
    local run_start_ts="$7"
    local cmd
    local safe_value
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

    cmd="$(build_cmd "$dataset" "$experiment" "$value")"
    safe_value="$(printf "%s" "$value" | tr ' /' '__')"
    log_file="$OUT_DIR/logs/${dataset}_${experiment}_${safe_value}_trial${trial}.log"

    render_progress "$((run_idx - 1))" "$run_total" "$run_start_ts"
    echo ">>> [${run_idx}/${run_total}] [$dataset][$experiment=$value][trial $trial] $cmd"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '"%s","%s","%s",%s,"%s","%s","%s","%s","%s","%s"\n' \
            "$dataset" "$experiment" "$value" "$trial" "dry_run" "NA" "NA" "NA" "$log_file" "$cmd" >> "$RESULT_CSV"
        render_progress "$run_idx" "$run_total" "$run_start_ts"
        return
    fi

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
            printf "[TASK] [%s][%s=%s][trial %s] elapsed=%ss\n" "$dataset" "$experiment" "$value" "$trial" "$(( $(date +%s) - task_start_ts ))"
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

    if has_complete_results "$log_file"; then
        acc="$(extract_metric "^\[ACC\]" "$log_file" | sed -E 's/.*accuracy: ([0-9.]+) %.*/\1/' || true)"
        batch_time="$(extract_metric "^\[BATCH TIME\]" "$log_file" | sed -E 's/.*\] ([0-9.]+) seconds.*/\1/' || true)"
        total_time="$(extract_metric "^\[TOTAL TIME\]" "$log_file" | sed -E 's/.*\] ([0-9.]+) seconds.*/\1/' || true)"

        [[ -n "$acc" ]] || acc="NA"
        [[ -n "$batch_time" ]] || batch_time="NA"
        [[ -n "$total_time" ]] || total_time="NA"
        run_status="ok"
    else
        if [[ "$status" -ne 0 ]]; then
            run_status="failed($status)"
        fi

        if grep -qiE "CUDA out of memory|out of memory|Killed" "$log_file"; then
            run_status="oom"
        fi
    fi

    printf '"%s","%s","%s",%s,"%s","%s","%s","%s","%s","%s"\n' \
        "$dataset" "$experiment" "$value" "$trial" "$run_status" "$acc" "$batch_time" "$total_time" "$log_file" "$cmd" >> "$RESULT_CSV"

    render_progress "$run_idx" "$run_total" "$run_start_ts"
}

read -r -a dataset_arr <<< "$DATASETS"
read -r -a experiment_arr <<< "$EXPERIMENTS"

total_runs=0
for dataset in "${dataset_arr[@]}"; do
    trials="$(dataset_trials "$dataset")"
    for experiment in "${experiment_arr[@]}"; do
        read -r -a values <<< "$(experiment_values "$dataset" "$experiment")"
        total_runs=$((total_runs + ${#values[@]} * trials))
    done
done

run_idx=0
run_start_ts="$(date +%s)"
for dataset in "${dataset_arr[@]}"; do
    trials="$(dataset_trials "$dataset")"
    for experiment in "${experiment_arr[@]}"; do
        read -r -a values <<< "$(experiment_values "$dataset" "$experiment")"
        for value in "${values[@]}"; do
            for trial in $(seq 1 "$trials"); do
                run_idx=$((run_idx + 1))
                run_one "$dataset" "$experiment" "$value" "$trial" "$run_idx" "$total_runs" "$run_start_ts"
            done
        done
    done
done

if [[ "$RUN_PLOTS" == "1" && "$DRY_RUN" != "1" ]]; then
    "$PYTHON_BIN" scripts/plot_param_sweep_results.py --input "$RESULT_CSV" --outdir "$OUT_DIR"
fi

echo "Results saved to: $RESULT_CSV"
