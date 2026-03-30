#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-Cora CiteSeer PubMed Reddit ogbn-arxiv ogbn-products}"
MODES="${MODES:-fast_full gcn_full fast_mini}"
RUN_PLOTS="${RUN_PLOTS:-1}"

mkdir -p "$OUT_DIR/logs"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/.mpl-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OUT_DIR/.cache}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"
RESULT_CSV="$OUT_DIR/results.csv"
printf "dataset,mode,trial,status,acc,batch_time,total_time,log_path,command\n" > "$RESULT_CSV"

dataset_trials() {
    case "$1" in
        Reddit) echo 1 ;;
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
            echo "$PYTHON_BIN fastgcn_test.py --dataset='Reddit' --norm_feat='false' --fast='true' --hidden_dim=128 --init_batch=1024 --sample_size=5120 --early_stop=20 --wd=1e-4"
            ;;
        ogbn-arxiv)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='ogbn-arxiv' --norm_feat='false' --fast='true' --init_batch=1024 --sample_size=10240 --early_stop=-1 --wd=0.0 --batch_norm='true' --hidden_dim=256 --num_layers=2 --drop=0.5 --scale_factor=5 --epochs=1000 --lr=0.001"
            ;;
        ogbn-products)
            echo "$PYTHON_BIN fastgcn_test.py --dataset='ogbn-products' --norm_feat='false' --fast='true' --init_batch=1024 --sample_size=15360 --early_stop=-1 --wd=0.0 --batch_norm='false' --hidden_dim=256 --num_layers=2 --drop=0.5 --scale_factor=1 --epochs=1000 --lr=0.01"
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
            echo "--samp_inference='true' --inference_init_batch=1024 --inference_sample_size=15360"
            ;;
        ogbn-arxiv)
            echo "--samp_inference='true' --inference_init_batch=8192 --inference_sample_size=169343 --report=10"
            ;;
        ogbn-products)
            echo "--samp_inference='true' --inference_init_batch=32768 --inference_sample_size=491520 --report=250"
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
            echo "${base_cmd/--fast='true'/--fast='false'}"
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

run_one() {
    local dataset="$1"
    local mode="$2"
    local trial="$3"
    local cmd
    local log_file
    local status
    local run_status
    local acc
    local batch_time
    local total_time

    cmd="$(build_cmd "$dataset" "$mode")"
    log_file="$OUT_DIR/logs/${dataset}_${mode}_trial${trial}.log"

    echo ">>> [$dataset][$mode][trial $trial] $cmd"
    set +e
    bash -lc "$cmd" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
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
}

read -r -a dataset_arr <<< "$DATASETS"
read -r -a mode_arr <<< "$MODES"

for dataset in "${dataset_arr[@]}"; do
    trials="$(dataset_trials "$dataset")"
    for mode in "${mode_arr[@]}"; do
        for ((trial = 1; trial <= trials; trial++)); do
            run_one "$dataset" "$mode" "$trial"
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
