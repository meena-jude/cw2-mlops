#!/usr/bin/env bash
# Phase 1: data asset, compute, environment and the first two training runs.
# Run from the repo root in Azure Cloud Shell (Bash):  bash scripts/phase1_cloudshell.sh
# Safe to re-run: anything that already exists is skipped.
set -euo pipefail

RG="rg-cw2-mlops"
WS="mlw-cw2"
CLUSTER="cpu-cluster"
ENV_NAME="aiops-sklearn"
DATA_NAME="aiops-model-ready"
DATA_DIR="${DATA_DIR:-$HOME/cw2data/aiops-v1}"

step() { printf '\n=== %s ===\n' "$*"; }
exists() { "$@" -o none >/dev/null 2>&1; }

step "0/7 Subscription, defaults and the ml extension"
az account show --query "{subscription:name, id:id}" -o table
az configure --defaults group="$RG" workspace="$WS"
az extension add -n ml --upgrade -y --only-show-errors

step "1/7 Check the data folder: $DATA_DIR"
for f in model_ready_train.csv model_ready_val.csv model_ready_test.csv abt_clean.csv \
  preprocessing_bundle.joblib; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "Missing $DATA_DIR/$f (see guide step B2)"; exit 1; }
done
echo "all 5 files found"

step "2/7 Data manifest (copy this JSON into data/manifests/v1.json)"
python3 scripts/make_manifest.py "$DATA_DIR" --version 1 --out "$HOME/v1.json"

step "3/7 Compute cluster $CLUSTER (F4s_v2, 0-2 nodes, scales to 0 after 120 s)"
if exists az ml compute show -n "$CLUSTER"; then
  echo "already exists"
else
  az ml compute create -n "$CLUSTER" --type amlcompute --size Standard_F4s_v2 \
    --tier dedicated --min-instances 0 --max-instances 2 \
    --idle-time-before-scale-down 120 --tags project=cw2 --query name -o tsv
fi

step "4/7 Build images on $CLUSTER (ACR Tasks are paused for student-credit subscriptions)"
az ml workspace update -n "$WS" --image-build-compute "$CLUSTER" -o none
echo "image_build_compute = $(az ml workspace show -n "$WS" --query image_build_compute -o tsv)"

step "5/7 Environment $ENV_NAME:1"
if exists az ml environment show -n "$ENV_NAME" -v 1; then
  echo "already exists"
else
  az ml environment create -f environments/aiops-env.yml --query "[name, version]" -o tsv
fi

step "6/7 Data asset $DATA_NAME:1 (uploads ~86 MB)"
if exists az ml data show -n "$DATA_NAME" -v 1; then
  echo "already exists"
else
  az ml data create -f data/aiops-model-ready-v1.yml --path "$DATA_DIR" \
    --query "[name, version, path]" -o tsv
fi

step "7/7 Submit Iter 0 (persistence) and Iter 1 (logistic regression)"
for pair in 0:persistence 1:logreg; do
  it="${pair%%:*}"
  model="${pair#*:}"
  job=$(az ml job create -f jobs/train.yml \
    --set inputs.model="$model" display_name="iter${it}-${model}" --query name -o tsv)
  echo "iter${it}-${model} submitted as job $job"
done

echo
echo "Done. Studio > Jobs > aiops-incident-early-warning."
echo "The first run builds the image on $CLUSTER (about 10-20 min), then each job takes ~2 min."
