#!/bin/bash

# Stress Test Runner for Trusta AST Backend
# Usage: bash run_stress_tests.sh [finetune|inference|all]

MODE=${1:-all}
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# Ensure we are in the project root or can find the scripts
cd "$PROJECT_ROOT"

# Setup logging
LOG_DIR="tests/stress_tests/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_stress_tests_$TIMESTAMP.log"

# Redirect stdout and stderr to tee to write to log file and console
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "========================================"
echo "Starting Stress Tests"
echo "Mode: $MODE"
echo "Date: $(date)"
echo "Log File: $MAIN_LOG"
echo "========================================"

# Check if service is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Error: Service is not running on http://localhost:8000"
    echo "Please start the service first: bash scripts/linux/run_service.sh"
    exit 1
fi

# Function to run finetune tests
run_finetune() {
    echo ""
    echo ">>> Running Finetune Stress Tests..."
    python3 stress_test_finetune.py
    if [ $? -eq 0 ]; then
        echo ">>> Finetune Tests PASSED"
    else
        echo ">>> Finetune Tests FAILED"
        exit 1
    fi
}

# Function to run inference tests
run_inference() {
    echo ""
    echo ">>> Running Inference Stress Tests..."
    python3 tests/stress_tests/stress_test_inference.py
    if [ $? -eq 0 ]; then
        echo ">>> Inference Tests PASSED"
    else
        echo ">>> Inference Tests FAILED"
        exit 1
    fi
}

if [ "$MODE" == "finetune" ]; then
    run_finetune
elif [ "$MODE" == "inference" ]; then
    run_inference
else
    run_finetune
    run_inference
fi

echo ""
echo "========================================"
echo "All Stress Tests Completed Successfully"
echo "========================================"
