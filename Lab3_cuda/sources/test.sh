#!/usr/bin/env bash
set -euo pipefail

dataset="0"

while [ $# -gt 0 ]; do
  case "$1" in
    -dataset)
      dataset="$2"; shift 2;;
    *)
      echo "Usage: $0 -dataset <number>"
      exit 1 ;;
  esac
done

echo ">> Building template binary"
make template

echo ">> Running on Dataset/${dataset}"
./Convolution_template -e "./Convolution/Dataset/${dataset}/output.ppm" \
    -i "./Convolution/Dataset/${dataset}/input0.ppm,./Convolution/Dataset/${dataset}/input1.raw" \
    -o "./test_result_${dataset}.ppm" \
    -t image
