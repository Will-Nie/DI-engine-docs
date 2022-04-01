#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 path_to_main.py" >&2
  exit 1
fi

for i in {1..5}; do
  SEED=$i python $1
done
