#!/bin/bash

INPUT="test/src/core/Tensor_test.cpp"
OUTPUT="test/src/core/TensorParallel_test.cpp"

declare -A REPLACEMENTS=(
  ["Tensor"]="TensorParallel"
  ["tensor_test"]="tensorparallel_test"
)

cmd="sed"

for key in "${!REPLACEMENTS[@]}"; do
  cmd+=" -e s|${key}|${REPLACEMENTS[$key]}|g"
done

$cmd "$INPUT" > "$OUTPUT"
