#!/bin/bash

# Set source and target directories
source_dir="/home/labs/amit/shuangyi/Project_MM_2023/scdata_MARS" #scdata_SPID"
target_dir="/home/labs/amit/noamsh/data/mm_2023/all_scdata"

# Ensure the target directory exists
mkdir -p "$target_dir"

# Iterate over files in source directory
for source_file in "$source_dir"/*; do
  if [ -f "$source_file" ]; then
    # Create a symlink in the target directory for each file in the source directory
    ln -s "$source_file" "$target_dir"
  fi
done