#!/bin/bash

# Check if "-log" is passed as an argument
if [[ "$*" == *"-log"* ]]; then
  include_all_log=true
  dst_dir="./acc_results_with_info"
else
  include_all_log=false
  dst_dir="./plots_acc_result"
fi

# Directory where the files are located
src_dir="./Spintorch_packages/plots"

# Function to copy files
copy_files() {
  file_pattern="$1"
  find "$src_dir" -name "$file_pattern" -print0 | while IFS= read -r -d '' file; do
    # Remove the source directory from the file path, preserving the structure
    relative_path="${file#$src_dir}"

    # Create the destination directory
    mkdir -p "$dst_dir/$(dirname "$relative_path")"

    # Copy the file
    cp "$file" "$dst_dir/$relative_path"
  done
}

# Find and copy .txt files
copy_files '*.txt'

# Find and copy .pkl files
copy_files '*.pkl'

# Find and copy all files starting with 'aa' if "-log" was passed
if $include_all_log ; then
  copy_files 'aa_*.*'
fi

# Create a .zip file if "-zip" was passed
if [[ "$*" == *"-zip"* ]]; then
  zip -r "${dst_dir}.zip" "$dst_dir"
fi

echo 'process done!'
