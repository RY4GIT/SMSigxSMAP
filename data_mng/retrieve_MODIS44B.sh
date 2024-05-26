#!/bin/bash

# Base URL for the remote directory
base_url="https://e4ftl01.cr.usgs.gov/MOLT/MOD44B.061/"

# Output directory for downloaded files
output_dir="/home/waves/projects/smap-drydown/data/MOD44B.061"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Download the files, excluding .jpg images
wget --recursive --no-parent --reject "*.jpg*" -nd --directory-prefix="$output_dir" "$base_url"

# Check if wget was successful
if [ $? -eq 0 ]; then
    echo "All files downloaded successfully."
else
    echo "Error occurred in downloading files."
fi
