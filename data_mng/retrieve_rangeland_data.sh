#!/bin/bash

destination_dir="/home/raraki/waves/projects/smap-drydown/data/rap-vegetation-cover-v3"
mkdir -p "$destination_dir"

start_year=2015
end_year=2022

for year in $(seq $start_year $end_year); do
    url="http://rangeland.ntsg.umt.edu/data/rap/rap-vegetation-cover/v3/vegetation-cover-v3-${year}.tif"
    wget -c -P "$destination_dir" "$url"
done

wget -c -P "$destination_dir" "http://rangeland.ntsg.umt.edu/data/rap/rap-vegetation-cover/v3/README"
