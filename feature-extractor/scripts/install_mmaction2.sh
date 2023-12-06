#!/bin/bash
# Install mmaction2
pip install -U openmim
mim install mmengine
mim install mmcv
pip install mmaction2

# fix the modulenotfounderror in mmaction2

# Get the location of the mmaction2 package
package_location=$(pip show mmaction2 | grep -i "Location" | awk -F 'Location: ' 'NF>1{print $2}' | tr -d '\r')

# Specify the subdirectory to append to the package location
subdirectory="mmaction/models/localizers"

# Create the target directory by appending the subdirectory
target_directory="$package_location/$subdirectory"

unzip "scripts/drn.zip"

mv "drn" "$target_directory"

echo "Moved 'drn' folder into '$target_directory'."
