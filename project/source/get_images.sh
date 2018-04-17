#!/bin/bash

baseUrl="http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/"
declare -a imageSets=("T01-T05.zip" "T06-T10.zip" "T11-T15.zip" "T16-T20.zip" "T21-T25.zip")

for imageSet in "${imageSets[@]}"; do
    link=$baseUrl$imageSet
    echo $link
    if [ ! -f $imageSet ]; then
        curl $link > $imageSet
        unzip $imageSet
    fi
done

find -name '*jpg' -exec mogrify -format ppm {} \;
