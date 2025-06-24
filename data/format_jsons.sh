#!/usr/bin/bash

for file in *.json; do
    mv ${file} ${file}.tmp
    python -m json.tool ${file}.tmp > ${file}
    rm ${file}.tmp
done