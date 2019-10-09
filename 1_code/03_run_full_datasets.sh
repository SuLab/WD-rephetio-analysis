#!/bin/bash

LOAD_DIR='../2_pipeline/01_querying_wikidata_for_hetnet_edges/out/'

for folder in ${LOAD_DIR}20*
    do python full_dataset_training.py $folder
done
