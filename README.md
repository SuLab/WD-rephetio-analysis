# WD-rephetio-analysis

Repository for rephetio-algoritim repurposing on the WikiData knowledge repository.

# Pre-requsites

In order to run all of the notbooks in this repo, you must first download historical
Wikidata Dumps and process them into a local blazegraph instance.  These dumps, while
generally <50GB in file size, expand to upwards of 500GB once loaded into blazegraph.

## Downloading historical wikidata dumps

Recent Wikidata entity dumps are avalibe for download from
[wikimedia.org](https://dumps.wikimedia.org/wikidatawiki/entities/). Older wikidata entity dumps
can be found at
[archive.org](https://archive.org/details/wikimediadownloads?sort=-date&and[]="Wikidata+entity+dumps"&and[]=subject%3A"wikidata").


## Processing wikidata dumps for use with blazegraph

The wikimedia repository [wikidata-query-rdf](https://github.com/wikimedia/wikidata-query-rdf) contains all
the sofware required to process a wikidata entitiy dump and load the resultant data into blazegraph. Please
read the [getting started](https://github.com/wikimedia/wikidata-query-rdf/blob/master/docs/getting-started.md)
page for help in using this software.

## Requirments

Anaconda envrionment file `environment.yml` is provided. In addition to this file, the
[hetnet_ml](https://github.com/mmayers12/hetnet_ml) repo is requred to be in your python path for
feature extraction to properly work.

# Contents

## Notebooks & Scripts

Notebooks are numbered in the folder `2_code` and to be run in number order.

## System info

The machine learning portion of this repo was run on a workstation with 32 Cores and 378 GB ram.
Be sure and edit the `n_jobs` paramter in n_fold_CV_training.py (line 93) and full_dataset_training.py (line 80)
if your machine has fewer cores.
