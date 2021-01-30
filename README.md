# ehr_classification

## Getting started

Create and activate and environment, download [SpaCy](https://spacy.io) language models if you want to do more than
 tokenization. Finally install the `ehr_classification` package.
 
```
# create a conda environment
conda env create -f environment_tf2.yml
conda activate ehr_tf2

#  download spacy models
python -m spacy download en_core_web_sm

# install the package in develop mode
cd ehr_classification
python setup.py develop
```

## Usage
Once the package is installed, two commands for the classification of medical notes are available in the conda environment.

### tokenize
Takes parquet file or directory and writes tokenized text to output file.
Expects a `NoteTXT` column that contains the medical note text, this will be tokenized mapped to word vectors.

```bash
# To tokenize into text files (e.g. to train embeddings), do:
tokenize text.parquet text.txt

#To tokenize and map to features (numpy array) do:
tokenize text.parquet no_out -b
```


### predict_ehr
Does forward pass using trained language models and outputs predictions.
We trained event and history models for application on discharge summaries and progress notes.
The parallelized commands below expect tokenized input from the previous step.

```bash
# tokenize and predict a single parquet file (or multiple parquet files)
predict_ehr -t event output_folder text1.parquet text2.parquet ... textn.parquet

# run in parallel with 4 gpus on a single machine, using files that have been tokenized before:
parallel -j 4 predict_ehr . -t event -g {%} -s 1 -i pickle {} ::: *.pic.lz4
parallel -j 4 predict_ehr . -t history -g {%} -s 1 -i pickle {} ::: *.pic.lz4
```