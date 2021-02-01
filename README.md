# ehr_classification

## Requirements

The code in this repository was tested on Ubuntu 18.04 and Ubuntu 20.04,
using [Anaconda](https://docs.conda.io/en/latest/miniconda.html)
to create a python environment containing all required packages
(especially [Python 3.7](https://www.python.org) and [TensorFlow 2](https://www.tensorflow.org)).
See `environment_tf2.yml` for a listing of dependencies and versions.
We tested the training and prediction code with
NVIDIA GeForce GTX 1080, TITAN RTX and Tesla V100 GPUs and the following
GPU driver and CUDA versions: 
```
NVIDIA Driver Version: 450.80.02 CUDA Version: 11.0
```

The code should also work without GPUs, but will be substantially slower.

## Installation

To use the `ehr_classification` methods, create and activate a conda environment.
Then download [SpaCy](https://spacy.io) language models for general purpose word vectors.
Finally install the `ehr_classification` package.

```
# clone the repository
git clone git@github.com:obi-ml-public/EHR-automatic-event-adjudication.git
cd EHR-automatic-event-adjudication

# if conda not available, install it: https://docs.conda.io/en/latest/miniconda.html
# create a conda environment
conda env create -f environment_tf2.yml
conda activate ehr_tf2

#  download spacy models
python -m spacy download en_core_web_lg

# install the package in develop mode
python setup.py develop
```

## Usage
Once the package is installed, different commands for the classification of medical notes
are available in the conda environment.
The `ehr_classification` package can also be imported and used directly in other python programs.

### tokenize
Takes parquet file or directory and writes tokenized text to output file.
Expects a `NoteTXT` column that contains the medical note text,
this will be tokenized mapped to word vectors.

```bash
# change to directory with simulated notes
cd simulated_notes

# To tokenize into text files (e.g. to train embeddings), do:
tokenize --word_vectors en_core_web_lg simulated_notes.parquet simulated_notes_tokenized.txt

# To tokenize and map to features (numpy array) use -b flag
# requires these columns in parquet file: ['PatientID', 'NoteID', 'ContactDTS', 'NoteTXT']
tokenize --word_vectors en_core_web_lg text.parquet no_out -b

# To run multiple tokenization processes in parallel do:
parallel -j 10 tokenize --word_vectors en_core_web_lg -b {} no_out ::: ./*.parquet
```

### train_ehr
Basic training method that uses a labeled dataset containing notes and labels.
Expects labeled data as parquet file with at least columns `['NoteTXT', 'label']`,
name of model and output directory to save the final model,
as well as checkpoints and tensorboard files to track training status.

```bash
# start training
train_ehr labels.parquet Event_PCI ./models

# start tensorboard instance to track progress (in separate shell)
tensorboard --logdir=./models/tensorboard/
```

### predict_ehr
Does forward pass using trained language models and outputs predictions.
We trained event and history models for application on discharge summaries and progress notes.
The parallelized commands below expect tokenized input from the previous step. The command expects
output dir and parquet file as minimal arguments.

```bash
# set environment variables for models and word vectors
export PREDICT_EHR_VECTORS=en_core_web_lg
export PREDICT_EHR_MODELS=PATH/TO/MODELS

# run predictions for events on one or more parquet files
cd simulated_notes
predict_ehr -t event . simulated_notes.parquet
predict_ehr -t event $output_dir text1.parquet text2.parquet text3.parquet

# run on multiple files in parallel with 4 gpus, using text that has been tokenized before:
parallel -j 4 predict_ehr . -t event -g {%} -s 1 -i pickle {} ::: simulated_notes/*.pic.lz4
```
