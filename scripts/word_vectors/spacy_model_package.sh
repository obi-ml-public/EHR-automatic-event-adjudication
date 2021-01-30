# create model packages that can be loaded by spacy

# remember to adjust the json file of the word vector metadata for versioning before packaging them
python -m spacy package /home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result_spacy_2.2.4/en_mgb_fasttext_200k /home/mhomilius/projects/ehr_classification/spacy_models/2.2.4/

# 2.3.5 versions
python -m spacy package /home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result_spacy_2.3.5/en_mgb_fasttext_200k /home/mhomilius/projects/ehr_classification/spacy_models/2.3.5/


# do this to install the package - this is necessary to customize the tokenizer rules
# this is for the 2.2.4 version
cd ~/projects/ehr_classification/spacy_models/2.2.4/en_mgb_fasttext_200k-0.0.1

# this installs (symlinks) it as development package
python setup.py develop

# this creates a compressed source distribution (TODO test installation)
python setup.py sdist

# copy to shared location
cp dist/en_mgb_fasttext_200k-0.0.1.tar.gz /mnt/obi0/phi/ehr/spacy_models/2.3.5/