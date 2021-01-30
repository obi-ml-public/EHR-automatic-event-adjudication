# initialize different spacy models for the mgb fastext word vectors
# spacy can prune them by frequency, so that less common tokens get mapped to the closest present token
# we have been using un-pruned vectors, but pruning slightly might be better given limited amount of training data

#### spacy 2.2.4
# do this from a spacy 2.2.4 environment
# en_mgb_fasttext_200k
spacy init-model \
en \
/home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result_spacy_2.2.4/en_mgb_fasttext_200k \
--prune-vectors 200000 \
--model-name mgb_fasttext_200k \
--vectors-name mgb_fasttext_200k.vectors \
--vectors-loc /home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result/filtered_20-05-23.bigram.vec

# en_mgb_fasttext_1m
spacy init-model \
en \
/home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result_spacy_2.2.4/en_mgb_fasttext_1m \
--prune-vectors 1000000 \
--model-name mgb_fasttext_1m \
--vectors-name mgb_fasttext_1m.vectors \
--vectors-loc /home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result/filtered_20-05-23.bigram.vec

# en_mgb_fasttext_full
spacy init-model \
en \
/home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result_spacy_2.2.4/en_mgb_fasttext_full \
--model-name mgb_fasttext_full \
--vectors-name mgb_fasttext_full.vectors \
--vectors-loc /home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result/filtered_20-05-23.bigram.vec

# now adjust the json file of the word vector metadata for versioning (0.0.1 etc.)


#### spacy 2.3.5 version
# do this from a spacy 2.3.5 environment
# en_mgb_fasttext_200k
spacy init-model \
en \
/home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result_spacy_2.3.5/en_mgb_fasttext_200k \
--prune-vectors 200000 \
--model-name mgb_fasttext_200k \
--vectors-name mgb_fasttext_200k.vectors \
--vectors-loc /home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/result/filtered_20-05-23.bigram.vec