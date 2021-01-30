# run fasttest v0.9.2 on collection of notes
/home/mhomilius/projects/ehr_classification/data/word_vectors/fasttext/fastText-0.9.2/fasttext \
skipgram -wordNgrams 2 \
-input /home/mhomilius/projects/ehr_classification/data/data/filtered_notes_20-05-23/filtered_notes_20-05-23_all.txt \
-output result/filtered_20-05-23.bigram -thread 80
