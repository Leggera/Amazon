#!/bin/bash

function normalize_text {
  awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}

if [ ! -d ./pretrain_data ]
then
    mkdir pretrain_data

    wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV.json.gz
    python3 to_json.py reviews_Movies_and_TV.json.gz pretrain_data/reviews_Movies_and_TV

    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar -xvf aclImdb_v1.tar.gz

    for j in train/pos train/neg test/pos test/neg train/unsup; do
	    for i in `ls aclImdb/$j`; do cat aclImdb/$j/$i >> temp; awk 'BEGIN{print;}' >> temp; done
	    normalize_text temp
	    mv temp-norm aclImdb/$j/norm.txt
	    rm temp
    done

    rm aclImdb_v1.tar.gz
    rm reviews_Movies_and_TV.json.gz

    cat pretrain_data/reviews_Movies_and_TV >> temp
    normalize_text temp
    mv temp-norm pretrain_data/reviews_Movies_and_TV_norm.txt
    rm temp

    mv aclImdb/train/pos/norm.txt pretrain_data/full-train-pos.txt
    mv aclImdb/train/neg/norm.txt pretrain_data/full-train-neg.txt
    mv aclImdb/test/pos/norm.txt pretrain_data/test-pos.txt
    mv aclImdb/test/neg/norm.txt pretrain_data/test-neg.txt
    mv aclImdb/train/unsup/norm.txt pretrain_data/train-unsup.txt
    cat pretrain_data/reviews_Movies_and_TV_norm.txt >> pretrain_data/train-unsup.txt
fi

cat ./pretrain_data/full-train-pos.txt ./pretrain_data/full-train-neg.txt ./pretrain_data/test-pos.txt ./pretrain_data/test-neg.txt ./pretrain_data/train-unsup.txt > pretrain_data/alldata.txt
awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < pretrain_data/alldata.txt > pretrain_data/alldata-id.txt

default_models=('-cbow 0 -sample 1e-2' '-cbow 1 -sample 1e-4')
default_parameters=('-size 150 -alpha 0.05 -window 10 -negative 25 -iter 25 -threads 32')
min_counts=('-min_count 1' '-min_count 3')

mkdir d2v_pretrained_IMDB
d2v_IMDB_fold="d2v_pretrained_IMDB/"
mkdir C_pretrained_IMDB
C_IMDB_fold="C_pretrained_IMDB/"


for model in "${default_models[@]}"; do
    for min_count in "${min_counts[@]}"; do
	d2v_out="doc2vec ""$model""$min_count"".txt"
	python3 run_doc2vec_proper.py -output "$d2v_IMDB_fold""$d2v_out" -train pretrain_data/alldata-id.txt $min_count $model $default_parameters
	#python3 run_doc2vec_20ng.py -output "$_20ng_fold""$d2v_out" $min_count $model $default_parameters &    
    done
done


#python3 IMDB_concat_dataframe.py -classifier linearsvc -vectors "$d2v_IMDB_fold"
python3 IMDB_concat_dataframe.py -classifier lr -vectors "$d2v_IMDB_fold"
