#!/bin/bash

# Assume you are in your_project directory
# i.e. ~/project-code/

newdir="language_model"
if [ ! -d "$newdir" ]; then
    mkdir "$newdir"
fi

cd "$newdir"

lmdir="lm_1b"
if [ ! -d "$lmdir" ]; then
    mkdir "$lmdir"
fi

wget https://raw.githubusercontent.com/tensorflow/models/master/research/lm_1b/data_utils.py -O "$lmdir/data_utils.py";

datdir="data"
if [ ! -d "$datdir" ]; then
    mkdir "$datdir"
fi

# Download the Weights. Takes long time
url_base="http://download.tensorflow.org/models/LM_LSTM_CNN/"
shards="all_shards-2016-09-10/"
model_urls=(ckpt-base ckpt-char-embedding ckpt-lstm ckpt-softmax0 ckpt-softmax1 ckpt-softmax2 ckpt-softmax3 ckpt-softmax4 ckpt-softmax5 ckpt-softmax6 ckpt-softmax7 ckpt-softmax8)
ITER=1; NFILES=${#model_urls[@]}
for a_url in ${model_urls[*]} 
do
    echo "-------------------------------------------------------------------------------------------------------"
    echo "	 Downloading file $ITER out of $NFILES"
    echo "-------------------------------------------------------------------------------------------------------"
    
    if [ -f "$datdir/$a_url" ]; then
        echo "$a_url is already downloaded, skipping"
    else
        wget "$url_base$shards$a_url" -O "$datdir/$a_url" -q --show-progress
    fi

    let ITER=${ITER}+1
done

# Gotta do separate because this ones have different bases
wget $url_base"test/news.en.heldout-00000-of-00050" -O "$datdir/news.en.heldout-00000-of-00050" -q --show-progress
wget $url_base"graph-2016-09-10.pbtxt" -O "$datdir/graph-2016-09-10.pbtxt" -q --show-progress
wget $url_base"vocab-2016-09-10.txt" -O "$datdir/vocab-2016-09-10.txt" -q --show-progress

echo "data/" > ".gitignore"