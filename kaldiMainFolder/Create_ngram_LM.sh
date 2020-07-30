#!/bin/bash

#*******************************************************************************#
#set-up for single machine or cluster based execution
. ./cmd.sh
#set the paths to binaries and other executables
[ -f path.sh ] && . ./path.sh

kaldi_root_dir='/saildisk/tools/kaldi'
basepath='/home/kunal/kaldiMainFolder-std_features'


#*******************************************************************************#
#Creating input to the LM training

cat $basepath/data/train_multi/text | awk '{first = $1; $1 = ""; print $0; }' > $basepath/data/train_multi/trans

while read line

do
echo "<s> $line </s>" >> $basepath/data/train_multi/lm_train.txt
done <$basepath/data/train_multi/trans

#*******************************************************************************#

lm_arpa_path=$basepath/data/local/lm

train_dict=dict
train_lang=lang_trigram_aurora4
train_folder=train

n_gram=3 # This specifies bigram or trigram. for bigram set n_gram=2 for tri_gram set n_gram=3

echo ============================================================================
echo "                   Creating  n-gram LM               	        "
echo ============================================================================

rm -rf $basepath/data/local/$train_dict/lexiconp.txt $basepath/data/local/$train_lang $basepath/data/local/tmp_$train_lang $basepath/data/$train_lang
mkdir $basepath/data/local/tmp_$train_lang

utils/prepare_lang.sh --num-sil-states 3 data/local/$train_dict '!SIL' data/local/$train_lang data/$train_lang

/home/kunal/IRSTLM/bin/build-lm.sh -i $basepath/data/$train_folder/lm_train.txt -n $n_gram -o $basepath/data/local/tmp_$train_lang/lm_phone_bg.ilm.gz

gunzip -c $basepath/data/local/tmp_$train_lang/lm_phone_bg.ilm.gz | utils/find_arpa_oovs.pl data/$train_lang/words.txt  > data/local/tmp_$train_lang/oov.txt

gunzip -c $basepath/data/local/tmp_$train_lang/lm_phone_bg.ilm.gz | grep -v '<s> <s>' | grep -v '<s> </s>' | grep -v '</s> </s>' | grep -v 'SIL' | $kaldi_root_dir/src/lmbin/arpa2fst - | fstprint | utils/remove_oovs.pl data/local/tmp_$train_lang/oov.txt | utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=data/$train_lang/words.txt --osymbols=data/$train_lang/words.txt --keep_isymbols=false --keep_osymbols=false | fstrmepsilon > data/$train_lang/G.fst 
$kaldi_root_dir/src/fstbin/fstisstochastic data/$train_lang/G.fst 

echo ============================================================================
echo "                   End of Script             	        "
echo ============================================================================