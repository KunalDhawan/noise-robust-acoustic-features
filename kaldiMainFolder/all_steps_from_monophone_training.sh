#!/bin/bash
steps/train_mono.sh --nj 10 data/train_multi data/lang_trigram_aurora4 exp/mono

utils/mkgraph.sh --mono data/lang_trigram_aurora4 exp/mono exp/mono/graph

steps/align_si.sh --boost-silence 1.25 --nj 5 data/train_multi data/lang_trigram_aurora4 exp/mono exp/mono_ali

steps/train_deltas.sh 2000 16000 data/train_multi data/lang_trigram_aurora4 exp/mono_ali exp/tri1

utils/mkgraph.sh data/lang_trigram_aurora4 exp/tri1 exp/tri1/graph

steps/align_si.sh --nj 5 data/train_multi data/lang_trigram_aurora4 exp/tri1 exp/tri1_ali

steps/nnet2/train_tanh.sh --initial-learning-rate 0.015 --final-learning-rate 0.002 --num-hidden-layers 5 --minibatch-size 128 --hidden-layer-dim 512 --num-jobs-nnet 5 --num-epochs 7 data/train_multi data/lang_trigram_aurora4 exp/tri1_ali exp/DNN

utils/mkgraph.sh data/lang_trigram_aurora4 exp/DNN exp/DNN/graph


for i in test_0330_airport_wv1 test_0330_airport_wv2 test_0330_babble_wv1 test_0330_babble_wv2 test_0330_car_wv1 test_0330_car_wv2 test_0330_clean_wv1 test_0330_clean_wv2 test_0330_restaurant_wv1 test_0330_restaurant_wv2 test_0330_street_wv1 test_0330_street_wv2 test_0330_train_wv1 test_0330_train_wv2
do
        steps/nnet2/decode.sh --nj 5 exp/DNN/graph data/$i exp/DNN/decode_$i
done

