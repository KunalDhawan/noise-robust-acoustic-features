#!/bin/bash
cp ~/feature_60_1_0_2/train_multi/feats_orig.scp ~/kaldiMainFolder/data/train_multi/feats_orig.scp

for i in airport_wv1 airport_wv2 babble_wv1 babble_wv2 car_wv1 car_wv2 clean_wv1 clean_wv2 restaurant_wv1 restaurant_wv2 street_wv1 street_wv2 train_wv1 train_wv2
do
	cp ~/feature_60_1_0_2/test_0330/$i/feats_orig.scp ~/kaldiMainFolder/data/test_0330_$i/feats_orig.scp
	
done

