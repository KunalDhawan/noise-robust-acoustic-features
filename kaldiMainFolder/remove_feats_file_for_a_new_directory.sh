#!/bin/bash
for i in train_multi test_0330_airport_wv1 test_0330_airport_wv2 test_0330_babble_wv1 test_0330_babble_wv2 test_0330_car_wv1 test_0330_car_wv2 test_0330_clean_wv1 test_0330_clean_wv2 test_0330_restaurant_wv1 test_0330_restaurant_wv2 test_0330_street_wv1 test_0330_street_wv2 test_0330_train_wv1 test_0330_train_wv2
do
#notice: removed train_multi for decode purpose 
	#steps/compute_cmvn_stats.sh data/$i exp/make_mfcc/$i mfcc
	#utils/validate_data_dir.sh data/$i
	#steps/decode.sh --nj 5 exp/mono/graph data/$i exp/mono/decode_$i
	#steps/nnet2/decode.sh --nj 5 exp/DNN/graph data/$i exp/DNN/decode_$i
	cd /home/kunal/kaldiMainFolder-60_1_0_2/data/$i
	rm cmvn.scp  feats_orig.scp  feats.scp
done
