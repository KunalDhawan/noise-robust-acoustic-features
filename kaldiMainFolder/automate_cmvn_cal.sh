#!/bin/bash
for i in train_multi test_0330_airport_wv1 test_0330_airport_wv2 test_0330_babble_wv1 test_0330_babble_wv2 test_0330_car_wv1 test_0330_car_wv2 test_0330_clean_wv1 test_0330_clean_wv2 test_0330_restaurant_wv1 test_0330_restaurant_wv2 test_0330_street_wv1 test_0330_street_wv2 test_0330_train_wv1 test_0330_train_wv2
do
        steps/compute_cmvn_stats.sh data/$i exp/make_extracted/$i adapted
done