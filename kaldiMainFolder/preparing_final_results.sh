#!/bin/bash
for i in test_0330_airport_wv1 test_0330_airport_wv2 test_0330_babble_wv1 test_0330_babble_wv2 test_0330_car_wv1 test_0330_car_wv2 test_0330_clean_wv1 test_0330_clean_wv2 test_0330_restaurant_wv1 test_0330_restaurant_wv2 test_0330_street_wv1 test_0330_street_wv2 test_0330_train_wv1 test_0330_train_wv2
do
        echo -e "\n \n for: $i  , the results are as follows : \n"
	cd /home/kunal/kaldiMainFolder-ivec_time_activation-github-final-nfft/exp/DNN
	cd decode_$i
	cat wer_* | grep "WER" | sort -n
done

echo -e "\n                 Coming to best results: "

sum=0
num=0
for i in test_0330_airport_wv1 test_0330_airport_wv2 test_0330_babble_wv1 test_0330_babble_wv2 test_0330_car_wv1 test_0330_car_wv2 test_0330_clean_wv1 test_0330_clean_wv2 test_0330_restaurant_wv1 test_0330_restaurant_wv2 test_0330_street_wv1 test_0330_street_wv2 test_0330_train_wv1 test_0330_train_wv2
do
        echo -e "\nFor: $i  , the best result is :"
        cd /home/kunal/kaldiMainFolder-ivec_time_activation-github-final-nfft/exp/DNN
        cd decode_$i
        cat wer_* | grep "WER" | sort -n | head -1
	num=$( cat wer_* | grep "WER" | sort -n | head -1 | cut -c 6-10)
	sum=$(awk "BEGIN {print $sum + $num; exit}")
done

echo -e " \n The sum is $sum "

printf " \n The avg is %.2f" "$(bc -l <<< "(($sum)/14)")"
 
