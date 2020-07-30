#!/bin/bash

dataset=$1
numbasis=$2
hsparsity=$3

featdir="/easyshare/kunal/kd_ivector_approach_final/kd_features_"
featdir+="$2"
featdir+="_1_0_"
featdir+="$3"
outdir="/home/kunal/feature_"
outdir+="$2"
outdir+="_1_0_"
outdir+="$3"

if [ "${dataset}" == "train_multi" ]
then
	mkdir -p ${outdir}/${dataset}
	if [ -e ${outdir}/${dataset}/feats_orig.scp ]
	then
		rm ${outdir}/${dataset}/feats_orig.scp
	fi

	# Convert txt to ark (text format) and create an scp file
	for subj_id in $(ls ${featdir}/${dataset})
	do
		for f in $(ls ${featdir}/${dataset}/${subj_id})
		do
			utt_id=${f%.txt}
			#utt_id=${f%.pncc}
			cat ${featdir}/${dataset}/${subj_id}/${f} | sed '1s/^/[ \n/' | sed '$s/$/ ]/' > ${featdir}/${dataset}/${subj_id}/${utt_id}.ark
			echo "${utt_id} ${featdir}/${dataset}/${subj_id}/${utt_id}.ark" >> ${outdir}/${dataset}/feats_orig.scp
		done
	done
elif [ "${dataset}" == "dev_0330" ]
then
	mkdir -p ${outdir}/${dataset}
	if [ -e ${outdir}/${dataset}/feats_orig.scp ]
	then
		rm ${outdir}/${dataset}/feats_orig.scp
	fi

	count=0
	for channel in wv1 wv2
	do
		for noise in clean car babble restaurant street airport train
		do
			((count+=1))
			count_hex=$(printf "%x\n" ${count})

			for subj_id in $(ls ${featdir}/${dataset}/${noise}_${channel})
			do
				for f in $(ls ${featdir}/${dataset}/${noise}_${channel}/${subj_id})
				do
					#utt_id=${f%.txt}
					utt_id=${f%.pncc}
					cat ${featdir}/${dataset}/${noise}_${channel}/${subj_id}/${f} | sed '1s/^/[ \n/' | sed '$s/$/ ]/' > ${featdir}/${dataset}/${noise}_${channel}/${subj_id}/${utt_id}.ark
					echo "${utt_id}${count_hex} ${featdir}/${dataset}/${noise}_${channel}/${subj_id}/${utt_id}.ark" >> ${outdir}/${dataset}/feats_orig.scp
				done
			done
		done
	done

	cat ${outdir}/${dataset}/feats_orig.scp | sort -k 1 > ${outdir}/${dataset}/feats_sorted.scp
	mv ${outdir}/${dataset}/feats_sorted.scp ${outdir}/${dataset}/feats_orig.scp
elif [ "${dataset}" == "test_0330" ]
then
	for channel in wv1 wv2
	do
		for noise in clean car babble restaurant street airport train
		do
			mkdir -p ${outdir}/${dataset}/${noise}_${channel}
			if [ -e ${outdir}/${dataset}/${noise}_${channel}/feats_orig.scp ]
			then
				rm ${outdir}/${dataset}/${noise}_${channel}/feats_orig.scp
			fi
			for subj_id in $(ls ${featdir}/${dataset}/${noise}_${channel})
			do
				for f in $(ls ${featdir}/${dataset}/${noise}_${channel}/${subj_id})
				do
					utt_id=${f%.txt}
					#utt_id=${f%.pncc}
					cat ${featdir}/${dataset}/${noise}_${channel}/${subj_id}/${f} | sed '1s/^/[ \n/' | sed '$s/$/ ]/' > ${featdir}/${dataset}/${noise}_${channel}/${subj_id}/${utt_id}.ark
					echo "${utt_id} ${featdir}/${dataset}/${noise}_${channel}/${subj_id}/${utt_id}.ark" >> ${outdir}/${dataset}/${noise}_${channel}/feats_orig.scp
				done
			done
		done
	done
fi
