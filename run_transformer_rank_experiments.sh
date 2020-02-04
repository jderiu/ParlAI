##!/usr/bin/env bash
for nl in 1 2 4
do
  for nhead in 2 4 6 10
  do
    for hid in 120 300 600
    do
      for opt in "adam" "sgd"
      do
        python -m parlai.scripts.train_model -t dailydialog -m transformer/ranker -mf tmp/dailydialog/transformer/model_tr_"$nl"_"$nhead"_"$hid"_"$opt" -nl $nl -hid $hid -n-heads $nhead -opt $opt -esz 300 --num-epochs 50 -veps 2.0 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 10 --candidates batch --eval-candidates batch
      done
    done
  done
done
