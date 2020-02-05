##!/usr/bin/env bash
for nl in 2 4
do
  for nhead in 2 4
  do
    for hid in 300 600
    do
      python -m parlai.scripts.train_model -t dailydialog -m transformer/generator -mf tmp/dailydialog/transformer_generator/model_tr_"$nl"_"$nhead"_"$hid" -nl $nl -hid $hid --n-heads $nhead -esz 300 --num-epochs 50 -veps 2.0 -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 50 -veps 1 --embedding-type fasttext_cc  -vp 10
      python parlai/scripts/self_chat.py  -t dailydialog:selfchat -m transformer/generator -mf tmp/dailydialog/transformer_generator/model_tr_"$nl"_"$nhead"_"$hid" --outfile tmp/dailydialog/model_tr_gen_"$nl"_"$nhead"_"$hid".json
    done
  done
done

