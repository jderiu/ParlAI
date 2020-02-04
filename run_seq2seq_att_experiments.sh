##!/usr/bin/env bash
for nl in 1 2
do
  for hs in 128 256 512
  do
    for rnn in "gru" "lstm"
    do
      python -m parlai.scripts.train_model -t dailydialog -m seq2seq -mf tmp/dailydialog/seq2seq/model_s2s_"$nl"_"$hs"_"$rnn"_bi -nl $nl -hs $hs -rnn $rnn -esz 300 -bi True -att general --num-epochs 50 -veps 2.0 -bs 64 -opt adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 10 --inference greedy -vmt ppl
    done
  done
done
