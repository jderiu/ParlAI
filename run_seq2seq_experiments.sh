##!/usr/bin/env bash
for nl in 1 2
do
  for hs in 128 256 512
  do
    for rnn in "gru" "lstm"
    do
      echo "model_s2s_$nl_$hs_$rnn_bi"
      #python -m parlai.scripts.train_model -t dailydialog -m seq2seq -mf /tmp/dailydialog/model_s2s_{$nl}_512_none_bi -nl 2 -hs 512 -esz 300 -bi True -att none --num-epochs 50 -veps 2.0 -bs 64 -opt adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 20
    done
  done
done
