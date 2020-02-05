##!/usr/bin/env bash
for nl in 2 4
do
  for hs in 512 1024
  do
    python -m parlai.scripts.train_model -t dailydialog -m seq2seq -mf tmp/dailydialog/seq2seq_att/model_s2s_"$nl"_"$hs"_gru_bi -nl $nl -hs $hs -rnn gru -esz 300 -bi true -att general --num-epochs 1 -veps 2.0 -bs 32 -opt adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.1 --embedding-type fasttext_cc  -vp 10 --inference greedy -vmt bleu-4 --warmup-updates 4000
    python -m parlai.scripts.self_chat -t dailydialog:selfchat -m seq2seq -mf tmp/dailydialog/seq2seq_att/model_s2s_"$nl"_"$hs"_gru_bi
  done
done
