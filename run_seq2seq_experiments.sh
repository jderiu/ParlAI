##!/usr/bin/env bash
export PYTHONPATH=.
for nl in 2 4
do
  for hs in 256 512
  do
      python -m parlai.scripts.train_model -t dailydialog -m seq2seq -mf tmp/dailydialog/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi -nl $nl -hs $hs -rnn gru -esz 300 -bi True -att none --num-epochs 50 -veps 2.0 -bs 64 -opt adam -lr 0.001 --warmup-updates 4000 --dropout 0.1 --embedding-type fasttext_cc  -vp 10 --inference greedy -vmt bleu-4
      python parlai/scripts/self_chat.py  -t dailydialog:selfchat -m seq2seq -mf tmp/dailydialog/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi --outfile tmp/dailydialog/model_s2s_"$nl"_"$hs"_gru_bi.json
  done
done
