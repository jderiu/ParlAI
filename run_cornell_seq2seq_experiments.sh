##!/usr/bin/env bash
export PYTHONPATH=.
for nl in 1 2
do
  for hs in 256 512
  do
      python -m parlai.scripts.train_model -t cornell_movie -m seq2seq -mf tmp/cornell_movie/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi -nl $nl -hs $hs -rnn gru -esz 300 -bi True -att none --num-epochs 50 -veps 1.0 -bs 32 --optimizer adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 50 --inference greedy -vmt bleu-4 -sval True
      python parlai/scripts/self_chat.py  -t cornell_movie:selfchat -m seq2seq -mf tmp/cornell_movie/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi --outfile tmp/cornell_movie/model_s2s_"$nl"_"$hs"_gru_bi.json
  done
done
