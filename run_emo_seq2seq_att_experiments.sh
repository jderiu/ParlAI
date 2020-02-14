##!/usr/bin/env bash
export PYTHONPATH=.
for nl in 1 2
do
  for hs in 256 512
  do
      python -m parlai.scripts.train_model -t empathetic_dialogues -m seq2seq -mf tmp/empathetic_dialogues/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi -nl $nl -hs $hs -rnn gru -esz 300 -bi True -att general --num-epochs 50 -veps 1.0 -bs 32 --optimizer adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 50 --inference greedy -vmt bleu-4 -sval True
      python parlai/scripts/self_chat.py  -t cornell_movie:selfchat -m seq2seq -mf tmp/empathetic_dialogues/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi --outfile tmp/empathetic_dialogues/model_s2s_"$nl"_"$hs"_gru_bi.json
  done
done
