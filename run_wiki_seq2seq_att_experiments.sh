##!/usr/bin/env bash
export PYTHONPATH=.
for nl in 1 2
do
  for hs in 256 512
  do
      python -m parlai.scripts.train_model -t wizard_of_wikipedia -m seq2seq -mf tmp/wizard_of_wikipedia/seq2seq_att/model_s2s_"$nl"_"$hs"_gru_bi -nl $nl -hs $hs -rnn gru -esz 300 -bi True -att general --num-epochs 60 -veps 2.0 -bs 64 --optimizer adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 50 --inference greedy -vmt bleu-4 -sval True
      python parlai/scripts/self_chat.py  -t wizard_of_wikipedia:selfchat -m seq2seq -mf tmp/wizard_of_wikipedia/seq2seq_att/model_s2s_"$nl"_"$hs"_gru_bi --outfile tmp/wizard_of_wikipedia/model_s2s_"$nl"_"$hs"_gru_bi_att.json
  done
done
