##!/usr/bin/env bash
export PYTHONPATH=.
for nl in 2
do
  for hs in 256 512
  do
      python -m parlai.scripts.train_model -t personachat -m seq2seq -mf tmp/personachat/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi -nl $nl -hs $hs -rnn lstm -esz 300 -bi true -att none --num-epochs 60 -veps 2.0 -bs 64 -opt adam -lr 0.001  --dropout 0.1 --embedding-type fasttext_cc -vp 50 --inference greedy -vmt ppl -sval True 
      python parlai/scripts/self_chat.py  -t personachat:selfchat -m seq2seq -mf tmp/personachat/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi --outfile tmp/personachat/model_s2s_"$nl"_"$hs"_gru_bi.json
  done
done
