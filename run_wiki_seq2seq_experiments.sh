##!/usr/bin/env bash
export PYTHONPATH=.
for nl in 1 2 4
do
  for hs in 128 256 512 1024
  do
      python -m parlai.scripts.train_model -t personachat -m seq2seq -mf tmp/personachat/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi -nl $nl -hs $hs -rnn gru -esz 300 -bi True -att none --num-epochs 60 -veps 2.0 -bs 32 --optimizer adam --lr-scheduler invsqrt --invsqrt-lr-decay-gamma 10 -lr 0.005 --dropout 0.3 --warmup-updates 4000 --embedding-type fasttext_cc  -vp 50 --inference greedy -vmt bleu-4
      python parlai/scripts/self_chat.py  -t personachat:selfchat -m seq2seq -mf tmp/personachat/seq2seq/model_s2s_"$nl"_"$hs"_gru_bi --outfile tmp/personachat/model_s2s_"$nl"_"$hs"_gru_bi.json
  done
done

python -m parlai.scripts.train_model -t personachat -m seq2seq -mf tmp/personachat/seq2seq/model_s2s_test -nl 2 -hs 256 -rnn lstm -esz 256 --lookuptable enc_dec -bi True -att none --num-epochs 60 -veps 2.0 -bs 32 --optimizer adadelta -lr 0.1 --gradient-clip=0.1 --dropout 0.1 --embedding-type fasttext_cc  -vp 50 --inference greedy -vmt ppl
