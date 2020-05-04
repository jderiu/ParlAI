##!/usr/bin/env bash
export PYTHONPATH=.
python parlai/scripts/train_model.py -t personachat:self_original20 -m seq2seq -mf tmp/personachat/seq2seq_20/model_s2s_2_512_gru_bi -nl 2 -hs 512 -rnn gru -esz 300 -bi true -att none --num-epochs 40 -veps 2.0 -bs 64 -opt adam -lr 0.001  --dropout 0.1 --embedding-type fasttext_cc -vp 50 --inference greedy -vmt ppl -sval True
python parlai/scripts/train_model.py -t personachat:self_original60 -m seq2seq -mf tmp/personachat/seq2seq_60/model_s2s_2_512_gru_bi -nl 2 -hs 512 -rnn gru -esz 300 -bi true -att none --num-epochs 40 -veps 2.0 -bs 64 -opt adam -lr 0.001  --dropout 0.1 --embedding-type fasttext_cc -vp 50 --inference greedy -vmt ppl -sval True

