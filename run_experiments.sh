##!/usr/bin/env bash
python -m parlai.scripts.train_model -t personachat -m transformer/ranker -mf /tmp/personachat/model_tr1_300_600_4 --n-layers 1 --embedding-size 300 --ffn-size 600 --n-heads 4 --num-epochs 50 -veps 0.75 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc --candidates batch -vp 20
python -m parlai.scripts.train_model -t personachat -m transformer/ranker -mf /tmp/personachat/model_tr2_300_300_2 --n-layers 1 --embedding-size 300 --ffn-size 300 --n-heads 2 --num-epochs 50 -veps 0.75 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc --candidates batch -vp 20

python -m parlai.scripts.train_model -t dailydialog -m transformer/ranker -mf /tmp/dailydialog/model_tr61_300_600_4 --n-layers 1 --embedding-size 300 --ffn-size 600 --n-heads 4 --num-epochs 50 -veps 0.75 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc --candidates batch -vp 20  --eval-candidates batch
python -m parlai.scripts.train_model -t dailydialog -m transformer/ranker -mf /tmp/dailydialog/model_tr61_300_600_4 --n-layers 2 --embedding-size 300 --ffn-size 300 --n-heads 2 --num-epochs 50 -veps 0.75 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc --candidates batch -vp 20  --eval-candidates batch

python -m parlai.scripts.train_model -t personachat -m seq2seq -mf /tmp/dailydialog/model_s2s2_512_none_bi -nl 2 -hs 512 -esz 300 -bi True -att none --num-epochs 50 -veps 2.0 -bs 64 -opt adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 20
python -m parlai.scripts.train_model -t personachat -m seq2seq -mf /tmp/dailydialog/model_s2s2_512_general_bi -nl 2 -hs 512 -esz 300 -bi True -att general --num-epochs 50 -veps 2.0 -bs 64 -opt adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 20
python -m parlai.scripts.train_model -t personachat -m seq2seq -mf /tmp/dailydialog/model_s2s2_512_general_bi_all -lt all -nl 2 -hs 512 -esz 300 -bi True -att general --num-epochs 50 -veps 2.0 -bs 64 -opt adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 20 