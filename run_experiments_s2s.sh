##!/usr/bin/env bash
export PYTHONPATH=. 
datasets=("dailydialog" "personachat" "wizard_of_wikipedia" "moviedialog")

for dataset in "${datasets[@]}"
do
  python -m parlai.scripts.train_model -t $dataset -m seq2seq -mf tmp/"$dataset"/seq2seq/model_s2s_2_512_gru_bi_final -nl 2 -hs 512 -rnn gru -esz 300 -bi True -att none --num-epochs 1 -veps 2.0 -bs 64 -opt adam -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc  -vp 10 --inference greedy -vmt bleu-4 --dict_maxtokens 50000 --dict_minfreq 5
  python -m parlai.scripts.self_chat -t $dataset:selfchat -m seq2seq -mf tmp/"$dataset"/seq2seq/model_s2s_2_512_gru_bi_final --inference nucleus --outfile tmp/"$dataset"/model_s2s_2_512_gru_bi_final.json
done