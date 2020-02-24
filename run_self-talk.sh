##!/usr/bin/env bash

datasets=("dailydialog" "personachat" "wizard_of_wikipedia" "cornell_movie")

for dataset in "${datasets[@]}"
do
  python -m parlai.scripts.self_chat -t $dataset:selfchat -m bert_ranker/bi_encoder_ranker -mf tmp/"$dataset"/bert_rank/model_bert_rank_final --outfile tmp/"$dataset"/model_bert_rank_final.json -nd 10 -host 160.85.252.62 -port 27017 -user $1 -pw $2
done

datasets=("personachat" "wizard_of_wikipedia" "cornell_movie")

python -m parlai.scripts.self_chat -t dailydialog:selfchat -m seq2seq -mf tmp/dailydialog/seq2seq/model_s2s_2_512_gru_bi_final --outfile tmp/dailydialog/model_s2s_2_512_gru_bi.json -nd 10 -host 160.85.252.62 -port 27017 -user $1 -pw $2
for dataset in "${datasets[@]}"
do
  python -m parlai.scripts.self_chat -t $dataset:selfchat -m seq2seq -mf tmp/"$dataset"/seq2seq/model_s2s_2_512_gru_bi.checkpoint --outfile tmp/"$dataset"/model_s2s_2_512_gru_bi.json -nd 10 -host 160.85.252.62 -port 27017 -user $1 -pw $2
done

datasets=("personachat" "wizard_of_wikipedia" "cornell_movie")

python -m parlai.scripts.self_chat -t dailydialog:selfchat -m seq2seq -mf tmp/dailydialog/seq2seq_att/model_s2s_2_512_gru_bi_final --outfile tmp/dailydialog/model_s2s_2_512_gru_bi.json -nd 10 -host 160.85.252.62 -port 27017 -user $1 -pw $2
for dataset in "${datasets[@]}"
do
  python -m parlai.scripts.self_chat -t $dataset:selfchat -m seq2seq -mf tmp/"$dataset"/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint --outfile tmp/"$dataset"/model_s2s_att_2_512_gru_bi.json -nd 10 -host 160.85.252.62 -port 27017 -user $1 -pw $2
done




