##!/usr/bin/env bash
export PYHTONPATH=.
#dailydialog
python parlai/scripts/human_dialogues.py  -t dailydialog -ns $4 -ne 10000 -host 160.85.252.221 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/bert_rank/model_bert_rank_final -mf2 tmp/dailydialog/huggingface/model.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5
python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/bert_rank/model_bert_rank_final -mf2 tmp/dailydialog/seq2seq_att/model_s2s_2_512_gru_bi_final -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5
python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/bert_rank/model_bert_rank_final -mf2 tmp/dailydialog/suckybot/model_s2s_2_256_gru_bi.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5

python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/huggingface/model.checkpoint -mf2 tmp/dailydialog/seq2seq_att/model_s2s_2_512_gru_bi_final -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5
python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/huggingface/model.checkpoint -mf2 tmp/dailydialog/suckybot/model_s2s_2_256_gru_bi.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5

python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/seq2seq_att/model_s2s_2_512_gru_bi_final -mf2 tmp/dailydialog/suckybot/model_s2s_2_256_gru_bi.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5

