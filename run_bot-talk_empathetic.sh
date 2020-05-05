##!/usr/bin/env bash
export PYHTONPATH=.
#empathetic_dialogues
python parlai/scripts/human_dialogues.py  -t empathetic_dialogues -ns $4 -ne 10000 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5

python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/bert_rank/model_bert_rank_final -mf2 tmp/empathetic_dialogues/huggingface/model.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5
python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/bert_rank/model_bert_rank_final -mf2 tmp/empathetic_dialogues/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5
python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/bert_rank/model_bert_rank_final -mf2 tmp/empathetic_dialogues/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5

python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/huggingface/model -mf2 tmp/empathetic_dialogues/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5
python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/huggingface/model -mf2 tmp/empathetic_dialogues/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5

python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/empathetic_dialogues/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2 -col $5


