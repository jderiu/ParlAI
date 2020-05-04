##!/usr/bin/env bash
#personachat
#python parlai/scripts/human_dialogues.py  -t personachat -ns $3 -ne 10000 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank_20/model_bert_rank_final -mf2 tmp/personachat/bert_rank/model_bert_rank_final -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank_60/model_bert_rank_final -mf2 tmp/personachat/bert_rank_20/model_bert_rank_final -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank_60/model_bert_rank_final -mf2 tmp/personachat/bert_rank/model_bert_rank_final -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_20/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/seq2seq/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_60/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/seq2seq/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_20/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/seq2seq_60/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_att_20/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_att_20/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/seq2seq_att_60/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_att_60/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

