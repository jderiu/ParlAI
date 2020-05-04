##!/usr/bin/env bash
#personachat
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/personachat/seq2seq/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/convai2/kvmemnn/model -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/convai2/kvmemnn/model -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/convai2/kvmemnn/model -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -mf2 tmp/convai2/kvmemnn/model -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
# dailydialog
python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/bert_rank/model_bert_rank_final -mf2 tmp/dailydialog/seq2seq/model_s2s_2_512_gru_bi_final -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/bert_rank/model_bert_rank_final -mf2 tmp/dailydialog/seq2seq_att/model_s2s_2_512_gru_bi_final -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t dailydialog:selfchat -mf1 tmp/dailydialog/seq2seq/model_s2s_2_512_gru_bi_final -mf2 tmp/dailydialog/seq2seq_att/model_s2s_2_512_gru_bi_final -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
#wizard of wiki
python parlai/scripts/bot_tournament.py -t wizard_of_wikipedia:selfchat -mf1 tmp/wizard_of_wikipedia/bert_rank/model_bert_rank_top4 -mf2 tmp/wizard_of_wikipedia/seq2seq/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t wizard_of_wikipedia:selfchat -mf1 tmp/wizard_of_wikipedia/bert_rank/model_bert_rank_top4 -mf2 tmp/wizard_of_wikipedia/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t wizard_of_wikipedia:selfchat -mf1 tmp/wizard_of_wikipedia/seq2seq/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/wizard_of_wikipedia/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
#empathetic_dialogues
python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/bert_rank/model_bert_rank_final -mf2 tmp/empathetic_dialogues/seq2seq/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/bert_rank/model_bert_rank_final -mf2 tmp/empathetic_dialogues/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t empathetic_dialogues:selfchat -mf1 tmp/empathetic_dialogues/seq2seq/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/empathetic_dialogues/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.62 -port 27017 -user $1 -pw $2



