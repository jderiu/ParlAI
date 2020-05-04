##!/usr/bin/env bash
export PYHTONPATH=.
#personachat
python parlai/scripts/human_dialogues.py  -t personachat -ns $4 -ne 10000 -host 160.85.252.221 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/convai2/lost_in_conversation/last_checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/convai2/kvmemnn/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/bert_rank/model_bert_rank_final -mf2 tmp/convai2/huggingface/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/convai2/lost_in_conversation/last_checkpoint -mf2 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/convai2/lost_in_conversation/last_checkpoint -mf2 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/convai2/lost_in_conversation/last_checkpoint -mf2 tmp/convai2/kvmemnn/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/convai2/lost_in_conversation/last_checkpoint -mf2 tmp/convai2/huggingface/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/convai2/kvmemnn/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint -mf2 tmp/convai2/huggingface/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -mf2 tmp/convai2/kvmemnn/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/personachat/suckybot/model_s2s_2_256_gru_bi -mf2 tmp/convai2/huggingface/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2

python parlai/scripts/bot_tournament.py -t personachat:selfchat -mf1 tmp/convai2/kvmemnn/model -mf2 tmp/convai2/huggingface/model -nd $3 -host 160.85.252.221 -port 27017 -user $1 -pw $2
