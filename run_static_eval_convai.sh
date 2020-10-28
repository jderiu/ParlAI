##!/usr/bin/env bash
export PYHTONPATH=.

python parlai/scripts/eval_model.py  -t personachat -mf "zoo:blender/blender_90M/model" --save-world-logs True -rf "tmp/personachat/logs/blender_report" -ne 10
python parlai/scripts/eval_model.py  -t personachat -mf "tmp/personachat/bert_rank/model_bert_rank_final" --save-world-logs True -rf "tmp/personachat/logs/bert_rank_report" -ne 10
python parlai/scripts/eval_model.py  -t personachat -mf "tmp/convai2/lost_in_conversation/last_checkpoint" --save-world-logs True -rf "tmp/personachat/logs/lost_in_conversation_report" -ne 10
python parlai/scripts/eval_model.py  -t personachat -mf "tmp/personachat/seq2seq_att/model_s2s_2_512_gru_bi.checkpoint" --save-world-logs True -rf "tmp/personachat/logs/seq2seq_att_report" -ne 10
python parlai/scripts/eval_model.py  -t personachat -mf "tmp/personachat/suckybot/model_s2s_2_256_gru_bi" --save-world-logs True -rf "tmp/personachat/logs/suckybot_report" -ne 10
python parlai/scripts/eval_model.py  -t personachat -mf "tmp/convai2/kvmemnn/model" --save-world-logs True -rf "tmp/personachat/logs/kvmemnn_report" -ne 10
python parlai/scripts/eval_model.py  -t personachat -mf "tmp/convai2/huggingface/model" --save-world-logs True -rf "tmp/personachat/logs/huggingface_report" -ne 10