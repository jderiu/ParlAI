##!/usr/bin/env bash
export PYTHONPATH=.
python parlai/scripts/train_model.py -t personachat:self_original20 -m bert_ranker/bi_encoder_ranker -mf tmp/personachat/bert_rank_20/model_bert_rank_final --num-epochs 50 -veps 1.0 -bs 16 --type-optimization top_layer --embedding-type fasttext_cc  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 200 -histsz 3
python parlai/scripts/train_model.py -t personachat:self_original60 -m bert_ranker/bi_encoder_ranker -mf tmp/personachat/bert_rank_60/model_bert_rank_final --num-epochs 50 -veps 1.0 -bs 16 --type-optimization top_layer --embedding-type fasttext_cc  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 200 -histsz 3
