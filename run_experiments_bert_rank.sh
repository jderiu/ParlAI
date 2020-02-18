##!/usr/bin/env bash

datasets=("dailydialog" "personachat" "wizard_of_wikipedia" "cornell_movie")

for dataset in "${datasets[@]}"
do
  python -m parlai.scripts.train_model -t $dataset -m bert_ranker/bi_encoder_ranker -mf tmp/"$dataset"/bert_rank/model_bert_rank_final --num-epochs 50 -veps 1.0 -bs 16 --type-optimization all_encoder_layers --embedding-type fasttext_cc  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 200 -histsz 3
  python -m parlai.scripts.self_chat -t $dataset:selfchat -m bert_ranker/bi_encoder_ranker -mf tmp/"$dataset"/bert_rank/model_bert_rank_final --outfile tmp/"$dataset"/model_bert_rank_final.json
done

python -m parlai.scripts.train_model -t cornell_movie -m bert_ranker/bi_encoder_ranker -mf tmp/cornell_movie/bert_rank/model_bert_rank_final --num-epochs 1 -veps 0.01 -bs 16 --type-optimization all_encoder_layers --embedding-type fasttext_cc  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 200  -histsz 3 --
python parlai/scripts/self_chat.py -t dailydialog:selfchat -m bert_ranker/bi_encoder_ranker -mf tmp/dailydialog/bert_rank/model_bert_rank_final --outfile tmp/dailydialog/model_bert_rank_final.json --fixed_candidates_path tmp/dailydialog/cands.txt

#python -m parlai.scripts.train_model -t cornell_movie -m transformer/ranker -mf tmp/cornell_movie/transformer/model_tr --num-epochs 1 -veps 0.01 -bs 16 --embedding-type fasttext_cc  -vp 10 -vmt bleu-4 --eval-candidates batch --dict_maxtokens 50000 --dict_minfreq 5
#python -m parlai.scripts.self_chat -t cornell_movie:selfchat -m bert_ranker/bi_encoder_ranker -mf tmp/cornell_movie/bert_rank/model_bert_rank_final --outfile tmp/cornell_movie/model_bert_rank_final.json --dict_maxtokens 50000 --dict_minfreq 5

python -m parlai.scripts.train_model -t wizard_of_wikipedia -m bert_ranker/bi_encoder_ranker -mf tmp/wizard_of_wikipedia/bert_rank/model_bert_rank_top4 --num-epochs 50 -veps 1.0 -bs 16 --type-optimization top4_layers --embedding-type fasttext_cc  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 200 -histsz 3  --eval-candidates batch
python -m parlai.scripts.train_model -t dailydialog -m bert_ranker/bi_encoder_ranker -mf tmp/dailydialog/bert_rank/model_bert_rank_final --num-epochs 50 -veps 0.1  -bs 16 --type-optimization top_layer --embedding-type fasttext_cc  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 200 -histsz 3 --eval-candidates batch
python -m parlai.scripts.train_model -t empathetic_dialogues -m bert_ranker/bi_encoder_ranker -mf tmp/empathetic_dialogues/bert_rank/model_bert_rank_final --num-epochs 50 -veps 0.1  -bs 16 --type-optimization top4_layers --embedding-type fasttext_cc  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 200 -histsz 3 --eval-candidates batch
python -m parlai.scripts.train_model -t taskmaster  -m bert_ranker/bi_encoder_ranker -mf tmp/taskmaster/bert_rank/model_bert_rank_final --num-epochs 50 -veps 0.1  -bs 16 --type-optimization top4_layers --embedding-type fasttext_cc  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 200 -histsz 3 --eval-candidates batch
python -m parlai.scripts.train_model -t moviedialog:Task:4 -m bert_ranker/bi_encoder_ranker -mf tmp/moviedialog/bert_rank/model_bert_rank_final --num-epochs 50 -veps 0.01  -bs 16 --type-optimization top_layer --embedding-type fasttext_cc-fixed  -vp 10 --data-parallel True -vmt bleu-4 --text-truncate 300 -histsz 3 --eval-candidates batch --dict_maxtokens 50000 --dict_minfreq 5
