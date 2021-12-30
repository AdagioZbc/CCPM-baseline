CUDA_VISIBLE_DEVICES=0 python -u main_notrainer_mc_2.py \
          --model_name_or_path hfl/chinese-roberta-wwm-ext \
          --do_train --train_file data/train.jsonl \
          --do_eval  --validation_file data/valid.jsonl \
          --learning_rate 5e-5 \
          --num_train_epochs 3 \
          --num_warmup_steps 400 \
          --output_dir mlm_filter \
          --per_device_eval_batch_size=16 \
          --per_device_train_batch_size=16