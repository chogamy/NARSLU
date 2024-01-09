batchs=(32)
epochs=(1) # 1 5 10 20 30 40
task=(ontonotes5)
n_decs=(1) # 1 2
model_types=(bert)

for epoch in ${epochs[@]}
do
    for batch in ${batchs[@]}
    do
        for n_dec in ${n_decs[@]}
        do
            for model_type in ${model_types[@]}
            do 
                CUDA_VISIBLE_DEVICES=0 python main.py --task ${task} \
                        --model_type ${model_type} \
                        --model_dir ${task}_${model_type}_${n_dec}dec_${epoch}_${batch} \
                        --do_train --do_eval \
                        --max_seq_len 512 \
                        --n_dec ${n_dec} \
                        --num_train_epochs $epoch \
                        --train_batch_size $batch
            done
        done
    done
done
