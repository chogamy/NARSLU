# batchs=(32)
# epochs=(20) #  20: bert, 30: 나머지
# task=(snips)
# n_decs=(1) # 1 2
# model_types=(bert)

# for epoch in ${epochs[@]}
# do
#     for batch in ${batchs[@]}
#     do
#         for n_dec in ${n_decs[@]}
#         do
#             for model_type in ${model_types[@]}
#             do 
#                 CUDA_VISIBLE_DEVICES=0 python main.py --task ${task} \
#                         --model_type ${model_type} \
#                         --model_dir ${task}_${model_type}_${n_dec}dec_${epoch}_${batch} \
#                         --do_eval \
#                         --max_seq_len 50 \
#                         --n_dec ${n_dec} \
#                         --num_train_epochs $epoch \
#                         --train_batch_size $batch
#             done
#         done
#     done
# done


batchs=(32)
epochs=(40) # 1 5 10 20 30 40
task=(atis)
n_decs=(2) # 1 2
model_types=(bert)
slot_loss_coefs=(1.2) # 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0

for epoch in ${epochs[@]}
do
    for batch in ${batchs[@]}
    do
        for n_dec in ${n_decs[@]}
        do
            for model_type in ${model_types[@]}
            do 
                for slot_loss_coef in ${slot_loss_coefs[@]} 
                do
                    CUDA_VISIBLE_DEVICES=0 python main.py --task ${task} \
                            --model_type ${model_type} \
                            --model_dir ${task}_${model_type}_${n_dec}dec_${epoch}_${batch}_slotloss_${slot_loss_coef} \
                            --slot_loss_coef ${slot_loss_coef} \
                            --do_eval \
                            --max_seq_len 60 \
                            --n_dec ${n_dec} \
                            --num_train_epochs $epoch \
                            --train_batch_size $batch
                done
            done
        done
    done
done
