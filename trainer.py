import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import (
    MODEL_CLASSES,
    compute_metrics,
    get_intent_labels,
    get_slot_labels,
    get_bslot_labels,
    get_islot_labels,
)

logger = logging.getLogger(__name__)

DEBUG = False


def dec2_infer(b_pred, i_pred, b_prob, i_prob):
    if b_pred == "UNK" and i_pred == "UNK":
        return "O"
    elif b_pred == "UNK":
        return i_pred
    elif i_pred == "UNK":
        return b_pred
    elif b_pred == "O" and i_pred == "O":
        return "O"
    else:
        if b_prob > i_prob:
            return b_pred
        elif b_prob <= i_prob:
            return i_pred   

def dec2_debug(b_slot_logits, i_slot_logits, out_slot_labels_ids, slot_map, bslot_map, islot_map):        
    label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
    preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

    with open('/root/NARBERT/data/snips/test/seq.in', 'r', encoding='utf-8') as f:
        word_list = [line.strip().split(' ') for line in f]

    wrong_file = open('wrong2.csv', 'w', encoding='utf-8')
    debug_file = open('debug2.csv', 'w', encoding='utf-8')
    
    for b in range(out_slot_labels_ids.shape[0]):
        l_ids = []
        for l in range(out_slot_labels_ids.shape[1]):
            if out_slot_labels_ids[b,l] != 0:
                l_ids.append(l)
                label_list[b].append(slot_map[out_slot_labels_ids[b,l]])

                b_prob = b_slot_logits[b,l].max()
                i_prob = i_slot_logits[b,l].max()

                b_pred = bslot_map[b_slot_logits[b,l].argmax()]
                i_pred = islot_map[i_slot_logits[b,l].argmax()]
                # print(out_slot_labels_ids[i, j])
                # print(b_pred, i_pred)
                # print("=========================")
                slot = dec2_infer(b_pred, i_pred, b_prob, i_prob)
                preds_list[b].append(slot)

        if label_list[b] != preds_list[b]:
            wrong_file.write(str(b) + '\n')
            wrong_file.write('word:,' + ','.join(word_list[b]) + '\n')
            wrong_file.write('real:,' + ','.join(label_list[b]) + '\n')
            wrong_file.write('pred:,' + ','.join(preds_list[b]) + '\n')
            wrong_file.write('\n')

            debug_file.write(str(b) + '\n')
            debug_file.write('word:,' + ','.join(word_list[b]) + '\n')
            debug_file.write('real:,' + ','.join(label_list[b]) + '\n')
            debug_file.write('pred:,' + ','.join(preds_list[b]) + '\n')

            for word_id, l in enumerate(l_ids):
                pred_slot = preds_list[b][word_id]
                real_slot = label_list[b][word_id]

                # if pred_slot != real_slot:
                if True:
                    debug_file.write(f'pred word:,{pred_slot},real word:,{real_slot},word:,{word_list[b][word_id]}\n')
                    # debug_file.write(f'pred word:,{pred_slot},real word:,{real_slot} \n')
                    slots = list(slot_map.values())
                    bslots = list(bslot_map.values())
                    islots = list(islot_map.values())

                    reverse_slot_map = {v: k for k, v in slot_map.items()}

                    # logits

                    b_slot_id_list = [reverse_slot_map[slot] for slot in bslots]
                    i_slot_id_list = [reverse_slot_map[slot] for slot in islots]

                    b_logits = ["." for _ in range(len(slots))]
                    i_logits = ["." for _ in range(len(slots))]

                    for id, p in zip(b_slot_id_list, b_slot_logits[b,l].tolist()):
                        b_logits[id] = str(p)
                        
                    for id, p in zip(i_slot_id_list, i_slot_logits[b,l].tolist()):
                        i_logits[id] = str(p)
                    
                    b_slot_probs = torch.tensor(b_slot_logits[b,l]).softmax(dim=-1)
                    i_slot_probs = torch.tensor(i_slot_logits[b,l]).softmax(dim=-1)


                    b_probs = ["." for _ in range(len(slots))]
                    i_probs = ["." for _ in range(len(slots))]

                    for id, p in zip(b_slot_id_list, b_slot_probs.tolist()):
                        b_probs[id] = str(p)
                    
                    for id, p in zip(i_slot_id_list, i_slot_probs.tolist()):
                        i_probs[id] = str(p)
                    
                    debug_file.write("slots:," + ",".join(slots) + "\n")
                    debug_file.write("b logit:," + ",".join(b_logits) + "\n")
                    debug_file.write("i logit:," + ",".join(i_logits) + "\n")
                    debug_file.write("b prob:," + ",".join(b_probs) + "\n")
                    debug_file.write("i prob:," + ",".join(i_probs) + "\n")
                    debug_file.write("eoi,"+ "\n")

            debug_file.write('\n')

    assert 1==0
        
def dec1_debug(slot_logits, out_slot_labels_ids, slot_map):
    label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
    preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

    with open('/root/NARBERT/data/snips/test/seq.in', 'r', encoding='utf-8') as f:
        word_list = [line.strip().split(' ') for line in f]
    
    wrong_file = open('wrong2.csv', 'w', encoding='utf-8')
    debug_file = open('debug2.csv', 'w', encoding='utf-8')

    
    for b in range(out_slot_labels_ids.shape[0]):
        l_ids = []
        for l in range(out_slot_labels_ids.shape[1]):
            if out_slot_labels_ids[b,l] != 0:
                l_ids.append(l)
                label_list[b].append(slot_map[out_slot_labels_ids[b,l]])

                cur_slot = slot_map[slot_logits[b,l].argmax()]
                preds_list[b].append(cur_slot)
                
        if label_list[b] != preds_list[b]:
            
            wrong_file.write(str(b) + '\n')
            wrong_file.write('word:,' + ','.join(word_list[b]) + '\n')
            wrong_file.write('real:,' + ','.join(label_list[b]) + '\n')
            wrong_file.write('pred:,' + ','.join(preds_list[b]) + '\n')
            wrong_file.write('\n')

            debug_file.write(str(b) + '\n')
            debug_file.write('word:,' + ','.join(word_list[b]) + '\n')
            debug_file.write('real:,' + ','.join(label_list[b]) + '\n')
            debug_file.write('pred:,' + ','.join(preds_list[b]) + '\n')

            for word_id, l in enumerate(l_ids):
                pred_slot = preds_list[b][word_id]
                real_slot = label_list[b][word_id]

                # if pred_slot != real_slot:
                if True:
                    debug_file.write(f'pred word:,{pred_slot},real word:,{real_slot},word:,{word_list[b][word_id]}\n')
                    cur_logit = [str(logit) for logit in slot_logits[b,l].tolist()]
                    probs = torch.tensor(slot_logits[b,l]).softmax(dim=-1).tolist()
                    probs = [str(p) for p in probs]

                    all_slots = slot_map.values()

                    debug_file.write("slots:," + ",".join(all_slots) + "\n")
                    debug_file.write("logit:," + ",".join(cur_logit) + "\n")
                    debug_file.write("prob:," + ",".join(probs) + "\n")
                    debug_file.write("eoi,"+ "\n")


            debug_file.write('\n')

    assert 1==0

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args) if args.task in ['atis', 'snips'] else None
        self.slot_label_lst = get_slot_labels(args)
        self.bslot_label_lst = get_bslot_labels(args)
        self.islot_label_lst = get_islot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(
            args.model_name_or_path, finetuning_task=args.task
        )
        self.model = self.model_class.from_pretrained(
            args.model_name_or_path,
            config=self.config,
            args=args,
            intent_label_lst=self.intent_label_lst,
            slot_label_lst=self.slot_label_lst,
        )

        # GPU or CPU
        self.device = (
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps
                // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for epoch_id, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                """
                0: input_ids
                1: attention_mask
                2: token_type_ids
                3: intent_label_ids
                4: slot_labels_ids
                5: bslot_labels_ids
                6: islot_labels_ids
                """
                if self.args.n_dec == 1:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "intent_label_ids": batch[3],
                        "slot_labels_ids": batch[4],
                        "bslot_labels_ids": None,
                        "islot_labels_ids": None,
                    }
                elif self.args.n_dec == 2:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "intent_label_ids": batch[3],
                        "slot_labels_ids": batch[4],
                        "bslot_labels_ids": batch[5],
                        "islot_labels_ids": batch[6],
                    }

                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]

                outputs = self.model(**inputs)
                loss = outputs["loss"]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # if (
                    #     self.args.logging_steps > 0
                    #     and global_step % self.args.logging_steps == 0
                    # ):
                    #     self.evaluate("dev")

                    # if (
                    #     self.args.save_steps > 0
                    #     and global_step % self.args.save_steps == 0
                    # ):
                    #     self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

            self.evaluate("dev")
            self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")
    
        
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        bslot_preds = None
        islot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                if self.args.n_dec == 1:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "intent_label_ids": batch[3],
                        "slot_labels_ids": batch[4],
                        "bslot_labels_ids": None,
                        "islot_labels_ids": None,
                    }
                elif self.args.n_dec == 2:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "intent_label_ids": batch[3],
                        "slot_labels_ids": batch[4],
                        "bslot_labels_ids": batch[5],
                        "islot_labels_ids": batch[6],
                    }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss = outputs["loss"]
                intent_logits = outputs["intent_logits"]
                slot_logits = outputs["slot_logits"]
                bslot_logits = outputs["bslot_logits"]
                islot_logits = outputs["islot_logits"]

                # tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
            else:
                intent_preds = np.append(
                    intent_preds, intent_logits.detach().cpu().numpy(), axis=0
                )
                out_intent_label_ids = np.append(
                    out_intent_label_ids,
                    inputs["intent_label_ids"].detach().cpu().numpy(),
                    axis=0,
                )

            # Slot prediction
            if self.args.n_dec == 1:
                if slot_preds is None:
                    if self.args.use_crf:
                        # decode() in `torchcrf` returns list with best index directly
                        slot_preds = np.array(self.model.crf.decode(slot_logits))
                    else:
                        slot_preds = (
                            slot_logits.detach().cpu().numpy()
                            if slot_logits is not None
                            else None
                        )

                    out_slot_labels_ids = (
                        inputs["slot_labels_ids"].detach().cpu().numpy()
                    )
                else:
                    if self.args.use_crf:
                        slot_preds = np.append(
                            slot_preds,
                            np.array(self.model.crf.decode(slot_logits)),
                            axis=0,
                        )
                    else:
                        if slot_logits is not None:
                            slot_preds = np.append(
                                slot_preds, slot_logits.detach().cpu().numpy(), axis=0
                            )

                    out_slot_labels_ids = np.append(
                        out_slot_labels_ids,
                        inputs["slot_labels_ids"].detach().cpu().numpy(),
                        axis=0,
                    )

            elif self.args.n_dec == 2:
                if bslot_preds is None:
                    bslot_preds = bslot_logits.detach().cpu().numpy()
                    islot_preds = islot_logits.detach().cpu().numpy()
                    out_slot_labels_ids = (
                        inputs["slot_labels_ids"].detach().cpu().numpy()
                    )
                else:
                    bslot_preds = np.append(
                        bslot_preds, bslot_logits.detach().cpu().numpy(), axis=0
                    )

                    islot_preds = np.append(
                        islot_preds, islot_logits.detach().cpu().numpy(), axis=0
                    )

                    out_slot_labels_ids = np.append(
                        out_slot_labels_ids,
                        inputs["slot_labels_ids"].detach().cpu().numpy(),
                        axis=0,
                    )

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}


        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        bslot_label_map = {i: label for i, label in enumerate(self.bslot_label_lst)}
        islot_label_map = {i: label for i, label in enumerate(self.islot_label_lst)}
            

        if DEBUG and mode == "test":
            slot_logits = slot_preds.copy() if slot_preds is not None else None
            bslot_logits = bslot_preds.copy() if bslot_preds is not None else None
            islot_logits = islot_preds.copy() if islot_preds is not None else None
            
            if slot_logits is not None:
                dec1_debug(slot_logits, out_slot_labels_ids, slot_label_map)
            else:
                dec2_debug(bslot_logits, islot_logits, out_slot_labels_ids, slot_label_map, bslot_label_map, islot_label_map)

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        
        if not self.args.use_crf and slot_preds is not None:
            slot_logits = slot_preds.copy()
            slot_preds = np.argmax(slot_preds, axis=2)

        if self.args.n_dec == 2:            
            # bslot_preds[:,:,:3] = float('-inf')
            # islot_preds[:,:,:3] = float('-inf')
            bslot_probs = np.max(bslot_preds, axis=2)
            bslot_preds = np.argmax(bslot_preds, axis=2)
            islot_probs = np.max(islot_preds, axis=2)
            islot_preds = np.argmax(islot_preds, axis=2)

        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    
                    out_slot_label_list[i].append(
                        slot_label_map[out_slot_labels_ids[i][j]]
                    )

                    if slot_preds is not None:
                        slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                        # if slot_preds[i][j] != out_slot_labels_ids[i][j]:
                        #     print(slot_label_map[out_slot_labels_ids[i][j]])
                        #     print(slot_label_map[slot_preds[i][j]])
                        #     for w, l in zip(slot_label_map.values(), slot_logits[i,j].tolist()):
                        #         print(w,l)
                        #     breakpoint()
                    else:                        
                        b_pred = bslot_label_map[bslot_preds[i][j]]
                        i_pred = islot_label_map[islot_preds[i][j]]
                        
                        
                        slot = dec2_infer(b_pred, i_pred, bslot_probs[i][j], islot_probs[i][j])
                        slot_preds_list[i].append(slot)

        with open('slot_preds.txt', 'w', encoding='utf-8') as f:
            for slot_pred in slot_preds_list:
                f.write(' '.join(slot_pred) + '\n')
        
        total_result = compute_metrics(
            intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list
        )
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            print(key, results[key])
            logger.info("  %s = %s", key, str(results[key]))
        print("=====================")

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(
                self.args.model_dir,
                args=self.args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
