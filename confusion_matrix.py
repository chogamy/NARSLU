import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import ListedColormap

import numpy as np

with open('/root/NARBERT/data/snips/test/seq.out', 'r', encoding='utf-8') as f:
    real_list = [line.strip().split(' ') for line in f]

with open('/root/NARBERT/data/snips/slot_label.txt', 'r', encoding='utf-8') as f:
    all_slot = [line.strip() for line in f]


with open('/root/NARBERT/slot_preds.txt', 'r', encoding='utf-8') as f:
    pred_list = [line.strip().split(' ') for line in f]


def cfm(pred_list, real_list, slots):
    cm = np.zeros((len(slots), len(slots)), dtype=np.int32)
    slot_to_id = {s: i for i, s in enumerate(slots)}

    # flatten
    pred_list = [slot_to_id[p] for pred in pred_list for p in pred]
    real_list = [slot_to_id[p] for pred in real_list for p in pred]

    
    cmap = plt.cm.get_cmap('YlOrRd')  

    colors = ['white'] + [cmap(i) for i in range(1, cmap.N)]

    cmap_custom = ListedColormap(colors)

    # cm = confusion_matrix(real_list, pred_list)
    for real, pred in zip(real_list, pred_list):
        cm[real][pred] += 1

    for id in slot_to_id.values():
        cm[id, id] = 0
    
    wrong_per_slot = cm.sum(axis=1)
    for slot, wrong_count in zip(slots, wrong_per_slot):
        print(slot, wrong_count)
    
    # assert 1==0

    # cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # fig, ax = plt.subplots(figsize=(50,50))
    # sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=slots, yticklabels=slots) # slots is to large to show
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.savefig('cmn.png')

    fig, ax = plt.subplots(figsize=(30,30))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=slots, yticklabels=slots, cmap=cmap_custom, linewidths=0.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.savefig('cm.png')


if __name__ == '__main__':
    cfm(pred_list, real_list, all_slot)
