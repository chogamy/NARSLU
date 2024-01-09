import os

from datasets import load_dataset  

DIR = os.path.dirname(os.path.abspath(__file__))

train_dataset = load_dataset('tner/ontonotes5', split='train')
val_dataset = load_dataset('tner/ontonotes5', split='validation')
test_dataset = load_dataset('tner/ontonotes5', split='test')

slot_to_id = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12, 
    "B-PERCENT": 13,
    "I-PERCENT": 14, 
    "B-ORDINAL": 15, 
    "B-MONEY": 16, 
    "I-MONEY": 17, 
    "B-WORK_OF_ART": 18, 
    "I-WORK_OF_ART": 19, 
    "B-FAC": 20, 
    "B-TIME": 21, 
    "I-CARDINAL": 22, 
    "B-LOC": 23, 
    "B-QUANTITY": 24, 
    "I-QUANTITY": 25, 
    "I-NORP": 26, 
    "I-LOC": 27, 
    "B-PRODUCT": 28, 
    "I-TIME": 29, 
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36
}

print(train_dataset)
print(val_dataset)
print(test_dataset)

slot_labels = ['PAD', 'UNK', 'MASK'] + list(slot_to_id.keys())
with open(os.path.join(DIR, 'slot_label.txt'), 'w') as f:
    for label in slot_labels:
        f.write(label + '\n')

id_to_slot = {v: k for k, v in slot_to_id.items()}

# assertion
for sent, label in zip(train_dataset['tokens'], train_dataset['tags']):
    assert len(sent) == len(label)

for sent, label in zip(val_dataset['tokens'], val_dataset['tags']):
    assert len(sent) == len(label)

for sent, label in zip(test_dataset['tokens'], test_dataset['tags']):
    assert len(sent) == len(label)


###################
# Save file #######
###################
# train
with open(os.path.join(DIR, 'train', 'seq.in'), 'w') as f:
    for sent in train_dataset['tokens']:
        f.write(" ".join(sent) + '\n')

with open(os.path.join(DIR, 'train', 'seq.out'), 'w') as f:
    for sent in train_dataset['tags']:
        sent = [id_to_slot[id] for id in sent]
        f.write(" ".join(sent) + '\n')

# valid
with open(os.path.join(DIR, 'dev', 'seq.in'), 'w') as f:
    for sent in val_dataset['tokens']:
        f.write(" ".join(sent) + '\n')

with open(os.path.join(DIR, 'dev', 'seq.out'), 'w') as f:
    for sent in val_dataset['tags']:
        sent = [id_to_slot[id] for id in sent]
        f.write(" ".join(sent) + '\n')

# test
with open(os.path.join(DIR, 'test', 'seq.in'), 'w') as f:
    for sent in test_dataset['tokens']:
        f.write(" ".join(sent) + '\n')

with open(os.path.join(DIR, 'test', 'seq.out'), 'w') as f:
    for sent in test_dataset['tags']:
        sent = [id_to_slot[id] for id in sent]
        f.write(" ".join(sent) + '\n')