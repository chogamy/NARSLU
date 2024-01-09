from ordered_set import OrderedSet

EXP_PATH = '/root/NARBERT/debug1.csv'
AMA_PATH = '/root/NARBERT/debug2.csv'

# COL = ['word: ', 'pred: ', 'real: ']

exp_wrong_set = {}
exp_wrong_set['pred'] = {}
exp_wrong_set['real'] = {}
ama_wrong_set = {}
ama_wrong_set['pred'] = {}
ama_wrong_set['real'] = {}

def find_wrong_ids_list(path):
    wrong_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        wrong_ids = []
        for line in f:
                        
            line = line.strip()

            if line == '':
                continue
            else:
                if len(line.split(",")) == 1:
                    wrong_ids.append(int(line))
                    current_id = int(line)
                    wrong_dict[current_id] = []
                else:
                    # key, text = line.split(",", 1)
                    
                    wrong_dict[current_id].append(line)

    return wrong_ids, wrong_dict

def dict_sort(file_name, ids_list, exp_dict, ama_dict):
    with open(f'{file_name}.csv', 'w', encoding='utf-8') as f:
        for id in ids_list:
            f.write(str(id) + '\n')

            if file_name == 'exp_and_ama':
                exp_items = []
                exp_item = []

        
                for item in exp_dict[id]:
                    key, text = item.split(",", 1)

                    if key == 'word:' or key == 'real:' or key =='pred:':
                        f.write(f'{key},{text}\n')
                    else:
                        exp_item.append(item)
                        if key == 'pred word:':
                            _, pred_slot, _, real_slot, _, word = item.strip().split(",")
                            if word not in exp_wrong_set['pred']:
                                exp_wrong_set['pred'][word] = {}
                            
                            if word not in exp_wrong_set['real']:
                                exp_wrong_set['real'][word] = {}
                                
                            if pred_slot not in exp_wrong_set['pred'][word]:
                                exp_wrong_set['pred'][word][pred_slot] = 0
                            
                            if real_slot not in exp_wrong_set['real'][word]:
                                exp_wrong_set['real'][word][real_slot] = 0
                                
                            exp_wrong_set['pred'][word][pred_slot] += 1
                            exp_wrong_set['real'][word][real_slot] += 1


                    if key=='eoi':
                        exp_items.append(exp_item)
                        exp_item = []

                ama_items = []
                ama_item = []
                for item in ama_dict[id]:
                    key, text = item.split(",", 1)

                    if key == 'word:' or key == 'real:' or key =='pred:':
                        if key =='pred:':
                            f.write(f'pred2:,{text}\n')
                    else:
                        ama_item.append(item)
                        if key == 'pred word:':
                            _, pred_slot, _, real_slot, _, word = item.strip().split(",")
                            if word not in ama_wrong_set['pred']:
                                ama_wrong_set['pred'][word] = {}
                            
                            if word not in ama_wrong_set['real']:
                                ama_wrong_set['real'][word] = {}
                                
                            if pred_slot not in ama_wrong_set['pred'][word]:
                                ama_wrong_set['pred'][word][pred_slot] = 0
                            
                            if real_slot not in ama_wrong_set['real'][word]:
                                ama_wrong_set['real'][word][real_slot] = 0
                                
                            ama_wrong_set['pred'][word][pred_slot] += 1
                            ama_wrong_set['real'][word][real_slot] += 1
                    
                    if key=='eoi':
                        ama_items.append(ama_item)
                        ama_item = []
                
                assert len(ama_items) == len(exp_items), f'{len(ama_items)} {len(exp_items)}'

                for exp_item, ama_item in zip(exp_items, ama_items):
                    _, exp_pred, _, exp_real, _, _ = exp_item.pop(0).strip().split(",")
                    _, ama_pred, _, ama_real, _, _ = ama_item.pop(0).strip().split(",")

                    exp_item.pop()
                    ama_item.pop()

                    assert exp_real == ama_real

                    real = exp_real

                    if exp_pred != real and ama_pred != real:
                        f.write(f'pred word1:,{exp_pred},real,{real}\n')
                        for line in exp_item:
                            f.write(f'{line}\n')
                        
                        f.write(f'------------------\n')
                        
                        f.write(f'pred word2:,{ama_pred},real,{real}\n')
                        for line in ama_item:
                            f.write(f'{line}\n')
                        
                        f.write(f'------------------\n')

                        f.write(f'====================\n')
                    
                    if exp_pred != real and ama_pred == real:
                        f.write(f'only 1\n')
                        f.write(f'pred word1:,{exp_pred},real,{real}\n')
                        for line in exp_item:
                            f.write(f'{line}\n')
                        f.write(f'------------------\n')

                        f.write(f'====================\n')
                    
                    if exp_pred == real and ama_pred != real:
                        f.write(f'only 2\n')
                        f.write(f'pred word2:,{ama_pred},real,{real}\n')
                        for line in ama_item:
                            f.write(f'{line}\n')

                        f.write(f'------------------\n')

                        f.write(f'====================\n')

                        
            elif file_name == 'only_exp':
                exp_items = []
                exp_item = []

                for item in exp_dict[id]:
                    key, text = item.split(",", 1)

                    if key == 'word:' or key == 'real:' or key =='pred:':
                        f.write(f'{key},{text}\n')
                    else:
                        exp_item.append(item)
                        if key == 'pred word:':
                            _, pred_slot, _, real_slot, _, word = item.strip().split(",")
                            if word not in exp_wrong_set['pred']:
                                exp_wrong_set['pred'][word] = {}
                            
                            if word not in exp_wrong_set['real']:
                                exp_wrong_set['real'][word] = {}
                                
                            if pred_slot not in exp_wrong_set['pred'][word]:
                                exp_wrong_set['pred'][word][pred_slot] = 0
                            
                            if real_slot not in exp_wrong_set['real'][word]:
                                exp_wrong_set['real'][word][real_slot] = 0
                                
                            exp_wrong_set['pred'][word][pred_slot] += 1
                            exp_wrong_set['real'][word][real_slot] += 1
                    
                    if key=='eoi':
                        exp_items.append(exp_item)
                        exp_item = []
                
                for exp_item in exp_items:
                    _, exp_pred, _, exp_real, _, _ = exp_item.pop(0).strip().split(",")
                    exp_item.pop()
                    real = exp_real

                    if exp_pred != real:
                        f.write(f'pred word1:,{exp_pred},real,{real}\n')
                        for line in exp_item:
                            f.write(f'{line}\n')
                        
                        f.write(f'------------------\n')
                        

                        f.write(f'====================\n')
                    
                
            elif file_name == 'only_ama':
                ama_items = []
                ama_item = []

                for item in ama_dict[id]:
                    key, text = item.split(",", 1)

                    if key == 'word:' or key == 'real:' or key =='pred:':
                        f.write(f'{key},{text}\n')
                    else:
                        ama_item.append(item)
                    
                    if key=='eoi':
                        ama_items.append(ama_item)
                        ama_item = []
                
                for ama_item in ama_items:
                    _, ama_pred, _, ama_real, _, _ = ama_item.pop(0).strip().split(",")
                    ama_item.pop()
                    real = ama_real

                    if ama_pred != real:
                        f.write(f'pred word2:,{ama_pred},real,{real}\n')
                        for line in ama_item:
                            f.write(f'{line}\n')
                        
                        f.write(f'------------------\n')

                        f.write(f'====================\n')
                
            else:
                raise ValueError
    
            f.write('\n')
    
if __name__ == '__main__':
    data_ids = [i for i in range(700)]

    exp_ids, exp_dict = find_wrong_ids_list(EXP_PATH)
    ama_ids, ama_dict = find_wrong_ids_list(AMA_PATH)


    exp_ids = OrderedSet(exp_ids)
    ama_ids = OrderedSet(ama_ids)

    exp_and_ama = exp_ids & ama_ids
    only_exp = exp_ids - ama_ids
    only_ama = ama_ids - exp_ids

    ['exp_and_ama', 'only_exp', 'only_ama']

    dict_sort('exp_and_ama', exp_and_ama, exp_dict, ama_dict)
    dict_sort('only_exp', only_exp, exp_dict, ama_dict)
    dict_sort('only_ama', only_ama, exp_dict, ama_dict)

    
    print(len(exp_and_ama))
    print(exp_and_ama)

    print(len(only_exp))
    print(only_exp)

    print(len(only_ama))
    print(only_ama)

    
    from pprint import pprint
    
    exp_wrong_word = {}
    exp_words = set()
    exp_wrong_count = 0
    for key in exp_wrong_set['pred']:
        exp_wrong_word[key] = sum(exp_wrong_set['pred'][key].values())
        exp_wrong_count += sum(exp_wrong_set['pred'][key].values())
        exp_words.add(key)
            
    exp_wrong_word = sorted(exp_wrong_word.items(), key=lambda item: item[1], reverse=True)

    # print(len(exp_wrong_word))
    # print(exp_wrong_count)
    # pprint(exp_wrong_set['pred']['is'])

    # pprint(exp_wrong_word)

    ama_wrong_word = {}
    ama_words = set()
    ama_wrong_count = 0
    for key in ama_wrong_set['pred']:
        ama_wrong_word[key] = sum(ama_wrong_set['pred'][key].values())
        ama_wrong_count += sum(ama_wrong_set['pred'][key].values())
        ama_words.add(key)
    
    ama_wrong_word = sorted(ama_wrong_word.items(), key=lambda item: item[1], reverse=True)

    print(len(ama_wrong_word))
    print(ama_wrong_count)
    pprint(ama_wrong_set['pred']['is'])
# pprint(exp_wrong_set['pred']['to'])

    
