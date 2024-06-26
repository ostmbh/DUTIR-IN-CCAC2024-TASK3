import json

def main():
    train = []
    with open('/home/dutir_exp_4t/makai/ccac2024/Dataset_final/test_data.jsonl', 'r') as f:
        for line in f:
            train.append(json.loads(line))
    
    users = {}
    with open('/home/dutir_exp_4t/makai/ccac2024/Dataset_final/user_data_0521.jsonl', 'r') as f:
        for line in f:
            users = json.loads(line)
    
    label = {'喜': 0, '哀': 1, '惊': 2, '恐': 3, '怒': 4}
    
    
    l = [0] * len(label)
    
    new_train = []
    for line in train:
        new_line = {}
        new_line['text'] = users[line['num']]['Location'] + '|' + users[line['num']]['Gender'] + '|' + line['text']
        new_line['label'] = label[line['emo']]
        new_train.append(new_line)
        l[label[line['emo']]] += 1  # 对应标签的计数加1
    
    label_inverse = {v: k for k, v in label.items()}  # 反转label字典
    
    for idx, count in enumerate(l):
        print(f"标签 {label_inverse[idx]}: {count} 个")
    
    with open('test_0521_with_embeddings.txt', 'w') as f:
        f.write("text	label\n")
    
    for idx,line in enumerate(new_train):
        with open('test_0521_with_embeddings.txt', 'a') as f:
            f.write("%s	%s\n" % (line['text'], line['label']))
    print(len(new_train))

if __name__ == "__main__":
    main()