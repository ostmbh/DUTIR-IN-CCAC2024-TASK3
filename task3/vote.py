import json
import glob
from collections import defaultdict, Counter

def read_json_files(directory):
    file_paths = glob.glob(f"{directory}/*.json")
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    print(len(data))
    return data

def aggregate_data(data, task_type):
    aggregated = defaultdict(list)
    for entry in data:
        data_id = entry['dataId']
        if task_type == 1:
            sarcasm_info = entry['isSarcasm']
        elif task_type == 2:
            sarcasm_info = entry['sarcasmType']
        elif task_type == 3:
            sarcasm_info = entry['sarcasmTarget']
        else:
            raise ValueError("Invalid task type. Please choose 1, 2, or 3.")
        if isinstance(sarcasm_info, list):
            aggregated[data_id].extend(sarcasm_info)
        else:
            aggregated[data_id].append(sarcasm_info)
    return aggregated

def majority_vote(aggregated_data, task_type, threshold=0.5):
    majority_voted = []
    controversial_ids = []
    for data_id, items in aggregated_data.items():
        counter = Counter(items)
        total_count = sum(counter.values())
        controversial = False
        
        if task_type == 3:
            entity_votes = defaultdict(int)
            for entity in items:
                entity_votes[entity] += 1

            majority_items = []
            for entity, count in entity_votes.items():
                if count / total_count >= threshold:
                    majority_items.append(entity)

            # Include all entities even if not reaching the threshold
            all_entities = list(entity_votes.keys())
            
            if len(majority_items) < len(all_entities):  # 如果并不是所有实体都达到了阈值，认为存在争议
                controversial = True
            
            if controversial:
                controversial_ids.append(data_id)
                
            majority_voted.append({"dataId": data_id, "sarcasmTarget": all_entities})
        else:
            max_count = max(counter.values())
            if max_count / total_count < threshold:
                controversial_ids.append(data_id)
            majority_item = counter.most_common(1)[0][0]
            if task_type == 1:
                majority_voted.append({"dataId": data_id, "isSarcasm": majority_item})
            elif task_type == 2:
                majority_voted.append({"dataId": data_id, "sarcasmType": majority_item})
    return majority_voted, controversial_ids

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def save_controversial_ids(controversial_ids, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(controversial_ids, f, ensure_ascii=False, indent=4)


def main():
    directory = r'./results/'  # 将此处替换为你的JSON文件目录
    task_type = 1  # 指定任务类型：1、2或3
    output_file = r'./results/task1_voted_results.json'
    controversial_file = r'./results/task1_voted_controversial.json'
    threshold = 0.5  # 设置争议阈值

    data = read_json_files(directory)
    aggregated_data = aggregate_data(data, task_type)
    majority_voted, controversial_ids = majority_vote(aggregated_data, task_type, threshold)
    save_results(majority_voted, output_file)
    save_controversial_ids(controversial_ids, controversial_file)

if __name__ == "__main__":
    main()
