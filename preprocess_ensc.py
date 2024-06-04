import json
import os

'''
AI hub 한글 표 데이터셋(기술과학 문서 기계독해 데이터)을
전처리하는 코드
'''

path="dataset_korean/152.기술과학 문서 기계독해 데이터"

train_label=os.path.join(path, "Training","02.라벨링데이터")
val_label=os.path.join(path, "Validation","02.라벨링데이터")

def transform_data(data):
    transformed_data = []

    for qa in data["qas"]:
        question = qa.get("question-1")
        answer = qa.get("answer")
        clue_text = qa["clue"][0]["clue_text"] if qa["clue"] else ""
        
        transformed_item = {
            "Table": clue_text,
            "Table Description": "표설명",
            "Question": question,
            "Solution": f"Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question].\n### [Table] ''' {clue_text} '''\n### [Table Description] 표설명\n### [Question] {question}\n### [Solution] {answer}"
        }

        transformed_data.append(transformed_item)
    
    return transformed_data

all_transformed_data=[]

# Transform the data
folders=[train_label, val_label]
for folder in folders:
        for file in os.listdir(folder):
            file_path=os.path.join(folder, file)
            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)
                transformed_data = transform_data(data)
                all_transformed_data.append(transformed_data)

# Flatten the list of transformed data
flattened_transformed_data = [item for sublist in all_transformed_data for item in sublist]

# Write the transformed data to a JSON file
with open('transformed_data.json', 'w', encoding='utf-8') as f:
    json.dump(flattened_transformed_data, f, indent=4, ensure_ascii=False)