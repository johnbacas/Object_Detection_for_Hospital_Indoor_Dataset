import json
from pprint import pprint

with open("yoloe-11l-seg_1152_txt_prompt_eval_results.json", "r") as f:
    data = json.load(f)
    class_results = data["Supervision_mAP50_75_50-95_scores_per_class"]
    for i, class_result in enumerate(class_results):
        print(f"{i} {class_result[0]}:\t{class_result[1]}     {class_result[2]}     {class_result[3]}")