import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer
from input_prompt import prompt_generator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

json_file_path = 'special_tokens_map.json'
restaurant_json = pd.read_json('restaurant_info.json') 

def load_tokens_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        tokens = json.load(file)
    return tokens

model = AutoModelForCausalLM.from_pretrained("save_checkpoints_llmrs/checkpoint-1000")
model.eval()


tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
pipe = pipeline(task="text-generation", model=model, device="cuda:0", max_length=2000, tokenizer=tokenizer)
pipe.model.eval()

logging.set_verbosity(logging.CRITICAL)
    
def recommend_prompt(input_dict):

    system_prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instuction = f"## Instruction:\n두 사용자가 방문한 장소 input 데이터를 바탕으로 두 사람이 함께 즐길 수 있는 레스토랑/카페/바를 추천해 주세요.\n\n ## Input:\n{input_dict} \n\n ## Response:\n"
    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{instuction}"}
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminiator = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    return prompt, terminiator

def generate_prompt(input_dict):
    input_dict = eval(input_dict)
    recommended_place = input_dict["recommended_place"]
    recommended_place_names = []
    for place in recommended_place:
        place_name = restaurant_json['name'][int(place)]
        if place_name:
            recommended_place_names.append(place_name)
        else:
            recommended_place_names.append(f"Unknown ID {place}")
    input_dict["recommended_place"] = recommended_place_names

    system_prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instuction = f"## Instruction:\n추천시스템이 추천한 장소들인 search 데이터를 바탕으로 어떻게 두 사람의 선호도를 충족시키는지 설명해 주세요.\n\n ## Input:\n{input_dict} \n\n ## Response:\n"
    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{instuction}"}
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminiator = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    return prompt, terminiator

print("*********** LLM 챗봇입니다. 무엇을 도와드릴까요? (대화 종료: q) ***********")

while True:
    print("LLM: 사용자 A와 B의 방문한 데이터를 입력해주세요.")
    prompt_recommend = input("user >  ")
    if prompt_recommend == "q":
        break
    prompt_input, terminator = recommend_prompt(prompt_recommend)
    result = pipe(prompt_input, max_new_tokens=2048, eos_token_id=terminator, repetition_penalty=1.1)
    result = result[0]['generated_text'].split(" {'content'")[0]
    prompt_generate = result.split("## Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
    prompt_input, terminator = generate_prompt(prompt_generate)
    result = pipe(prompt_input, max_new_tokens=2048, eos_token_id=terminator, repetition_penalty=1.1)
    result = result[0]['generated_text'].split(" {'content'")[0]
    result = result.split("Input:")[1]
    print("bot >  추천 결과: LLM 이 분석한 사용자 A 와 사용자 B 의 선호도 및 추천식당 정보는 아래와 같습니다." + result)

print("대화를 종료합니다.")
    
# {"user_a": [45,6,254,978], "user_b": [182,932,2671,2342]}
    
