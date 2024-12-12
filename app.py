import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import json

# 모델 및 토크나이저 로드
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("save_checkpoints_llmrs/checkpoint-1000")
    tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=0)
    return pipe

pipe = load_model()

# 레스토랑 정보 로드
@st.cache_data
def load_restaurant_data():
    return pd.read_json("restaurant_info.json")

restaurant_json = load_restaurant_data()

# 추천 프롬프트 생성
def recommend_prompt(input_dict):
    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = "## Instruction:\n두 사용자가 방문한 장소 input 데이터를 바탕으로 두 사람이 함께 즐길 수 있는 레스토랑/카페/바를 추천해 주세요.\n\n"
    input_text = f"## Input:\n{input_dict}\n\n## Response:\n"
    return f"{system_prompt}{instruction}{input_text}"

# Streamlit UI
st.title("LLM 기반 레스토랑 추천 시스템")
st.write("두 사용자의 선호도를 기반으로 레스토랑을 추천합니다.")

# 사용자 입력 받기
user_a_data = st.text_input("사용자 A의 방문 장소 ID (예: 45,6,254,978):")
user_b_data = st.text_input("사용자 B의 방문 장소 ID (예: 182,932,2671,2342):")

if st.button("추천 실행"):
    if user_a_data and user_b_data:
        # 사용자 입력 처리
        try:
            user_a_list = [int(x) for x in user_a_data.split(",")]
            user_b_list = [int(x) for x in user_b_data.split(",")]

            input_dict = {
                "user_a": user_a_list,
                "user_b": user_b_list
            }

            # 추천 프롬프트 생성 및 결과 생성
            prompt = recommend_prompt(input_dict)
            result = pipe(prompt, max_new_tokens=200, repetition_penalty=1.1)
            recommendation = result[0]['generated_text']

            st.success("추천 결과")
            st.write(recommendation)

        except Exception as e:
            st.error(f"에러가 발생했습니다: {e}")
    else:
        st.error("모든 입력값을 제공해주세요!")

st.write("대화를 종료하려면 창을 닫아주세요.")
