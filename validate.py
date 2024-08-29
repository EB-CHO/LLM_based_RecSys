def key_checker(response):
    keys = {'user_a_prefer','user_b_prefer','recommended_place','menu'}
    response_keys = response.keys()
    if keys == response_keys:
        return True
    else:
        return False

def value_checker(response):
    for key in response.keys():
        if not response[key]:
            return False
    return True
                
def recommend_checker(response):
    for value in response['recommended_place']:
        if not 0 <= value <= 3160:
            return False
    return True

class LLM_inference():
    def __init__(self, llm_model, gcn):
        self.llm_model = llm_model # 학습시킨 LLM model 사용
        self.gcn = gcn
        self.error_cnt = 0

    def validate(self, response):
        if isinstance(response, dict) and key_checker(response) and value_checker(response) and recommend_checker(response):
            return True
        else:
            return False
    
    #FIXME: GCN input 구조에 맞게 수정
    def extract_gcn_value(self, input_prompt):
        return NotImplementedError

    def __call__(self, input_prompt):

        while self.error_cnt < 10:
            response = self.llm_model(input_prompt)
            validate_output = self.validate(response)
            if validate_output:
                self.error_cnt = 0
                return response
            else:
                self.error_cnt += 1
                response = self.__call__(input_prompt)
        
        if self.error_cnt == 10:
            self.error_cnt = 0
            gcn_input = self.extract_gcn_value(input_prompt)
            return self.gcn(gcn_input)

