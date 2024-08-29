import pandas as pd

class prompt_generator():
    def __init__(self, restaurant_json):
        self.restaurant_json = restaurant_json
        self.restaurant_data = {row['restaurant_id']: row['name'] for index, row in self.restaurant_json.iterrows()}
    
    def recommend_places(self, data):
        user_a = data["user_a"]
        user_b = data["user_b"]
        user_a_prefer = data["user_a_prefer"]
        user_b_prefer = data["user_b_prefer"]
        recommended_place = data["recommended_place"]
        menu = data["menu"]

        recommendation = {
            "instruction_recommend": "두 사용자가 방문한 장소 input 데이터를 바탕으로 두 사람이 함께 즐길 수 있는 레스토랑/카페/바를 추천해 주세요.",
            "input": {"user_a": user_a, "user_b": user_b},
            "output": {
                "user_a_prefer": user_a_prefer, # ["조용한, 분위기 있는"]
                "user_b_prefer": user_b_prefer,
                "recommended_place": recommended_place, 
                "menu": menu
            }
        }
        
        return recommendation

    def generate_recommendation_output(self, data):
        user_a_prefer = data["user_a_prefer"]
        user_b_prefer = data["user_b_prefer"]


        if isinstance(user_a_prefer, str):
            user_a_prefer = [user_a_prefer]
        if isinstance(user_b_prefer, str):
            user_b_prefer = [user_b_prefer]

        recommended_place = data["recommended_place"]
        menu = data["menu"]

        recommended_place_names = []
        for place in recommended_place:
            place_name = self.restaurant_data.get(place)
            if place_name:
                recommended_place_names.append(place_name)
            else:
                recommended_place_names.append(f"Unknown ID {place}")

        menu_output = ""
        if isinstance(menu, list):
            menu_output = ', '.join(m if m is not None else '' for m in menu)
        elif isinstance(menu, str):
            menu_output = menu  
        else:
            menu_output = ', '.join(str(m) for m in menu if m) 

        output = (
            f"A는 {', '.join(user_a_prefer)} 무드를 선호하고 "
            f"B는 {', '.join(user_b_prefer)} 무드를 선호하므로 "
            f"장소 {', '.join(recommended_place_names)}를 추천합니다. "
            f"대표 메뉴는 {menu_output}입니다."
        )

        result = {
            "instruction_generate": "추천시스템이 추천한 장소들인 search 데이터를 바탕으로 어떻게 두 사람의 선호도를 충족시키는지 설명해 주세요.",
            "input": {
                "user_a_prefer": user_a_prefer,
                "user_b_prefer": user_b_prefer,
                "recommended_place": recommended_place_names,
                "menu": menu
            },
            "output": output
        }

        return result

    def prompt_input(self, data, mode=0):
        if mode == 0: #recommend
            return self.recommend_places(data)
        
        elif mode == 1: #generate
            return self.generate_recommendation_output(data)
        
if __name__ == "__main__":
    
    restaurant_json = pd.read_json('restaurant_info.json')
    prompt_gen = prompt_generator( restaurant_json)
