import streamlit as st
import pandas as pd
import time
import requests
import json

SUM_PATH = "temp_data.csv"
# column = {id, nickname, num_total_comment, num_curse_comment}
DETAIL_PATH = "temp_total_data.csv"
# column = {id, curse_comment}

def load_det():
    df = pd.read_csv(DETAIL_PATH, index_col = 0)
    return df

def load_sum():
    # 채팅 로그 summary를 불러오는 함수
    df = pd.read_csv(SUM_PATH, index_col = 0)
    df["ratio"] = df.num_curse_comment/df.num_total_comment
    df = df.sort_values(by = ['ratio'], ascending = False)

    top_k = min(len(df), 10)

    return df.iloc[:top_k]

def main():

    st.title("10조 최종 프로젝트")

    # initialize refresh button 
    refresh_flag = 0 # refresh 버튼을 누를 때에만 활성화

    btn_refresh = st.button(label = "refresh")

    if btn_refresh:
        t = time.localtime()
        cur_time = time.strftime("%H:%M:%S", t)
        st.success(f"refreshed ({cur_time})")
        refresh_flag = 1

    # body
    
    st.write(
        """
        ## 악성 채팅자 목록
        (혐오 발화 / 전체 채팅)을 기준으로 상위 10명을 표시합니다.
        """
    )

    # load data
    if refresh_flag == 1: # refresh 누를 때마다 정보 가져오기
        st.session_state.data_sum = load_sum()
        st.session_state.data_detail = load_det()

    if "data_sum" in st.session_state:
        show_summary(st.session_state.data_sum)
        show_details(st.session_state.data_sum, st.session_state.data_detail)
    
    result_container = st.container()

    with result_container:        
        ban_button = st.button("ban")
        if ban_button:

            with open ("/opt/ml/Twitch_bot/config.json", "r") as f:
                config = json.load(f)

            broadcaster_id = config['broadcaster_id']
            client_id = config['client_id']
            access_token = config['access_token']
            
            url = f"https://api.twitch.tv/helix/moderation/bans?broadcaster_id={broadcaster_id}&moderator_id={broadcaster_id}"
            
            head = {
                'Client-ID' : client_id,
                'Authorization':"Bearer " + access_token,
                'Content-Type': 'application/json'
                }

            #data = {'user_id' : ban_id, 'duration' : 300, 'reason': "no reason"}
            data_2 = {'data': [{'user_id' : str(st.session_state.ban_id), 'duration' : 30, 'reason': "no_reason"}]}
            json_data = json.dumps(data_2)
            print(json_data)

            r = requests.post(url = url, headers = head, data = json_data).json()
            
            st.success(f"{st.session_state.ban_nickname} is successfully banned")

def show_details(sum_data, detail_data):
    ids = sum_data["id"].values
    names = sum_data["nickname"].values
    buttons = []

    standard = min(10, len(ids))

    # button 만들기
    cols = st.columns(5)

    for i in range(standard):
        ii = i%5
        with cols[ii]:
            buttons.append(st.button(label = f"{i+1} : {names[i]}", key = f"{int(ids[i])}"))

    for idx, btn in enumerate(buttons):
        if btn:
            show_detail(detail_data, names[idx], ids[idx])

def show_detail(detail_data, nickname, id):
    st.session_state.ban_nickname = nickname
    st.session_state.ban_id = id

    st.header(f"{nickname}의 혐오 발화 목록")

    cond = detail_data["id"] == id
    data = detail_data[cond].drop("id", axis = 1)
    st.write(data)

def show_summary(data):
    data_to_display = data.drop("id", axis =1)
    data_to_display.index = range(1,len(data)+1)
    container = st.container()
    container.write(data_to_display)

if __name__ == "__main__":
    main()


