import sys

from requests import models  
import irc.bot  
import json
import requests
import pandas as pd

class TwitchBot(irc.bot.SingleServerIRCBot):
    
    def __init__(self, client_id, irc_token, access_token, channel, broadcaster_id): 

        self.client_id = client_id  
        self.access_token = access_token
        self.channel = '#' + channel  
        self.irc_token = irc_token
        self.hogumastack = 0  
        self.broadcaseter_id = broadcaster_id
        self.user_name = "veonico"
        self.id2idx = {}
        self.summ = [] # id, nickname, n_total, n_curse
        self.detail = [] # id, curse
        self.idx = 0
        self.chatting_count = 0
    
        with open ("/opt/ml/Baseline/korean_badwords.json", "r") as f:
            self.badwords = json.load(f)['badwords']
        
        # IRC bot 연결 생성  
        server = 'irc.chat.twitch.tv'  
        port = 6667  

        print('\n서버 ' + server + ', 포트 ' + str(port) + '에 연결 중...\n')  
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port, "oauth:" + self.irc_token)], self.user_name, self.user_name)

    def on_welcome(self, c, e):

        print(self.channel + '에 연결되었습니다.')  
  
        #봇을 사용하기 전에 채널 권한 부여가 필요  
        c.cap('REQ', ':twitch.tv/membership')  
        c.cap('REQ', ':twitch.tv/tags')  
        c.cap('REQ', ':twitch.tv/commands')  

        c.join(self.channel)

    def on_pubmsg(self, c, e):

        #[{'key': 'badge-info', 'value': None}, {'key': 'badges', 'value': 'broadcaster/1'}, {'key': 'client-nonce', 'value': '2a4fa3954c32201f629d553a90d5a867'}, {'key': 'color', 'value': None}, {'key': 'display-name', 'value': '진수야널좋아해'}, {'key': 'emotes', 'value': None}, {'key': 'first-msg', 'value': '0'}, {'key': 'flags', 'value': '0-1:A.7'}, {'key': 'id', 'value': 'd42557e9-900d-4da1-924b-c3417fb8a286'}, {'key': 'mod', 'value': '0'}, {'key': 'room-id', 'value': '193694284'}, {'key': 'subscriber', 'value': '0'}, {'key': 'tmi-sent-ts', 'value': '1638863668191'}, {'key': 'turbo', 'value': '0'}, {'key': 'user-id', 'value': '193694284'}, {'key': 'user-type', 'value': None}]
        
        # 채팅 카운트 증가
        self.chatting_count += 1

        # 필요한 정보만 추리기
        user_nickname = e.source.split("!")[0]
        user_id = e.tags[-2]['value']
        chatting = e.arguments[0]

        # summary

        if user_id not in self.id2idx.keys(): # 없을 시 해당 id에 대한 record 초기화
            idx = self.idx
            self.id2idx[user_id] = idx
            self.summ.append([user_id, user_nickname, 1, 0])
            self.idx += 1
        else: # 있을 시 해당 id에 대한 total_commen 1 증가
            idx = self.id2idx[user_id]
            self.summ[idx][2] += 1

        # detail
            
        for badword in self.badwords:
            if badword in chatting:
                self.summ[idx][3] += 1
                self.detail.append([user_id, chatting])

                """ban code
                url = f"https://api.twitch.tv/helix/moderation/bans?broadcaster_id={self.broadcaseter_id}&moderator_id={self.broadcaseter_id}"
                
                head = {
                    'Client-ID' : self.client_id,
                    'Authorization':"Bearer " + self.access_token,
                    'Content-Type': 'application/json'
                    }

                #data = {'user_id' : ban_id, 'duration' : 300, 'reason': "no reason"}
                data_2 = {'data': [{'user_id' : ban_id, 'duration' : 30, 'reason': badword}]}
                json_data = json.dumps(data_2)

                print(json_data)
                r = requests.post(url = url, headers = head, data = json_data).json()
                print(r)
                """
                break
        
        # 채팅이 다섯 개 쌓일 때마다 csv 파일 저장
        if self.chatting_count == 5:
            sum_df = pd.DataFrame(self.summ, columns = ["id","nickname","num_total_comment","num_curse_comment"])
            sum_df.to_csv("/opt/ml/frontend/temp_data.csv")
            detail_df = pd.DataFrame(self.detail, columns = ["id","curse_comment"])
            detail_df.to_csv("/opt/ml/frontend/temp_total_data.csv")
            self.chatting_count = 0

        return