import sys  
import json
import requests
from Bot import TwitchBot
from urllib.parse import urlencode

def main():

    # read configuration
    with open ("config.json", "r") as f:
        config = json.load(f)

    client_id = config['client_id']
    client_secret = config['client_secret']
    channel_name = config['channel_name']
    irc_token = config['irc_token']

    redirect_uri = "http://localhost:5500"
    scope = ["moderation:read","moderator:manage:banned_users"]

    # give authority to the client

    data = {
        'client_id': client_id,
	    'redirect_uri': redirect_uri,
	    'response_type': 'code',
	    'scope': " ".join(scope)   
    }

    url_give_auth = 'https://id.twitch.tv/oauth2/authorize?' + urlencode(data, doseq=True)
    print('Authorization URL:')
    print(url_give_auth)

    code = input("input the authorization code")
    
    url = "https://id.twitch.tv/oauth2/token"
    params = {'client_id': client_id,
            'client_secret': client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'force_verify' : 'true',
            'redirect_uri': redirect_uri
            }

    response = requests.post(url, params = params).json()
    acc_token = response['access_token']

    # get broadcaster id

    url = f"https://api.twitch.tv/helix/users?login={channel_name}"
    head = {
        'Client-ID' : client_id,
        'Authorization':"Bearer " + acc_token
        }

    response = requests.get(url, headers = head).json()
    broadcaster_id = response['data'][0]['id']

    config['broadcaster_id'] = broadcaster_id
    config['access_token'] = acc_token

    with open ("config.json", "w") as f:
        json.dump(config, f)

    bot = TwitchBot(client_id = client_id, irc_token = irc_token, access_token = acc_token, channel = channel_name, broadcaster_id = broadcaster_id)
    bot.start()

if __name__ == "__main__":
    main()
    