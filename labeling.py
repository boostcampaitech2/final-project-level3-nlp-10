import pickle
import time
import os
import sys
import argparse
import pandas as pd
from pprint import pprint

with open("badword.pickle", "rb") as fr:
    badword = pickle.load(fr)

def labeling(data_name):
    
    if os.path.isfile(f"{data_name}.csv"):
        df = pd.read_csv(f"{data_name}.csv")
    else:
        print(f"There is no such file as {data_name}.csv")
        return None
    
    i = 0
    
    if os.path.isfile(f"tmp/tmp_{data_name}.csv"):
        tmp = pd.read_csv(f"tmp/tmp_{data_name}.csv")
        df.update(tmp)
        df = df.astype({"text": str,
                        "curse": int,
                        "hate": int,
                        "pass": int,
                       })
        i += len(tmp)
        
    while i < len(df):
        os.system('clear')
        while True:
            if i < 0:
                i = 0
            print(f"[text {i}/{len(df)-1}]")
            print(df.iloc[i].text)
            print()
            
            badword_in_text = [word for word in badword if (word in df.iloc[i].text)]
            if badword_in_text:
                print("Badword(s) in text:", badword_in_text)
                print()

            print("="*80)
            print("\nIs this text curse speech, hate speech, or nothing?")
            print("Pass if text is uncertain.\n")
            print("[0: Nothing, 1:Curse, 2: Hate, 3: Both, p: Pass, b: Back, quit: Quit]")
            print()
            
            user_input = input("input keyword > ")
            if user_input == '0':
                df.loc[i, 'curse'] = 0
                df.loc[i, 'hate'] = 0
                df.loc[i, 'pass'] = 0
                i += 1
            elif user_input == '1':
                df.loc[i, 'curse'] = 1
                df.loc[i, 'hate'] = 0
                df.loc[i, 'pass'] = 0
                i += 1
            elif user_input == '2':
                df.loc[i, 'curse'] = 0
                df.loc[i, 'hate'] = 1
                df.loc[i, 'pass'] = 0
                i += 1
            elif user_input == '3':
                df.loc[i, 'curse'] = 1
                df.loc[i, 'hate'] = 1
                df.loc[i, 'pass'] = 0
                i += 1
            elif user_input == 'p':
                df.loc[i, 'curse'] = 0
                df.loc[i, 'hate'] = 0
                df.loc[i, 'pass'] = 1
                i += 1
                break
            elif user_input == 'b':
                i -= 1
                break
            elif user_input == 'quit':
                df.loc[:i-1, :].to_csv(f"tmp/tmp_{data_name}.csv", index=False)
                print(f"Data saved to 'tmp_{data_name}.csv'")
                sys.exit()
            else:
                print("Wrong Input")
                time.sleep(1)
                os.system('clear')
                continue
            break
        df.loc[:i, :].to_csv(f"tmp/tmp_{data_name}.csv", index=False)
        time.sleep(0.2)
    
    os.system('clear')
    print("Labeling Finished!")
    df.to_csv(f"{data_name}_labeled.csv", index=False)
    print(f"Data saved to '{data_name}_labeled.csv'")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labeling Tool.")
    parser.add_argument("--data", default="test", type=str, help="Data to label.")
    args = parser.parse_args()
    labeling(args.data)