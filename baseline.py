import argparse
import json
import tqdm

def rule_based_filter(text, filter, flag):
    """
    input:
        text : text file to be filtered
        filter : json file filled with bad words to be filtered
        flag : if True, save filtering result

    output:
        None
    """

    with open (filter, "r") as f:
        badwords = json.load(f)['badwords']
    with open (text, "r") as f:
        texts = f.readlines()

    result = []
    
    for i, text in tqdm.tqdm(enumerate(texts)):
        for badword in badwords:
            if badword in text:
                result.append((i, badword))
                break
    
    if flag:
        with open("output.txt", "w") as f:
            for i in range(len(result)):
                id_ = result[i][0]
                string = f"text_ind : {id_}, text : {texts[id_].rstrip()}, badword : {result[i][1]}\n"
                f.write(string)

    print("="*12)
    print(f"totally {len(texts)} comments input")
    print(f"and {len(result)} comments filtered ({round(100* len(result)/len(texts), 4)}%)")
    print("="*12)

    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--text", type = str, default = "text.txt", help = "text file to be filtered")
    argparser.add_argument("--filter", type = str, default = "korean_badwords.json", help = "json file filled with bad words to be filtered")
    argparser.add_argument("--save_result", action = "store_true", help = "save result filtered in the name of \"output.txt\"")

    args = argparser.parse_args()

    text = args.text
    filter = args.filter
    flag = args.save_result

    rule_based_filter(text,filter,flag)
