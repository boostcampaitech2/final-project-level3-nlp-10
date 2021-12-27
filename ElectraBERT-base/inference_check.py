import pandas as pd
import argparse

def check_inference(name):
    df_inf = pd.read_csv(f"./data/{name}WithElectra.tsv", sep='\t')
    labeled = (df_inf["none"] < df_inf["curse"]).astype(int)
    df_inf['label'] = labeled

    df_trn = pd.read_csv(f"./data/{name}.tsv", sep='\t')

    idx1 = df_inf[df_inf['curse'] < 0.9].index
    idx2 = df_inf[df_inf['none'] < 0.9].index
    idx3 = df_trn[df_inf['label'] != df_trn['label']].index

    df_trn.iloc[idx1 & idx2 | idx3][['text', 'label']].to_csv("check_again.tsv", sep='\t', index=False)
    df_trn.drop(index=(idx1 & idx2 | idx3))[['text', 'label']].to_csv("same_label.tsv", sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checking Inference result for Active Learning')
    parser.add_argument('--data', type=str, help='data')
    args = parser.parse_args()

    check_inference(name=args.data)