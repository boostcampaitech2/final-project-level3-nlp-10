import pandas as pd
import numpy as np

def checked_concat(with_train=False):
    df_ca = pd.read_csv("check_again.tsv", sep='\t')
    df_sl = pd.read_csv("same_label.tsv", sep='\t')
    if with_train:
        df_train = pd.read_csv("./data/train.tsv", sep='\t')
        df_trn_again = pd.concat([df_ca, df_sl, df_train], ignore_index=True)
    else:
        df_trn_again = pd.concat([df_ca, df_sl], ignore_index=True)

    # 중복되는 text 제거
    text_set = set()
    drop_index = []
    for i in range(len(df_trn_again)):
        if df_trn_again.iloc[i, 0] in text_set:
            print('dropped text:', df_trn_again.iloc[i, 0])
            drop_index.append(i)
        else:
            text_set.add(df_trn_again.iloc[i, 0])

    # 데이터 순서 섞기
    x = np.random.permutation(len(df_trn_again.drop(index=drop_index).reset_index(drop=True)))
    df_trn_again_ = df_trn_again.drop(index=drop_index).reset_index(drop=True).iloc[x].reset_index(drop=True)
    df_trn_again_.to_csv("./data/train.tsv", sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concat checked data into train data')
    parser.add_argument('--with-train', action='store_true', help='concat checked data with original train data')
    args = parser.parse_args()

    checked_concat(with_train=args.with_train)