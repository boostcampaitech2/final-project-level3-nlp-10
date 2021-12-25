"""
    배포된 모델 테스트틀 위한 테스트코드
    원하는 시나리오를 적용해서 모델 결과를 알 수 있습니다.
"""

# ignore warning : pytest --p no:warnings
# Model input, output test
import os
import mlflow
import numpy as np
from tokenizers import BertWordPieceTokenizer
from utils import GOOGLE_APPLICATION_CREDENTIAL, MLFLOW_TRACKING_URI


os.environ['GOOGLE_APPLICAION_CREDENTIALS']=GOOGLE_APPLICATION_CREDENTIAL
os.environ['MLFLOW_TRACKING_URI']=MLFLOW_TRACKING_URI


tokenizer = BertWordPieceTokenizer('./vocab.txt', lowercase=False)
model = mlflow.pyfunc.load_model(model_uri='models:/ToxicityText/Production')

"""setting"""
def tokenized(text):
    tokens = tokenizer.encode(text).ids
    if len(tokens) <= 200:
        for _ in range(200-len(tokens)): tokens.append(0)
    elif len(tokens) > 200:
        for _ in range(len(tokens)-200): tokens.pop()
    return tokens


def return_predict(text):
    input = np.array([tokenized(text)])
    y = model.predict(input)[0]
    return softmax(y).argmax(-1)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


"""Tokenizer"""
# 토크나이징된 텍스트의 길이는 200이어야 한다.
def test_tokenized_length_is_200():
    input = tokenized('야!')
    assert len(input)==200


"""Model"""
def test_input_none():
    print(return_predict(''))
    assert return_predict('')==0


def test_input_1():
    print(tokenized('ㅈㄴ'))
    assert return_predict('ㅈㄴ')==1


def test_input_2_5():
    print(tokenized('와 ㅈㄴ 똑똑해'))
    assert return_predict('와 ㅈㄴ 똑똑해')==1


def test_input_3():
    print(tokenized('왠 짱괘'))
    assert return_predict('왠 짱괘')==1


def test_input_3_5():
    print(tokenized('ㅅㅂ'))
    assert return_predict('ㅅㅂ')==1


def test_input_4():
    print(tokenized('유하유하유하'))
    assert return_predict('유하유하유하')==0


def test_input_5():
    print(tokenized('ㅈ같네'))
    assert return_predict('ㅈ같네')==1


def test_input_6():
    print(tokenized('개같이 멸망'))
    assert return_predict('개같이 멸망')==0


def test_input_7():
    print(tokenized('개꿀ㅋㅋㅋㅋㅋㅋㅋㅋ'))
    assert return_predict('개꿀ㅋㅋㅋㅋㅋㅋㅋㅋ')==0


def test_input_8():
    print(tokenized('한남'))
    assert return_predict('한남')==1


def test_input_9():
    print(tokenized('한녀'))
    assert return_predict('한녀')==1


def test_input_10():
    print(tokenized('계속 부들거려줘~ 정신병신아 ㅋㅋ'))
    assert return_predict('계속 부들거려줘~ 정신병신아 ㅋㅋ')==1