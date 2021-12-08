## Electra training

" `beomi/KcELECTRA-base`에 ElectraForSequenceClassification을 올려 학습하는 코드입니다.  

먼저, `dataset.ipynb`에서 토크나이저를 불러와 데이터셋에서 거르지 못하는 단어들을 확인합니다. 토크나이저의 [UNK]는 1번이므로 1번이 들어간 문장을 따로 저장하여 어떤 단어가 [UNK]로 대체되었는지 확인합니다.  

이모티콘 혹은 처음보는 특수문자가 대체되었다면 전처리 함수에 추가하여 전처리 후에 토크나이저를 재학습시키시면 됩니다.

다음 `train.ipynb`에서 일렉트라를 불러와 학습합니다. 파라미터는 마음대로 했습니다.

마지막으로, `eval.ipynb`에서 추론하고 싶은 문장들을 불러와 진행하고 데이터를 저장합니다.