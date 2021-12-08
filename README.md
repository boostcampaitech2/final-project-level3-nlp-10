# nlp-10 final project 라벨링 툴 v1.0

데이터 라벨링 툴입니다.

서버에 pull 해서 터미널에서 실행하시면 됩니다.

## 실행법

`python labeling.py --data {데이터 이름}`

{데이터 이름}에 curse나 hate 등 데이터 이름을 넣어 실행합니다.

## 사용법

터미널에 띄워진 텍스트와 참고용으로 나오는 텍스트 안의 badword를 보고 input keyword를 입력합니다.

**[ input keyword 종류 ]**

0: 현재 텍스트는 아무런 욕설/혐오 표현이 없습니다.

1: 현재 텍스트에는 욕설 표현이 존재합니다. (혐오 x)

2: 현재 텍스트에는 혐오 표현이 존재합니다. (욕설 x)

3: 현재 텍스트는 욕설/혐오 표현에 모두 해당합니다.

p: 현재 텍스트에 대해 판단이 어려우므로 넘어갑니다.

b: 이전 텍스트로 돌아갑니다. (**주의**: 텍스트 번호에 따라 세이브 파일을 만드므로 이전 텍스트로 돌아가서 종료하면 해당 텍스트 작업 까지만 저장됩니다.)

quit: 저장 후 종료합니다.

## 데이터 종류

1. curse

원본 데이터: https://github.com/2runo/Curse-detection-data

욕설 감지 데이터셋을 이용했습니다.

2. hate

원본 데이터: https://github.com/kocohub/korean-hate-speech

Korean HateSpeech Dataset을 이용했습니다.

3. test

curse 데이터의 상위 11개 데이터만 뽑아 만들었습니다.

## 데이터 세이브

라벨링 하는 중에 계속 현재 문서 위치까지 tmp 폴더에 .csv파일로 저장합니다.

그리고 다시 labeling.py를 실행하면 tmp 폴더에 저장된 파일을 불러와 작업을 재개할 수 있습니다.

예를 들어, curse 데이터를 작업하면 tmp 폴더에 `tmp_curse.csv`파일이 생성됩니다.

curse 데이터 작업 중 종료하고 다시 실행할 경우 `python labeling.py --data curse`로 다시 실행하면 자동으로 세이브 파일을 불러옵니다.

만약 라벨링을 처음부터 다시 하고 싶다면, tmp 폴더에서 세이브 파일을 지우고 `labeling.py`를 실행해주세요.

라벨링이 끝나면 `{데이터 이름}_labeled.csv`파일이 생성됩니다.
