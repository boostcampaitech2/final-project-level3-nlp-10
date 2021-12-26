# 실시간 악성채팅 감지 시스템

## 실제 활용 예시

![최종프로젝트](https://user-images.githubusercontent.com/47023884/147060316-82c2f9b8-df6f-4a3a-bc87-a6a330a05293.gif)

## 문제 정의

개인방송을 하다보면 채팅창의 물을 흐리는 악질적인 시청자가 존재하는 경우가 있습니다. </br>
시청자 수가 적을 경우 일일이 제재를 가하면 되지만, 시청자 수가 많은 경우 이들을 전부 관리하는 데에 한계가 있기 마련입니다. </br>
따라서 인공지능을 통해 악질 시청자를 분류해 채팅창을 쉽게 관리하는 것을 목표로 프로젝트를 진행하였습니다.

### 예상되는 문제점

1. **기존 서비스 존재**

이미 기존에 싹둑이나 Nightbot과 같이 채팅창을 관리해주는 서비스가 존재하고 있습니다.

2. **오인 제재**

혐오표현이 아니더라도, 혐오표현과 유사한 문구가 포함된 문장은 혐오표현으로 분류될 수 있습니다.

3. **관대한 방송인**

방송인이 채팅을 제재하는 기준은 방송인에 따라 다릅니다. 아무리 심한 욕이라도 검열을 하지 않는 방송인이 있는 반면 아무리 사소한 표현이라도 검열하는 방송인이 있습니다.

### 극복 방안

기존 서비스는 Rule-based로 악성 채팅을 분류합니다. 따라서 검열할 문구를 사전에 입력해주어야 하며, 검열 목록에 없는 표현들은 혐오표현이라도 검열되지 않습니다. </br>
본 프로젝트는 인공지능을 기반으로 만들어졌기 때문에 검열 목록에 없더라도 혐오표현으로 분류되면 검열이 가능합니다. </br>
또한, 오인 제재와 관대한 방송인 문제를 극복하기 위해 저희는 인공지능이 직접적으로 제재를 하기 보다는, 제재를 할 수 있도록 정보를 제공하는 형태로 제작하였습니다. </br>

## Data Labeling

### 사용한 데이터셋 목록

- 욕설 감지 데이터셋 ([Curse-detection-data](https://github.com/2runo/Curse-detection-data))
- BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection ([Korean Hate Speech Dataset](https://github.com/kocohub/korean-hate-speech))
- 트위치 라이브 스트리밍 채팅 데이터 로그 (crawled with [Chatty](https://chatty.github.io/))

트위치 채팅 데이터는 Chatty를 이용해 수집하였습니다. </br>
전처리 코드와 raw data는 ElectraBERT-base/data/twitch 폴더 안에서 확인할 수 있습니다.

### 데이터 라벨링 기준

데이터 라벨링의 기준(혐오표현의 기준)을 세우는 데에 고려한 사항들은 다음과 같습니다.
- 욕설 표현이 들어가 있는 경우 혐오표현으로 분류 </br>
  (씨X, 병X, 개새X, 새X 등)
- 단어 자체는 욕설이라 보기 힘들지만 문장 안에서 욕설이라고 표현될 수 있는 경우 혐오표현으로 분류
- 현재 단어가 단순 강조의 의미이거나 유행어로 사용된다 하더라도, 본 뜻이 충분히 욕설일 경우 혐오표현으로 분류 </br>
  (존X, 개같이, 씹 등)
- 성적인/음란한 표현들도 혐오표현으로 분류 </br>
  (섹X, 따먹는다, 꼴린다, 아다 등)
- 채팅창의 물을 흐린다고 할 수 있을만큼 공격적인 표현은 혐오표현으로 분류
- 정도가 매우 심한 지역/성별/인종 차별 발언도 혐오표현으로 분류

## Training

학습을 하기 위해 아래와 같은 arguments들을 정해서 학습시킬 수 있습니다.  
Meta Pseudo Labels이라는 학습 방법을 사용했으며 [링크](https://github.com/kekmodel/MPL-pytorch)에 있는 코드를 사용하여 적용했습니다.  
또한, MLflow를 설정하여 fine-tuning한 모델 중 가장 좋은 F1을 보인 모델을 서버에 저장할 수 있습니다.

```
python train.py --dropout1 0.3 \
                --dropout2 0.4 \
                --teacher_learning_rate 1e-7 \  
                --student_learning_rate 1e-7 \
                --label_smoothing 0 \
                --embeeding_dim 100 \
                --hidden_size 128 \
                --num_classes 2 \
                --epochs 1 \
                --seed 42 \
                --vocab_size 30000 \
                --batch_size 32 \
                --unlabeled_sample_frac 0.025 \
                --temperature 0 \
                --uda_lambda 1 \
                --uda_step 1 \
                --threshold 0.7 \
                --patient 20 \
                --finetune_learning_rate 1e-3 \
                --finetune_epochs 10 \
                --finetune_max_lr 0.01 \
                --finetune_pct_start 0.1
```

## Benchmark
### Models
labeled된 데이터의 20%를 테스트 데이터로 사용하고 10 epochs로 학습했을 때의 성능비교입니다.

|Model|Best Acc|Best F1|Inference time(128 batch, CPU)|
|-|-|-|-|
|Rule based|0.81|0.690|**0.01s**(1300 words)|
|CNN|0.90|0.878|0.03s|
|BiLSTM|**0.93**|0.90|0.28s|
|CNN + BiLSTM|0.92|**0.911**|0.3s|

### Parameters
|Model|Params|
|-|-|
|CNN + BiLSTM|**3390786**|
|MobileBERT-uncased|24582914|
|BERT-base|110618882|

### Training
labeled된 트위치 채팅 데이터를 테스트 데이터로 사용했습니다.  

|Training|Best Acc|Best F1|
|-|-|-|
|Only labeled|0.76|0.742|
|Pseudo labeling|0.75|0.728|
|MPL -> Fine-tuning|**0.79**|**0.757**|

* pseudo labeling 방법은 학습하려고 하는 모델에 unlabeled 데이터를 넣어 생성된 pseudo label을 다시 학습하는 방법으로 진행했습니다.  

## Reference
- [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580)
- 
