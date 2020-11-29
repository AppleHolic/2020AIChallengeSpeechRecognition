# 2020 AI Challenge - SpeechRecognition
[2020 AI Challenge](http://aifactory.space/aichallenge/total/search) 음성 인식 코드. 
Solo team showyourvoice 8th rank

### 룰 관련
- Rule에 따라 코드가 태스크 별로 하나로 병합되어 있습니다.
- 2가지 태스크를 수행한 코드들입니다.
  - 잡음 환경에서의 음성 인식 (noisy_task.py)
  - 어린아이 음성 인식 (children_task.py)
- Metric 및 제출 코드 관련
  - 어린이 음성 인식 metric : test cer 7.1
  - 잡음 음성 인식 metric : test cer 6.3 
  - children_task.py 코드는 noisy_task.py의 코드에 비해 약간 성능이 밀렸으나, 학습 속도 측면에서 크게 차이 났었습니다.

### 주요 알고리즘
  - 데이터 처리 : 
    - Text : grapheme, greedy decoder
    - Audio : 
      - Specaugment *Torch Audio - masking 활용*
      - 16khz sample rate, MFCC - Mel 40 또는 80 dims, 400 fft, 400 window, 200 hop length
  - 모델 및 학습 방법 관련
    - LAS [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) 
    - LAS의 RNN 및 기본 RNN 모델에 Layer Normalization 추가 : RNN + Layer Normalization
    - Optimizer : [AdamP](https://github.com/clovaai/AdamP) 
    - Drop out 등
  - Drop out, layer normalization, adamp 모두 성능 개선에 영향을 끼침
