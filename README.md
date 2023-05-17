# Swin-CIFAR10_test
- Description: Testing of Swin Transformer on CIFAR-10 dataset


## 실행 방법
1. CIFAR10_swin_test.ipynb이 CIFAR-10 데이터셋을 이용해 티스트를 진행한 주피터 파일입니다.
2. 해당 주피터 파일을 차례로 실행하면 됩니다.
3. 데이터셋의 경우 세 번째 셀에서 테스트를 진행할 CIFAR-10 데이터셋을 다운로드합니다.
4. 이후, 기본적인 하이퍼 파라미터 설정, train/test 데이터셋 구분, 모델을 로드합니다. 이때, 모델은 기본적인 SwinTransformer를 사용하였습니다.
5. 계속 셀을 실행하면 모델의 학습 단계를 수행합니다.


## 주의 및 참고 사항
- 현재 기존의 원본 GitHub 코드에서 수정한 부분이 존재합니다. 최대한 원본 코드를 살릴려고 하였지만 달라진 부분이 있으니 사용 시 주의해주세요.
- 현재 CIFAR-10 데이터 셋에 대한 간단한 테스트용으로 pretrained 모델('pretrained-CIFAR10.pth')을 생성하였지만, 로드하여 테스트를 진행하는 코드는 없는 상태입니다.
    - CIFAR-10 데이터셋은 CIFAR-10은 머신러닝 및 컴퓨터 비전에서 사용되는 벤치마크 데이터셋
    - CIFAR-10 = 총 60,000장의 32x32 픽셀 컬러 이미지
    - 10개의 클래스: 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭 / 각 6,000장씩
    - 50,000 for training / 10,000 for testing
- 현재 데이터셋 CIFAR-10에 맞춰 설정한 주요 하이퍼파라미터 정보는 다음과 같습니다.
    - 이미지 크기 = 32
    - 패치 크기 = 2
    - 윈도우 크기 = 4
    - 각 스테이지 블록 수 = [2, 2, 2, 2]
    - Head 부분의 attention 수 = [2, 4, 8, 16]
    - 드롭아웃 비율 = 0.1


## Swin Transformer에 대한 간단한 설명
- Swin Transformer는 기존의 Vision Transformer처럼 NLP 분야에 주로 쓰이는 Transformer를 비전 분야에 사용하였습니다.
기본적인 이미지 분석 방법은 Vision Transformer와 같이 이미지를 픽셀로 읽지 않고, 패치 단위로 읽어 이미지 분석을 수행합니다.
하지만, 이미지 전체를 한 번에 처리하는 Vision Transformer와는 달리, Swin Transformer는 피라미드 네트워크와 같이 윈도우로 이미지를 처음에 분할하고, 각각의 윈도우에 포함된 패치끼리만 연산 및 병합해 처리합니다.
즉, Swin Transformer는 Vision Transformer와 같이 패치 단위를 적용해 이미지를 분석하지만, Swin Transformer는 이와 함께 윈도우 기법을 추가해 패치를 구분지어 병합해 나가는 과정이 추가되었습니다.


<img src='https://github.com/Paralies/Swin-CIFAR10_test/assets/69889235/17c2355c-49c3-48ec-b97d-e6c574d509d3' width="60%" height="60%">


[Image from: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).]



- 참고 사이트: https://visionhong.tistory.com/31
