# 1. Competition Summary
![image](https://user-images.githubusercontent.com/75927764/125191787-2c7c4e00-e27f-11eb-8e10-44c9506fb89b.png)
</br>(위 사진은 예시입니다. 대회에서는 아시아인 실물 사진을 사용하였습니다.)
</br>
- 마스크 착용 여부, 성별, 나이를 고려한 18개 클래스를 분류하는 문제
![image](https://user-images.githubusercontent.com/75927764/125191608-4cf7d880-e27e-11eb-85de-74919af6defa.png)
- 전체 사람 명 수 : 4,500   (train : test = 60 : 40)
- 한 사람당 사진의 개수: 7 (마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장)
- 이미지 크기: (384, 512)


### Issue
- 심한 클래스 불균형
- 예측 난이도 : 마스크 착용 여부 <<<<< 성별 분류 << 나이 분류
### Result :trophy:
- Accuracy : 80.9365%
- F1 score : 0.7660
- 최종 순위 : **6 / 200**
- 총 제출횟수 : 63회
</br></br>

# 2. Work Flow 🏃
### Day01                                    (67.98%, 0.53) : accuracy, f1 score

- baseline 코드 작성 및 제출
- 간단한 EDA

### Day02                                    (74.25%, 0.68)

- Data Augmentation : 효과 좋은 기법 찾기
- 남자 이미지만 오버샘플링
- pretrained 모델 사용

### Day03                                    (74.25%, 0.68)

- face detection crop
- 사람을 기준으로 validation 분리
- 랜덤 오버샘플링

### Day04, 05                               (77.50%, 0.7039)

- CenterCrop(400,200) 적용
- 다양한 Loss 적용 : FocalLoss, F1Loss, Labelsmoothing, weight 추가
- LR scheduler 적용 : CosineAnnealing, StepLR
- Weight Freeze 시도
- gradient accumulation 및 mixed precision 적용

### Day06                                    (79.66%, 0.7568)

- 배치사이즈 탐색
- optimizer 탐색
- labeling 방법 수정 (age filter 60→58)

### Day07                                    (79.66%, 0.7568)

- 마스크 착용 이미지 5장 중 랜덤으로 선택 (언더샘플링)
- multi output classification 시도
- soft loss 구현

### Day08                                    (79.66%, 0.7568)

- mixup 시도
- 모델 두개로 분리 (mask분류모델과 age,gender 분류모델) - 각각 다른 augmentation 적용
- AutoAugmentation 시도

### Day09                                    (80.04%, 0.7629)

- augmentation 단순화 : CenterCrop, 밝기조정, normalize만 적용
- valid 데이터를 다시 라벨 비율에 맞춰 split
- 싱글모델 KFold
- Self-Training 시도

### Day10                                    (80.04%, 0.7629)

- 싱글모델 KFold
- 2~3개 모델 앙상블

### Day11                                     (80.88%, 0.7735)

- Loss 재정의 : F1 Loss * 0.4 + Focal Loss * 0.6
- Hard voting, Soft voting 앙상블

</br></br>
# 3. Final Submission :triangular_flag_on_post:
- Architecture : backbone(torchvision.models.resnext50_32x4d) + Linear(128) + classifier
- Augmentation : Brightness, CenterCrop
- LossFunction : F1Loss0.4 + FocalLoss0.6
- etc) RandomWeightedSampler, WeightedLoss, Improving Label at Train Time(Age 60→58)
</br></br></br>

# 4. Experiments :chart_with_upwards_trend:
### DetectionCrop

이미지를 원본(384x512) 그대로 사용했을 때 마스크 착용여부는 99% 수준의 성능으로 매우 잘 구분하지만,</br>
나이와 성별을 예측하는 것은 매우 어려운 문제였습니다(70~80% 수준의 성능).</br>

사람의 나이, 성별을 예측하기 위한 특징으로는 머리 길이, 목주름 등이 있는데</br>
그 때문에 얼굴 주변에 집중해야 할 필요성을 느꼈습니다.

### weighted Sampler, weighted Loss 사용

데이터 불균형 문제를 해소하기 위해, 학습시 각 Class별 데이터 비율에 역수를 취하여 가중치로 사용하였습니다.


### labeling 방법 수정 (age filter 60→58)

60세 이상 데이터가 매우 적고, 60세 초과는 존재하지 않았습니다.</br>
사람 얼굴 이미지는 55세나 60세나 비슷한 특징을 가진다고 생각하여</br>
55세부터 적용해봤으나 오히려 성능 하락을 겪었고, 58세 filtering에서 성능 향상을 보였습니다.

### ArcFaceLoss

임베딩 스페이스에서 클래스 간 거리를 넓히기 위해 사용하였습니다.</br>
기존 Softmax에 비해 분류 성능이 더 올라갔지만, 결과적으로 F1 Loss + Focal Loss의 성능이 더 좋았습니다.

### F1 Loss + Focal Loss

각각 따로 사용할 생각만 했었는데, 합쳐서 사용해보니 성능이 올라갔습니다.
대회 랭킹 기준이 F1 score라서 F1 Loss를 더해줬습니다.
Loss별 가중 비율은 4:6으로 줬는데, 마지막 날이라 제출 횟수가 부족해서 더 실험은 못해봤습니다.

### Mixed Precision (FP16 Train)

학습시 텐서를 16비트 크기로 사용하여 연산량과 메모리 사용량을 줄이는 방법</br>
training 및 inference 속도가 약 20%정도 향상됐습니다.</br>
학습 시간이 오래걸린다고 판단하여 적용해봤습니다. 의외로 성능이 그대로 유지되어 계속 사용했는데, 이유는 파악하지 못했습니다.

### 오버샘플링

데이터 불균형 문제를 해결하기 위해 시도해 봤으나, 오히려 오버피팅이 너무 심해졌습니다.</br>
남성 이미지만 늘려도보고, 부족한 라벨만 골라서 늘려봤지만 모두 실패했습니다.

### 마스크 착용 이미지 랜덤 선택 (언더샘플링)

사람마다 7장씩 이미지가 있는데, 그 중 5장이 마스크를 착용한 이미지여서 데이터 불균형이 심했습니다.</br>
해결하기 위해 5장 중 n장을 랜덤 선택하여 학습데이터로 사용했습니다.

실패한 이유를 생각해봤는데, </br>
기존처럼 대부분의 데이터가 마스크로 가려져있는 경우 눈가나 머리카락 등을 기준으로 나이, 성별을 예측할 것입니다.</br>
마스크로 가려진 경우가 상대적으로 줄어들게 되면 '나이,성별 구분'의 기준이 되는 feature 선택이 매우 힘들어지게 될 것 같습니다.</br>
여기에 더해 '마스크 착용 여부 구분'의 성능까지 줄어들테니 매우 좋지 않은 방법이었다고 결론지었습니다.

### Soft Loss

onehot인코딩을 하지않고 loss를 계산했습니다. </br>
첫번째 이외의 후보군까지 고려하니까 당연히 더 잘 학습할 줄 알았으나 오히려 성능 하락을 겪었습니다.

### LabelSmoothingLoss

Soft Loss와 비슷한 맥락에 smoothing을 추가해봤습니다. </br>
smoothing 비율을 0.6부터 1.0까지 다양하게 적용해 봤지만 적절한 비율을 찾지 못한채 대회가 마감되었습니다.

### multi output classification (3+2+3)

모델 아키텍쳐의 head 출력 벡터를 18차원이 아니라 8차원으로 줄이는 방법을 사용해봤습니다.</br>
성능은 18 head 보다 조금 떨어지지만 거의 비슷했습니다.</br>
이제 떠올랐는데, 여기에 soft loss를 함께 적용해봤으면 어땠을까 생각해봅니다.</br>

### 

마스크 분류모델과 나이,성별 분류모델 두가지로 나눠보았습니다.</br>
역시나 마스크 분류 성능은 매우 좋지만 나이,성별 분류 성능은 매우 낮았습니다.</br>
단일 모델로 했을 때와 성능 차이가 없는데 학습 및 추론 속도만 오래걸려서 그만둔 방법입니다.</br>

이후 생각해보니 모델을 분리하는 것의 가장 큰 장점은 각 task마다 다른 분포를 가진 데이터로 학습시킬 수 있다는 점인 것 같습니다.</br>
마스크 분류모델은 얼굴만 crop한 데이터로도 가능할 것 같고 나이,성별 분류모델은 마스크를 제대로 착용한 이미지만으로 학습시키면 괜찮을 것 같다고 생각합니다.</br>
</br>
_(추가)_ 대회를 마치고, 위에서 생각했던 방식대로 각각 다른 형태의 데이터들로 학습시켜 보니 조금의 성능 향상이 있었습니다.</br>
