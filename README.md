# YOLO V8을 이용한 마인크래프트 PVP 봇 만들기

- Real Time Object Detection으로 마인크래프트 게임 화면 속에서 플레이어를 감지한 뒤, 감지한 플레이어를 추적하여 공격하는 작업을 수행하는 모델을 제작한다. 결투에서 승리하는 것이 목적이다. 

## Members
- 한건희(2024078868), 컴퓨터소프트웨어학부 1학년, geonhee625@gmail.com / paulgh625@hanyang.ac.kr
- 황정민(2024009889), 컴퓨터소프트웨어학부 1학년, barwui_min@naver.com / sosonamu@hanyang.ac.kr

## I. Proposal
### - Motivation: Why are you doing this?


▶ Computer Vision 분야에 관심이 있고 그 중에서도 Real Time Object Detection과 Image Classification에 관심이 있어, 이를 사람들이 잘 알고 있는 '마인크래프트'와 접목하여 재미있고 흥미로운 프로젝트를 만들고 싶었다.


AI의 재미있는 점은 학습을 적절히 진행하면 인간을 뛰어넘을 정도의 성능을 발휘할 수 있다는 점이다.


평소 알고있던 게임에 인공지능을 적용시켜 플레이하면 어떤 일이 일어날지 궁금하였고, 실제로 맞붙어 보았을 때 인간을 상대로 압도적인 성능을 발휘할 수 있을지가 궁금하여 이 프로젝트를 진행하였다.

<br>

### - What do you want to see at the end?


▶ 인간 플레이어보다 훨씬 빠른 탐지능력으로 정확한 공격을 하여 이기는 모델을 훈련시키고 훈련시킨 모델이 조작하는 플레이어와 직접 대결하는 것이다.


## II. Datasets
- 많은 데이터를 직접 게임을 하면서 하나하나 캡쳐하여 레이블링 하는 것은 너무 힘들기 때문에 데이터셋을 공유해주는 사이트(<https://universe.roboflow.com>)에서 마인크래프트 속 플레이어 외 21종의 엔티티의 레이블링된 사진 데이터를 가져와 사용하였다.

- 데이터는 총 2158개의 사진 파일로 이루어져 있으며, 분류하는 Class는 Player를 포함하여 22개이다.


![App Screenshot](/imagesDOCU/data_num.png)


- train용 데이터는 996개, validation용 데이터는 208개, test용 데이터는 54개이다. (각각, 79% | 17% | 4% 이다)


<img src = "/imagesDOCU/train용_데이터.png" width="70%" height="70%">
<img src = "/imagesDOCU/valid용_데이터.png" width="70%" height="70%">
<img src = "/imagesDOCU/test용_데이터.pngg" width="70%" height="70%">


- 각각의 데이터들은 다음과 같이 라벨링 할 수 있으며, 라벨링을 완료하면 각 사진 파일과 같은 이름을 가진 라벨링 txt파일이 labels 폴더와 함께 생성된다. 해당 txt파일 안에는 각각의 사진 파일에서 라벨링된 사각형의 꼭짓점 값을 저장하고 있으며, 이 텍스트 파일은 나중에 모델을 학습할 때 사용된다.

<img src = "/imagesDOCU/player_1.png" width="50%" height="50%">
<img src = "/imagesDOCU/player_2.png" width="50%" height="50%">


## III. Methodology 
- Explaining your choice of algorithms (methods)
- 이 프로젝트에서 Object Detection을 주로 사용하였다.
- Object Detection은 이미지를 입력받은 물체가 있는 영역의 위치를 Bounding Box로 표시한 후, Bounding Box 내에 존재하는 물체를 Label로 분류하여 이미지 내 물체의 위치와 종류를 찾아내는 기술이다.
- YOLOv8을 사용하였다. YOLO(You Only Look Once)라는 이름에서 알 수 있듯, Overfeat, FPN등과는 다른 이미지 검출 모델과는 다르게 이미지를 한 번만 보고 물체를 판단한다는 특징이 있는데, 이는 동영상과 같은 실시간 Object Detection을 수행하기에 적합하다.
- YOLO의 구조는 다음과 같다. (https://github.com/ultralytics/ultralytics/issues/189)
<img src = "/imagesDOCU/YOLO.jpg" width="60%" height="60%">
- 기본적으로 YOLO는 CNN 모델을 기반으로 feature를 추출하는데, 처음 Input 이미지를 7x7 Grid Cell로 나눈 뒤 각 Grid Cell별로 2개의 Bounding Box를 예측하게 된다. 그럼 결과적으로 한장의 이미지에 98개의 Bounding Box를 예측하게 되고, 마지막으로 NMS를 통해 최종적으로 확률이 높은 예측 결과를 남겨 Label화 시킨다.
-NMS(Non-Maximum Suppression) : 이미지가 Object Detection 알고리즘을 거치면 각 Bounding Box에 어떤 물체일 확률값, Score를 가지는데, NMS 알고리즘을 통해 한 오브젝트에서 가장 Score가 높은 박스를 제외한 박스를 제거하는 알고리즘이 NMS이다.


 참고 
-(https://brunch.co.kr/@aischool/11)
-(https://ctkim.tistory.com/entry/Non-maximum-Suppression-NMS)
- Explaining features (if any)

## IV. Training A Model
### - Trial 1
- 모델 학습을 더 좋은 환경에서 하기 위해 구글 코랩에서 GPU를 빌려 진행함.
```
!nvidia-smi
```
![App Screenshot](/imagesDOCU/사양.png)
- 필요한 모듈 import
```
cvzone==1.6.1
ultralytics==8.2.18
opencv-python==4.9.0.80
numpy=1.26.4
hydra-core>=1.2.0
matplotlib>=3.2.2
numpy>=1.18.5
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.64.0
filterpy==1.4.5
scikit-image==0.19.3
lap==0.4.0
```
- 학습 시작
```
!yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/Datasets/PlayerDetector/data.yaml epochs=30 imgsz=640
```
> model은 large 모델을 사용하였으며, 이미지 크기는 640*640으로 설정하여 학습하였다.

<br>

- 학습이 완료된 .pt파일의 이름을 MCPVPAI_Large.pt로 바꿔준 뒤 opencv 모듈을 이용하여 detect된 구역을 직사각형으로 표시하여 화면에 띄우는 프로그램을 작성한다.
```python
from ultralytics import YOLO
import mss
import cv2
import cvzone
import math
import numpy as np

model = YOLO("model/MCPVPAI_Large.pt")

classNames = ['Bee', 'Cave Spider', 'Chest', 'Cow', 'Creeper', 'Dolphin', 'Enderman', 'Goat', 'Iron Golem', 'Llama',
               'Panda', 'Pig', 'Piglin', 'Player', 'Polar Bear', 'Sheep', 'Spider', 'Trader Lama', 'Villager House',
                 'Villager', 'Wolf', 'Zombified Piglin']

with mss.mss() as sct:
    monitor = {"top": 200, "left": 1000, "width": width, "height": height}

    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf * 100}%', (max(0, x1), max(35, y1)), scale = 1, thickness = 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('s'):
            break
```
>  화면 캡쳐: mss
>  <br>
>  이미지 편집 및 출력: cv2, cvzone

<br>

```python
classNames = ['Bee', 'Cave Spider', 'Chest', 'Cow', 'Creeper', 'Dolphin', 'Enderman', 'Goat', 'Iron Golem', 'Llama',
               'Panda', 'Pig', 'Piglin', 'Player', 'Polar Bear', 'Sheep', 'Spider', 'Trader Lama', 'Villager House',
                 'Villager', 'Wolf', 'Zombified Piglin']
```
- classNames에서 알 수 있듯이 학습한 모델은 Player 말고도 다른 객체들도 detect 할 수 있다. 다른 객체들도 데이터 셋에 포함시켜 학습한 이유는, 플레이어와 전투를 벌일 때 다른 오브젝트가 화면에 감지 되어도 이를 Player로 감지하여 공격하는 현상을 방지하기 위함이다. Player로 인식되는 객체에게만 공격을 이어나갈 수 있도록 다른 오브젝트를 구분할 수 있도록 하였다.

---

### [Result 1]
![result](/imagesDOCU/large.gif)

#### - 문제점
- 플레이어가 화면을 인식하고 Player를 찾아내는 작업은 수행하지만, 처리 시간이 늦다.
- 처리 시간이 충분히 빠르지 않으면 그로 인한 지연시간 동안은 상대를 찾아 공격하지 못할 뿐만 아니라 상대에게 피할 시간을 주어 패배할 것이다.

#### - 수정해야 할 점
- 모델 처리 속도를 높여 지연 시간을 줄인다.
```
!pip install opencv-python
```
- 위의 명령어로 opencv를 설치하면 일반적으로 CPU밖에 사용하지 못한다.
- 더 빠른 처리 속도를 위해선 GPU를 사용해야하고, 이를 위해선 opencv 프로젝트를 다운받아 이를 GPU로 따로 빌드하여 컴퓨터에 적용하여야 한다.
- GPU 빌드 방법은 영상을 참고하여 설치 하였다. (<https://www.youtube.com/watch?v=Gfl6EyIhFvM>)

---


### - Trial 2
- 두번째 학습
```
!yolo task=detect mode=train model=yolov8n.pt data=../content/drive/MyDrive/Datasets/PlayerDetector/data.yaml epochs=30 imgsz=320
```
> model은 nano 모델을 사용하였으며, 이미지 크기는 320*320으로 설정하여 학습하였다.
1. Large 모델보다 더 가벼운 nano 모델을 사용해서 학습하여 처리 시간을 줄인다.
2. 이미지 크기를 640 * 640에서 320* 320으로 바꿔 모델의 계산량을 줄여 처리 시간을 줄인다.

---

#### - 수정된 코드
```python
from ultralytics import YOLO
import mss
import cv2
import cvzone
import math
import numpy as np

model = YOLO("model/MCPVPAI_Nano.pt")

classNames = ['Bee', 'Cave Spider', 'Chest', 'Cow', 'Creeper', 'Dolphin', 'Enderman', 'Goat', 'Iron Golem', 'Llama',
               'Panda', 'Pig', 'Piglin', 'Player', 'Polar Bear', 'Sheep', 'Spider', 'Trader Lama', 'Villager House',
                 'Villager', 'Wolf', 'Zombified Piglin']

with mss.mss() as sct:
    monitor = {"top": 200, "left": 1000, "width": width, "height": height}

    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf * 100}%', (max(0, x1), max(35, y1)), scale = 1, thickness = 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('s'):
            break
```

---

### [Result 2]

![result](/imagesDOCU/nano.gif)

- 전보다 훨씬 처리 속도가 빨라져 지연 시간이 확연히 줄어들었다.



## V. Evaluation & Analysis
- Graphs, tables, any statistics (if any)

## VI. Related Work (e.g., existing studies)
- Tools, libraries, blogs, or any documentation that you have used to do this project.

## VII. Conclusion: Discussion


[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgeonheegit%2FAI-DeepLearning&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://github.com/geonheegit)
