# YOLO V8을 이용한 마인크래프트 PVP 봇 만들기

- Real Time Object Detection으로 마인크래프트 게임 화면 속에서 플레이어를 감지한 뒤, 감지한 플레이어를 추적하여 공격하는 작업을 수행하는 모델을 제작한다. 결투에서 승리하는 것이 목적이다. 

## Members
- 한건희(2024078868), 컴퓨터소프트웨어학부 1학년, geonhee625@gmail.com / paulgh625@hanyang.ac.kr
- 황정민(2024009889), 컴퓨터소프트웨어학부 1학년, barwui_min@naver.com / sosonamu@hanyang.ac.kr

## I. Proposal
- Motivation: Why are you doing this?
▶ Computer Vision 분야에 관심이 있고, 그 중에서도 Real Time Object Detection과 Image Classification에 관심이 있어, 이를 사람들이 잘 알고 있는 '마인크래프트'와 접목하여 재미있고 흥미로운 프로젝트를 만들고 싶었다. AI의 재미있는 점은 학습을 적절히 진행하면 인간을 뛰어넘을 정도의 성능을 발휘할 수 있다는 점이다. 평소 알고있던 게임에 인공지능을 적용시켜 플레이하면 어떤 일이 일어날지 궁금하였고, 실제로 맞붙어 보았을 때 인간을 상대로 압도적인 성능을 발휘할 수 있을지가 궁금하여 이 프로젝트를 진행하였다.

- What do you want to see at the end?
▶ 인간 플레이어보다 훨씬 빠른 탐지능력으로 정확한 공격을 하여 이기는 모델을 훈련시키고 훈련시킨 모델이 조작하는 플레이어와 직접 대결하는 것이다.

## II. Datasets
- 많은 데이터를 직접 게임을 하면서 하나하나 캡쳐하여 레이블링 하는 것은 너무 힘들기 때문에 데이터셋을 공유해주는 사이트(<https://universe.roboflow.com>)에서 마인크래프트 속 플레이어 외 21종의 엔티티의 레이블링된 사진 데이터를 가져와 사용하였다.

- 데이터는 총 2158개의 사진 파일로 이루어져 있으며, 분류하는 Class는 Player를 포함하여 22개이다.
![App Screenshot](/데이터셋_개수.png)
- train용 데이터는 996개, validation용 데이터는 208개, test용 데이터는 54개이다. (각각, 79% | 17% | 4% 이다)
![App Screenshot](/train용_데이터.png)
![App Screenshot](/valid용_데이터.png)
![App Screenshot](/test용_데이터.png)

## III. Methodology 
- Explaining your choice of algorithms (methods)
- Explaining features (if any)

## IV. Training A Model
- 모델 학습을 더 좋은 환경에서 하기 위해 구글 코랩에서 GPU를 빌려 진행함
```
!nvidia-smi
```
![App Screenshot](/사양.png)
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

## V. Evaluation & Analysis
- Graphs, tables, any statistics (if any)

## VI. Related Work (e.g., existing studies)
- Tools, libraries, blogs, or any documentation that you have used to do this project.

## VII. Conclusion: Discussion
