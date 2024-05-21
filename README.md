
# YOLO V8을 이용한 마인크래프트 플레이어 추적

- Real Time Object Detection으로 마인크래프트 게임 화면 속에서 플레이어를 감지한 뒤, 감지한 플레이어를 추적하여 공격하는 작업을 수행하는 모델을 제작한다. 결투에서 승리하는 것이 목적이다. 

## Members
- 한건희(2024078868), 컴퓨터소프트웨어학부 1학년, geonhee625@gmail.com / paulgh625@hanyang.ac.kr
- 황정민(2024009889), 컴퓨터소프트웨어학부 1학년, barwui_min@naver.com / sosonamu@hanyang.ac.kr

## I. Proposal
- Motivation: Why are you doing this?
▶ Computer Vision 분야에 관심이 있고, 그 중에서도 Real Time Object Detection과 Image Classification에 관심이 있어 이를 접목하여 재미있고 흥미로운 프로젝트를 진행하는 과정에서 관련 지식을 얻기 위함이다.

- What do you want to see at the end?
▶ 인간 플레이어보다 훨씬 빠른 탐지능력으로 정확한 공격을 하여 이기는 모델을 훈련시키고 이를 관전하는 것이다.

## II. Datasets
Describing your dataset
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

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
