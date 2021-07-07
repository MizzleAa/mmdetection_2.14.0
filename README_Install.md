### 1. 설치 방법 
  
python -m venv venv  
venv\Scripts\activate  
python -m pip install --upgrade pip  
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html  
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html  
pip install requirements.txt  
pip install mmdet==2.14.0  
  
### 2. pth 다운로드  

https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

### 3. 실행 예시
  
python demo/image_demo.py demo/demo.jpg configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

### 4. Documents

https://mmdetection.readthedocs.io/en/latest/api.html  
