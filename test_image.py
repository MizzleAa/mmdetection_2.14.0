from mmdet.apis import (inference_detector,
                        init_detector, show_result_pyplot)

import mmcv
import numpy as np
import torch
class Predict:
    def load(self, args):
        '''
            정의된 model 및 dict load
        '''
        result = init_detector(args["config"], args["checkpoint"], device=args["device"])
        return result

    def action(self, model, args):
        '''
            예측 수행
        '''
        result = inference_detector(model, args["input"])

        return result

    def show(self, model, args, predict):
        '''
            결과 확인(원본)
        '''
        show_result_pyplot(model, args["input"], predict, args["score_thr"])

    def save(self, model, args, predict):
        '''
            결과 저장(score보다 클때)
        '''
        model.show_result(args["input"], predict, out_file=args["output"], score_thr=args["score_thr"])

    def json(self, model, args):
        '''
            임계치 값 기준 
        '''
        img = args["input"]
        result = args["predict"]
        score_thr = args["score_thr"]

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        result = []
        
        for box,label in zip(bboxes,labels):
            label_text = model.CLASSES[label]
            bbox = box.astype(np.float32)
            box = bbox[0:4]
            score = bbox[4]

            if score > score_thr:
                result.append(
                    {
                        "key":label,
                        "label":label_text,
                        "box":box.astype(np.int16).tolist(),
                        "score":score*100
                    }
                )

        return result

def main():
    predict = Predict()

    model_args = {
        "device":"cuda:0",
        "config":"./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
        "checkpoint":"./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    }

    image_args = [
        {
            "input":"./inputs/norangE.jpg",
            "output":"./outputs/norangE.jpg",
            "score_thr":0.8,
        },
        {
            "input":"./inputs/sungangE.jpg",
            "output":"./outputs/sungangE.jpg",
            "score_thr":0.8,
        },
    ]

    model = predict.load(model_args)

    for arg in image_args:
        result = predict.action(model, arg)
        
        #predict.show(model,arg,result)
        arg["predict"] = result
        data = predict.json(model, arg)
        print(data)
        predict.save(model, arg, result)

if __name__ == '__main__':
    main()
