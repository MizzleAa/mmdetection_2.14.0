from mmdet.apis import (inference_detector,
                        init_detector, show_result_pyplot)


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
            결과 저장(score)
        '''
        model.show_result(args["input"], predict, out_file=args["output"])

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
            "score_thr":0.3,
        },
        {
            "input":"./inputs/sungangE.jpg",
            "output":"./outputs/sungangE.jpg",
            "score_thr":0.3,
        },
    ]

    model = predict.load(model_args)

    for arg in image_args:
        result = predict.action(model, arg)
        #predict.show(model,arg,result)
        predict.save(model,arg,result)

if __name__ == '__main__':
    main()
