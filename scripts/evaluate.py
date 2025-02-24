from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json, sys

def evaluate(ground_truth_json, prediction_json, save_file = None):

    coco = COCO(ground_truth_json)
    coco_result = coco.loadRes(prediction_json)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    if save_file:
        with open(save_file, 'w') as f:
            json.dump(coco_eval.eval, f)


if __name__ == '__main__':
    predictions_json = sys.argv[1]
    evaluate('/3d_data/datasets/coco/annotations/captions_val2014.json', predictions_json, '/3d_data/retreiver/base/eval_scores.json')
    