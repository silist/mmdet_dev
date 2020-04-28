import argparse
import os
import json

import mmcv
import torch
import numpy as np
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from tools.eval_csp.coco import COCO
from tools.eval_csp.eval_MR_multisetup import COCOeval


def validate(annFile, dt_path):
    mean_MR = []
    for id_setup in range(0, 4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(dt_path)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_MR.append(cocoEval.summarize_nofile(id_setup))
    return mean_MR

class MultipleKVAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    def _parse_int_float_bool(self, val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format_only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=MultipleKVAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results) with the argument "--out", "--eval", "--format_only" '
         'or "--show"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    
    # cfg.data_root = '/disk1/feigao/projects/detection/dataset/citypersons/'
    # cfg.data.val.ann_file = '/disk1/feigao/projects/detection/dataset/citypersons/annotations/citypersonsval_new.json'
    # cfg.data.test.ann_file = '/disk1/feigao/projects/detection/dataset/citypersons/annotations/citypersonsval_new.json'

    ckpt = args.checkpoint
    output_save_fp = ckpt[:ckpt.rfind('.')] + '_outputs.pkl'
    if os.path.exists(output_save_fp):
        exit(0)

    print(ckpt)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()

    # Dump
    mmcv.dump(outputs, output_save_fp)
    
    # outputs = np.asarray(mmcv.load('outputs.pkl'))
    # rank=0
    if rank == 0:
        if args.out:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            # if args.eval == 'mrs':
            evaluate_mrs(args.checkpoint, outputs)
            # else:
                # dataset.evaluate(outputs, args.eval, **kwargs)


def evaluate_mrs(ckpt, outputs):
    # print('\nShape of outputs:', len(outputs), len(outputs[0]))
    # print('Calculating MRs...')

    res = []
    for _id, boxes in enumerate(outputs):
        boxes=boxes[0]
        if type(boxes) == list:
            boxes = boxes[0]
        boxes[:, [2, 3]] -= boxes[:, [0, 1]]
        if len(boxes) > 0:
            for box in boxes:
                # box[:4] = box[:4] / 0.6
                temp = dict()
                temp['image_id'] = _id+1
                temp['category_id'] = 1
                temp['bbox'] = box[:4].tolist()
                temp['score'] = float(box[4])
                res.append(temp)
    
    res_save_fp = ckpt[:ckpt.rfind('.')] + '_val_res.json'
    mrs_fp = ckpt[:ckpt.rfind('.')] + '_val_mrs.txt'

    with open(res_save_fp, 'w') as f:
        json.dump(res, f)

    val_gt_fp = r'/disk1/feigao/gits/CSP-pedestrian-detection-in-pytorch/eval_city/val_gt.json'
    
    MRs = validate(val_gt_fp, res_save_fp)

    with open(mrs_fp, 'w') as f:
        print(ckpt)
        print('[Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
            % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))
        print('[Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
            % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100), file=f)


if __name__ == '__main__':
    main()
