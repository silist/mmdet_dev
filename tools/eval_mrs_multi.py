# -*- encoding: utf-8 -*-

"""
@File    :   eval_mrs_multi.py
@Time    :   2020/04/28 04:25:12
@Author  :   silist
@Version :   1.0
@Desc    :   Eval MRs for multi checkpoints.
"""

import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"


CFG_BASE = r'/disk1/feigao/gits/mmdet_dev/configs/ttfnet_citypersons'
WORKDIR_BASE = r'/disk1/feigao/gits/mmdet_dev/work_dirs/citypersons'

# all_cfgs = ['ttfnet_r18_1x.py', 'ttfnet_r18_2x_colab.py','ttfnet_r18_2x_colab_re_3ep.py', 'ttfnet_r34_5x.py', 'ttfnet_r50_5x.py', 'ttfnet_r50_5x_sbn_dcn.py', 'ttfnet_r50_5x_sbn_gc.py', 'ttfnet_r50_6x_dcn.py', 'ttfnet_r50_rescale_6x.py', 'ttfnet_d53_2x_resume.py', 'ttfnet_d53_2x_5x.py', 'ttfnet_dla34_5x.py']
# all_workdirs = ['ttfnet18_1x_cls_2', 'ttfnet18_2x_colab', 'ttfnet_r18_2x_colab_re_3ep', 'ttfnet_r34_5x', 'ttfnet_r50_5x', 'ttfnet_r50_5x_sbn_dcn', 'ttfnet_r50_5x_sbn_gc', 'ttfnet_r50_6x_dcn', 'ttfnet_r50_rescale_5x', 'ttfnet_d53_2x', 'ttfnet_d53_2x_5x', 'ttfnet_dla34_3x']

all_cfgs = ['ttfnet_r50_5x.py', 'ttfnet_r50_5x_sbn_dcn.py', 'ttfnet_r50_5x_sbn_gc.py', 'ttfnet_r50_6x_dcn.py', 'ttfnet_r50_rescale_6x.py', 'ttfnet_d53_2x_resume.py', 'ttfnet_d53_2x_5x.py', 'ttfnet_dla34_5x.py']
all_workdirs = ['ttfnet_r50_5x', 'ttfnet_r50_5x_sbn_dcn', 'ttfnet_r50_5x_sbn_gc', 'ttfnet_r50_6x_dcn', 'ttfnet_r50_rescale_5x', 'ttfnet_d53_2x', 'ttfnet_d53_2x_5x', 'ttfnet_dla34_3x']


def get_epoch_num(epoch_fp):
    if 'latest' in epoch_fp:
        return 999
    return int(epoch_fp[epoch_fp.rfind('_')+1: epoch_fp.rfind('.')])

def get_workflow():
    assert len(all_cfgs) == len(all_workdirs)
    workflow = []
    all_ckpt = []
    for i in range(len(all_cfgs)):
        cfg = os.path.join(CFG_BASE, all_cfgs[i])
        workdir = os.path.join(WORKDIR_BASE, all_workdirs[i])
        ckpts = list(filter(lambda i: '.pth' in i and 'latest' not in i, list(os.listdir(workdir))))
        ckpts = [os.path.join(workdir, i) for i in ckpts]
        # Sort
        # ckpts.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        ckpts.sort(key=get_epoch_num, reverse=True)
        all_ckpt.append(ckpts)
    for i in range(len(all_cfgs)):
        cfg = os.path.join(CFG_BASE, all_cfgs[i])
        first_part = 2
        last_part = 4
        for ckpt in all_ckpt[i][:first_part]:
            workflow.append((cfg, ckpt))
        for ckpt in all_ckpt[i][first_part: last_part]:
            workflow.append((cfg, ckpt))
    return workflow


if __name__ == "__main__":
    # Test
    # from datetime import datetime
    # dt = datetime.now()
    # print(dt.strftime( '%Y-%m-%d %H:%M:%S %f' )  )
    # ckpt_fp = r'/disk1/feigao/gits/mmdet_dev/work_dirs/citypersons/ttfnet_r50_5x/latest.pth'
    # cfg_fp = r'/disk1/feigao/gits/mmdet_dev/configs/ttfnet_citypersons/ttfnet_r50_5x.py'

    # # Eval
    # subprocess.call(['python', 'tools/test_citypersons.py', cfg_fp, ckpt_fp, '--eval', 'mrs'])

    wf = get_workflow()

    # for cfg, ckpt in wf:
    #     print(ckpt)

    for cfg, ckpt in wf:
        # subprocess.call(['python', 'tools/test_citypersons.py', cfg, ckpt, '--eval', 'mrs'])
        # DIST
        subprocess.call(['python', '-m', 'torch.distributed.launch', '--master_port=28895', 'tools/test_citypersons.py', cfg, ckpt, '--eval', 'mrs', '--launch', 'pytorch'])

