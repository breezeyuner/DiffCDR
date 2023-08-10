import os
import torch
import numpy as np
import random
import argparse
import json
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run


def prepare_1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ratio', default='[0.8, 0.2]')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--la_lr', type=float, default=0.001)
    parser.add_argument('--diff_lr', type=float, default=0.001)

    parser.add_argument('--root', default='./')
    parser.add_argument('--exp_part', default='None_CDR')
    parser.add_argument('--save_path', default='./model_save_default/model.pth')
    parser.add_argument('--use_cuda', default=1)
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    return args

def prepare_2(args,config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        config['base_model'] = args.base_model
        config['task'] = args.task
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
        config['la_lr'] = args.la_lr
        config['diff_lr'] = args.diff_lr

    return config


if __name__ == '__main__':
    args = prepare_1()
    
    config_path = args.root + 'config.json'

    config = prepare_2(args,config_path)
    config['root'] = args.root + 'data/'
    config['use_cuda'] = 0 if args.use_cuda =='0' else 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.process_data_mid:
        for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
            DataPreprocessingMid(config['root'], dealing).main()
    if args.process_data_ready:
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
            for task in ['1', '2', '3']:
                DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()
    print('task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};'.
          format(args.task, args.base_model, args.ratio, args.epoch, args.lr, args.gpu, args.seed))
    print('diff_steps:{};diff_sample_steps:{};diff_scale:{};diff_dim:{};diff_task_lambda:{};'.
          format(config['diff_steps'],config['diff_sample_steps'],config['diff_scale'],config['diff_dim'],config['diff_task_lambda']))

    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main(args.exp_part,args.save_path)



