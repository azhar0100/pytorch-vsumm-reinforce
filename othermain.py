from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate
import re
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from ax import optimize
import atexit

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
import vsum_tools

from vasnet_model import VASNet

import yaml

abbrev_to_name = {
    's':'summe'  ,
    't':'tvsum'  ,
    'o':'ovp'    ,
    'y':'youtube'
}

file_dict = {
    'summe'  : 'eccv16_dataset_summe_google_pool5',
    'tvsum'  : 'eccv16_dataset_tvsum_google_pool5',
    'ovp'    : 'eccv16_dataset_ovp_google_pool5',
    'youtube': 'eccv16_dataset_youtube_google_pool5'
}

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-d', '--dataset-dir', type=str, help="path to h5 datasets (required)",default="datasets/")
parser.add_argument('-t', '--train-sets' , type=str, default="stoy", help="Which of the 4 datasets to train on")
parser.add_argument('--split-id', type=int, default=-1, help="split index (default: -1)")
parser.add_argument('-m', '--metric', type=str, choices=['tvsum', 'summe'],
                    help="evaluation metric ['tvsum', 'summe']",default='summe')

# Optimization options
parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--max-epoch', type=int, default=60, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")

# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False


def evaluate(model, dataset, test_keys, use_gpu,eval_metric=None):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        if eval_metric is None:
            eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))

    return mean_fm


def othermain():
    # args = argparse.Namespace(evaluate=False,use_gpu=True,seed=1,args.dataset)

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset_dir))

    datasets = {}
    for i in args.train_sets:
        d = {}
        d['dataset'] = h5py.File(args.dataset_dir + file_dict[abbrev_to_name[i]] + ".h5",  'r')
        dataset = d['dataset']
        d['num_videos'] = len(dataset.keys())
        d['splits'] = read_json(args.dataset_dir + file_dict[abbrev_to_name[i]] + ".json")
        assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
        d['split'] = splits[args.split_id]
        d['train_keys'] = split['train_keys']
        d['test_keys'] = split['test_keys']
        datasets[i] = d
        print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(d['train_keys']), len(d['test_keys'])))
    print("Initialize model")
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
    model = VASNet()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    datasets.append(dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        start_epoch = int(re.search(r'.*([0-9]+)\.pth\.tar.*',args.resume).group(1)) - 1

    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        evaluate(model, dataset, test_keys, use_gpu)
        return

    print("==> Start training")
    start_time = time.time()
    model.train()

    for i in np.random.shuffle(datasets.keys()):
        datasets[i]['baselines'] = {key: 0. for key in train_keys} # baseline rewards for videos
        datasets[i]['reward_writers'] = {key: [] for key in train_keys} # record reward changes for each video

        baselines      = datasets[i]['baselines']
        reward_writers = datasets[i]['reward_writers']
        train_keys = datasets[i]['train_keys']
        dataset = datasets[i]['dataset']

        @atexit.register
        def exit_func():
            elapsed = round(time.time() - start_time)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

            model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
            model_save_path = osp.join(args.save_dir, 'model_epoch' + str(args.max_epoch) + '.pth.tar')
            save_checkpoint(model_state_dict, model_save_path)
            print("Model saved to {}".format(model_save_path))

            dataset.close()


        for epoch in range(start_epoch, args.max_epoch):
            idxs = np.arange(len(train_keys))
            np.random.shuffle(idxs) # shuffle indices

            for idx in idxs:
                key = train_keys[idx]
                seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
                seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
                if use_gpu: seq = seq.cuda()
                probs = model(seq) # output shape (1, seq_len, 1)

                cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
                m = Bernoulli(probs)
                epis_rewards = []
                for _ in range(args.num_episode):
                    actions = m.sample()
                    log_probs = m.log_prob(actions)
                    reward = compute_reward(seq, actions, use_gpu=use_gpu)
                    expected_reward = log_probs.mean() * (reward - baselines[key])
                    cost -= expected_reward # minimize negative expected reward
                    epis_rewards.append(reward.item())

                optimizer.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                datasets[i]['baselines'][key] = 0.9 * datasets[i]['baselines'][key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
                datasets[i]['reward_writers'][key].append(np.mean(epis_rewards))

            # try:
            epoch_reward = np.mean([datasets[i]['reward_writers'][key][epoch] for key in train_keys])
            print("epoch {}/{}\t reward {}\t".format(epoch+1, args.max_epoch, epoch_reward))
            write_json(datasets[i]['reward_writers'], osp.join(args.save_dir, i+'-rewards.json'))
            # except:
                # print("The weird exception was encountered")
                # pass
    evaluate(model, dataset[file_dict['summe']], test_keys, use_gpu, 'avg')
    evaluate(model, dataset[file_dict['tvsum']], test_keys, use_gpu, 'max')

if __name__ == '__main__':
    othermain()
