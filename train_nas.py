import argparse
import os
import os.path as osp
import sys
import time
import logging
import warnings
from collections import defaultdict
from glob import glob

import torch
import numpy as np
import pickle

from torch import nn
from torch.utils import data
from viterbi import maxsum, sumprod_log, complete, score
from torchvision import datasets
from torchvision import transforms
from types import SimpleNamespace

import latency
import misc
from model import SlimMobilenet


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


parser = argparse.ArgumentParser(description='Train mobilenet slimmable/AOWS model')
parser.add_argument('--data', metavar='DIR', default='/imagenet',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no-cuda', action="store_true")
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-last', action='store_true',
                    help='resume to last checkpoint if found (supersedes resume)')
parser.add_argument("--debug", nargs="*", choices=["mini"], default="none",
                    help="do a mini epoch to check everything works.")
parser.add_argument('--max-width', type=float, default=1.5)
parser.add_argument('--min-width', type=float, default=0.2)
parser.add_argument('--levels', type=int, default=14)
parser.add_argument('--latency-target', type=float, default=0.04, help="latency target in objective")
parser.add_argument('--window', type=int, default=100000,
                    help="size of window over which the moving average of losses is computed in OWS and AOWS.")
parser.add_argument('--latency-model', type=str, default='output/model_trt16_K100.0.jsonl', help="latency model")
parser.add_argument('--gamma-iter', type=int, default=12, help="Number of Viterbi iterations to set gamma.")
parser.add_argument('--AOWS', action="store_true", help="use AOWS")
parser.add_argument('--AOWS-warmup', type=int, default=5, help="AOWS warmup epochs")
parser.add_argument('--AOWS-min-temp', type=float, default=0.0005, help="minimum (final) temperature")
parser.add_argument('--expname', default='output/nas_output')


def main():
    args = parser.parse_args()

    logger.info("=> creating model")
    model = SlimMobilenet(min_width=args.min_width, max_width=args.max_width, levels=args.levels)
    logger.info(model)
    if not args.no_cuda:
        model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss(reduction='none')
    if not args.no_cuda:
        criterion = criterion.cuda()

    args.lr = args.lr * args.batch_size / 256
    logger.info("learning rate scaling: using lr={}".format(args.lr))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    torch.backends.cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augment = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), ]

    dataset_train = datasets.ImageFolder(os.path.join(args.data, 'train'),
                                         transforms.Compose(augment + [transforms.ToTensor(), normalize]))
    train_loader = data.DataLoader(dataset_train,
                                   batch_size=args.batch_size,
                                   num_workers=args.workers, pin_memory=True)

    start_epoch = 0
    ows_state = SimpleNamespace()

    filters = model.filters if hasattr(model, 'filters') else model.module.filters
    ows_state.histories = [{c: misc.MovingAverageMeter(args.window) for c in F.configurations} for F in filters]
    ows_state.latency = dict(latency.load_model(args.latency_model))

    if args.resume_last:
        avail = glob(osp.join(args.expname, 'checkpoint*.pth'))
        avail = [(int(f[-len('.pth') - 3:-len('.pth')]), f) for f in avail]
        avail = sorted(avail)
        if avail:
            args.resume = avail[-1][1]
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            key = next(iter(state_dict.keys()))
            if key.startswith('module.') and args.no_cuda:
                state_dict = {k[len('module.'):]: v for (k, v) in state_dict.items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'ows_state' in checkpoint:
                ows_state.histories = checkpoint['ows_state'].histories
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")
            args.resume = ''

    if args.resume:
        # Solve OWS one first time in order to allow re-evaluation on the last epoch with varying latency target
        best_path, _, _, _, timing = solve_ows(model, start_epoch, len(train_loader), -1, ows_state, args, eval_only=True)
        logger.info('Evaluation from resumed checkpoint...')
        best_path_str = (f"Best configuration: {best_path}, "
                         f"predicted latency: {timing}")
        logger.info(best_path_str)

    for epoch in range(start_epoch, args.epochs):
        history = train(train_loader, model, criterion, optimizer, epoch, ows_state, args)

        logger.info(f"=> saving decision history for epoch {format(epoch + 1)}")
        decision_target = 'decision{:03d}.pkl'.format(epoch + 1)
        if args.expname:
            os.makedirs(args.expname, exist_ok=True)
            decision_target = osp.join(args.expname, decision_target)
        with open(decision_target, 'wb') as f:
            pickle.dump(history, f, protocol=4)

        logger.info(f"=> saving checkpoint for epoch {epoch + 1}")

        current_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        current_state['ows_state'] = ows_state
        best_path_str = (f"Best configuration: {history['OWS'][-1]['best_path']}, "
                         f"predicted latency: {history['OWS'][-1]['pred_latency']}")
        logger.info(best_path_str)
        with open(osp.join(args.expname, f"ows_result_{epoch + 1:03d}.txt"), 'w') as f:
            f.write(best_path_str + '\n')
        filename = save_checkpoint(current_state, args.expname)
        logger.info(f"checkpoint saved to {filename}.")



def train(train_loader, model, criterion, optimizer, epoch, ows_state, args):
    meters = defaultdict(misc.AverageMeter)

    model.train()

    filters = model.filters if hasattr(model, 'filters') else model.module.filters
    history = defaultdict(list)

    end = time.time()
    for iteration, (input, target) in enumerate(train_loader):
        if "mini" in args.debug and iteration > 20: break

        best_path, temperature, gamma_max, best_perf, timing = solve_ows(
            model, epoch, len(train_loader), iteration, ows_state, args)

        # measure data loading time
        meters["data_time"].update(time.time() - end)

        if not args.no_cuda:
            target = target.cuda(non_blocking=True)

        compute_results = misc.SWDefaultDict(misc.SWDict)

        minconf = [F.configurations[0] for F in filters]
        maxconf = [F.configurations[-1] for F in filters]

        optimizer.zero_grad()

        # sandwich rule: train maximum configuration
        outp = model(input, configuration=maxconf)
        loss = criterion(outp['x'], target)
        loss.mean().backward()
        compute_results['max']['x'] = outp['x'].detach()
        compute_results['max']['loss_numpy'] = loss.detach().cpu().numpy()
        compute_results['max']['prob'] = torch.nn.functional.softmax(compute_results['max']['x'], dim=1)

        # sandwich rule: train minimum and random configuration with self-distillation
        for kind in ('min', 'rand'):
            conf = None if kind == 'rand' else minconf
            outp = model(input, configuration=conf)

            loss = misc.soft_cross_entropy(outp['x'], compute_results['max']['prob'].detach())
            compute_results[kind]['soft_loss_numpy'] = loss.detach().cpu().numpy()
            with torch.no_grad():
                hard_loss_numpy = criterion(outp['x'], target).detach().cpu().numpy()
                compute_results[kind]['loss_numpy'] = hard_loss_numpy

            compute_results[kind]['x'] = outp['x'].detach()
            if kind == 'rand':
                compute_results['rand']['decision'] = outp['decision'].cpu().numpy()
            loss.mean().backward()

        for path, image_loss, image_refloss in zip(compute_results['rand']['decision'],
                                                    compute_results['rand']['loss_numpy'],
                                                    compute_results['max']['loss_numpy']):
            for i, pi in enumerate(path):
                ows_state.histories[i][pi].update(-(image_loss - image_refloss) / len(path), epoch, iteration)

        for refname in ('min', 'max', 'rand'):
            meters['loss_' + kind].update(compute_results[kind]['loss_numpy'].mean(), input.size(0))
            refloss = compute_results[refname]['loss_numpy']
            (prec1, prec5), refcorrect_ks = misc.accuracy(compute_results[refname]['x'].data,
                                                          target, topk=(1, 5), return_correct_k=True)
            refcorrect1, refcorrect5 = [a.cpu().numpy().astype(bool) for a in refcorrect_ks]
            history['loss_' + refname].append(refloss)
            history['top1_' + refname].append(refcorrect1)
            history['top5_' + refname].append(refcorrect5)
            meters['top1_' + refname].update(prec1.item(), input.size(0))
            meters['top5_' + refname].update(prec5.item(), input.size(0))
            if 'soft_loss_numpy' in compute_results[refname]:
                meters['loss_soft_' + kind].update(compute_results[kind]['soft_loss_numpy'].mean(), input.size(0))
                history['loss_soft_' + refname].append(compute_results[refname]['soft_loss_numpy'])

        history['configuration'].append(compute_results['rand']['decision'])
        history['configuration'].append(compute_results['rand']['loss_numpy'])

        optimizer.step()

        # measure elapsed time
        meters["batch_time"].update(time.time() - end)
        end = time.time()

        if iteration % args.print_freq == 0:
            toprint = f"Epoch: [{epoch}][{iteration}/{len(train_loader)}]\t"
            toprint += ('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Prec@1 {top1_rand.val:.3f} ({top1_rand.avg:.3f})\t'
                        'Prec@5 {top5_rand.val:.3f} ({top5_rand.avg:.3f})\t'.format(**meters))

            for key, meter in meters.items():
                if key.startswith('loss'):
                    toprint += f'{key} {meter.val:.4f} ({meter.avg:.4f})\t'
            logger.info(toprint)

            # prints a string summarizing the sampling probabilities for each filter
            probas_str = ""
            for i, F in enumerate(filters):
                if F.probability is not None:
                    probas_str += '|{} '.format(i)
                    for p in F.probability:
                        probas_str += str(int(100 * p)) + ' '
            probas_log = None
            if any(F.probability is not None for F in filters):
                probas_log = tuple(F.probability for F in filters),
            history['OWS'].append(dict(best_path=best_path, temperature=temperature, gamma_max=gamma_max,
                                        best_pref=best_perf, pred_latency=timing, probas_log=probas_log))
            if probas_str:
                probas_str = '\n' + probas_str
            ows_str = f"predicted latency: {timing}, perf: {best_perf}, T: {temperature}, gamma: {gamma_max}"
            logger.info('best_path: ' + ','.join(map(str, best_path)) + ows_str + probas_str)


    return history


def aows_temp(epoch, epoch_len, iteration, args):
    schedule = [(0, 1.0), (args.AOWS_warmup, 1.0),
                (args.AOWS_warmup + 1, 0.01),
                (10, 0.001), (args.epochs, args.AOWS_min_temp)]
    cur_phase = 0
    for iphase, (phase, _) in enumerate(schedule):
        if epoch >= phase:
            cur_phase = iphase
    phase, start_temp = schedule[cur_phase]
    if cur_phase == len(schedule) - 1:
        return start_temp
    end_phase, end_temp = schedule[cur_phase + 1]
    max_iter = epoch_len * (end_phase - phase)
    cur_iter = epoch_len * (epoch - phase) + iteration
    ratio = cur_iter / max_iter
    log_T = (1.0 - ratio) * np.log10(start_temp) + ratio * np.log10(end_temp)
    return 10 ** log_T


def solve_ows(model, epoch, len_epoch, iteration, ows_state, args, eval_only=False):
    """
    Solves OWS equation and sets AOWS probabilities when AOWS is activated.
    """
    if hasattr(model, 'module'): model = model.module

    unaries = [[0.0]] + [[M.avg for M in C.values()] for C in ows_state.histories] + [[0.0]]

    if not hasattr(ows_state, 'pairwise'):
        pairwise = []

        possible_in_channels = [3]
        possible_outputs = iter([F.configurations for F in model.filters] + [[1000]])
        for L in model.components:
            possible_out = next(possible_outputs)
            pair = np.zeros((len(possible_in_channels), len(possible_out)))
            for incoming, p in enumerate(possible_in_channels):
                for outgoing, l in enumerate(possible_out):
                    var = latency.Vartype(**L._asdict(), in_channels=p, out_channels=l)
                    pair[incoming, outgoing] = ows_state.latency[var]
            pairwise.append(pair)
            possible_in_channels = possible_out
        ows_state.pairwise = pairwise

    unaries, pairwise, states = complete(unaries, ows_state.pairwise)

    def solve(gamma):
        _, ipath = maxsum(unaries, -gamma * pairwise, states)
        perf, timing = score(ipath, unaries, pairwise, detail=True)
        return ipath, perf, timing

    gamma_min = 0.0
    gamma_max = 10.0
    timing_max = solve(gamma_max)[2]

    expanding_iterations = 0
    while timing_max > args.latency_target:
        expanding_iterations += 1
        if expanding_iterations > 2:
            logging.warning("Too many expanding loops for gamma, try adjusting gamma_max in the code")
        gamma_max *= 2
        timing_max = solve(gamma_max)[2]

    for _ in range(args.gamma_iter):
        mid_gamma = 0.5 * (gamma_min + gamma_max)
        timing_middle = solve(mid_gamma)[2]
        if timing_middle > args.latency_target:
            gamma_min = mid_gamma
        else:
            gamma_max = mid_gamma
    ipath, perf, timing = solve(gamma_max)

    T = np.inf
    if args.AOWS and epoch >= args.AOWS_warmup and not eval_only:
        T = aows_temp(epoch, len_epoch, iteration, args)
        marginals = sumprod_log(unaries / T, -gamma_max * pairwise / T, states)
        assert marginals.shape[0] == len(model.filters) + 2, "{} {}".format(marginals.shape[0], len(model.filters))
        for F, marginal in zip(model.filters, marginals[1:-1]):
            F.probability = marginal[:len(F.configurations)]

    best_path = tuple(F.configurations[i] for (F, i) in zip(model.filters, ipath[1:-1]))
    return best_path, T, gamma_max, perf, timing


def save_checkpoint(state, expname=''):
    filename = f"checkpoint{state['epoch']:03d}.pth"
    if expname:
        os.makedirs(expname, exist_ok=True)
        filename = osp.join(expname, filename)
    torch.save(state, filename)
    return filename


if __name__ == '__main__':
    logger = logging.getLogger(__file__)
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                        format='%(name)s: %(message)s')
    main()
