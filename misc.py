import signal
import logging
from collections import defaultdict
from collections import deque
import numpy as np
import torch


# https://stackoverflow.com/a/21919644/805502
class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


# https://stackoverflow.com/a/25294767/805502
def tuplify(listything):
    if isinstance(listything, list): return tuple(map(tuplify, listything))
    if isinstance(listything, dict): return {k:tuplify(v) for k,v in listything.items()}
    return listything


class SWDict(dict):
    """
    Single-write dict. Useful for making sure no inference is computed twice.
    """
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError('key', key, 'already set')
        super().__setitem__(key, value)


class SWDefaultDict(defaultdict):
    """
    Single-write defaultdict.
    """
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError('key', key, 'already set')
        super().__setitem__(key, value)


class MovingAverageMeter(object):
    def __init__(self, window):
        self.window = window
        self.reset()

    def reset(self):
        self.history = deque()
        self.avg = 0
        self.sum = None
        self.val = None

    @property
    def count(self):
        return len(self.history)

    @property
    def isfull(self):
        return len(self.history) == self.window

    def __getstate__(self):
        state = self.__dict__.copy()
        state['history'] = np.array(state['history'])
        return state

    def __setstate__(self, state):
        state['history'] = deque(state['history'])
        self.__dict__.update(state)

    def update(self, val, epoch, iteration):
        self.history.append(val)
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val
        if len(self.history) > self.window:
            self.sum -= self.history.popleft()
        self.val = val
        self.avg = self.sum / self.count

    def __repr__(self):
        return "<MovingAverageMeter of window {} with {} elements, val {}, avg {}>".format(
            self.window, self.count, self.val, self.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), return_correct_k=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_ks = []

    for k in topk:
        correct_k = correct[:k].float().sum(0)
        res.append(correct_k.sum().mul_(100.0 / batch_size))
        correct_ks.append(correct_k)
    if return_correct_k:
        return res, correct_ks
    return res


def soft_cross_entropy(output, target):
    """
    For knowledge distillation in self-distillation
    """
    output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
    target = target.unsqueeze(1)
    output_log_prob = output_log_prob.unsqueeze(2)
    cross_entropy_loss = -torch.bmm(target, output_log_prob).view(output.size(0))
    return cross_entropy_loss