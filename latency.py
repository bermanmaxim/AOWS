import argparse
import json
import logging
import os
import os.path as osp
import random
import sys
import time
from collections import Counter
from collections import defaultdict
from collections import namedtuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import lil_matrix
# lazy: from torch2trt import torch2trt

from misc import DelayedKeyboardInterrupt
from misc import tuplify
from model import SlimMobilenet
from model import LayerType
from viterbi import complete
from viterbi import maxsum


logger = logging.getLogger(__name__)
Vartype = namedtuple("Vartype", LayerType._fields + ('in_channels', 'out_channels'))
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate samples and fit a latency model.")

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_bench = subparsers.add_parser('benchmark', 
        help="Benchmark a single channel configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_bench.add_argument("configuration",
        help="configuration to test (comma-separated channels or MOBILENET)")

    parser_gen = subparsers.add_parser('generate', 
        help="Generate latency samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for subparser in (parser_bench, parser_gen):
        subparser.add_argument("-D", "--device", choices=["cpu", "gpu", "trt"], 
            default="gpu", help="Use GPU, CPU or TensorRT latency")
        subparser.add_argument("--dtype", choices=["fp32", "fp16"], 
            default="fp16", help="Datatype for network")
        subparser.add_argument("-B", "--batch-size", type=int, default=64, 
            help="Batch size used for profiling")
        subparser.add_argument("-I", "--iterations", type=int, default=60, 
            help="Profiling iterations")
        subparser.add_argument("-W", "--warmup", type=int, default=10, 
            help="Warmup iterations")
        subparser.add_argument("--reduction", choices=['mean', 'min'], default='mean',
                               help="Reduce timings by their mean or by their minimum (minimum can reduce variance)")
    parser_gen.add_argument("--biased", action="store_true", 
        help="Bias sampling towards missing configurations")
    parser_gen.add_argument("-N", "--count", type=int, default=8000,
        help="Minimum number of samples to generate")
    parser_gen.add_argument("-R", "--repetitions", type=int, default=0, 
        help="Minimum number of samples per choice")
    parser_gen.add_argument("--save-every", type=int, default=1,
        help="Number of inferences before saving intermediate output")
    parser_gen.add_argument("samples_file", help="Output samples file")

    parser_fit = subparsers.add_parser('fit', help="Fit a latency model")
    parser_fit.add_argument("-K", "--regularize", type=float, default=0.0,
        help="Amount of monotonicity regularization (Equation 7)")
    parser_fit.add_argument("samples_file", help="Training samples")
    parser_fit.add_argument("model_file", help="Output model file")

    parser_val = subparsers.add_parser('validate', help="Validate a latency model")
    parser_val.add_argument("samples_file", help="Validation samples")
    parser_val.add_argument("model_file", help="Model file")
    parser_val.add_argument("plot_file", help="Plot file")

    args = parser.parse_args()

    if 'configuration' in args:
        defaults = {'MOBILENET': "32,64,128,128,256,256,512,512,512,512,512,512,1024,1024"}
        if args.configuration in defaults:
            args.configuration = defaults[args.configuration]
        args.configuration = [int(''.join(ci for ci in c if ci.isdigit())) for c in args.configuration.split(',')]
    
    return args


def get_model(min_width=0.2, max_width=1.5, levels=14):
    return SlimMobilenet(min_width=min_width, max_width=max_width, levels=levels)


def benchmark(device, dtype, batch_size, iterations, warmup, reduction, configuration, silent=False):
    if device == 'cpu':
        dev = torch.device('cpu')
    elif device in ['gpu', 'trt']:
        dev = torch.device('cuda')
    fp = dict(fp16=torch.float16, fp32=torch.float32).get(dtype)
    net = SlimMobilenet.reduce(configuration).to(dev).type(fp).eval()
    x = torch.ones((batch_size, 3, 224, 224)).to(dev).type(fp)
    if device == 'trt':
        from torch2trt import torch2trt
        net = torch2trt(net, [x], fp16_mode=(dtype == 'fp16'), max_batch_size=batch_size)

    for i in range(warmup):
        outputs = net(x)
        torch.cuda.current_stream().synchronize()

    timings = []
    t0 = time.time()
    for i in range(iterations):
        outputs = net(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        timings.append(t1 - t0)
        t0 = t1

    ms = 1000.0 * getattr(np, reduction)(timings) / batch_size
    if not silent:
        print(f"{configuration}: {ms}ms")

    return ms


def gen_configuration_biased(net, repetitions):
    M = min(repetitions.values())
    unary = []
    pairwise = []
    for i, L in enumerate(net.components):
        input_choices = [net.in_channels] if i == 0 else net.configurations[i - 1]
        output_choices = ([net.out_channels] if i == len(net.components) - 1
                          else net.configurations[i])
        U = np.zeros(len(input_choices))
        P = np.zeros((len(input_choices), len(output_choices)))
        for i1, I in enumerate(input_choices):
            for i2, O in enumerate(output_choices):
                var = Vartype(**L._asdict(), in_channels=I, out_channels=O)
                P[i1, i2] = float(repetitions[var] == M)
        unary.append(U)
        pairwise.append(P)
    unary.append(np.zeros(len(output_choices)))
    un, pair, states = complete(unary, pairwise)
    iconfig = maxsum(un, pair, states, K=1)[1]
    configuration = [C[i] for (C, i) in zip(net.configurations, iconfig[1:-1])]
    return configuration


def gen_configuration(net, repetitions, biased=False):
    if biased:
        return gen_configuration_biased(net, repetitions)
    return [random.choice(conf) for conf in net.configurations]


def collect_repetitions(net, configuration=None):
    if configuration is None:
        configuration = net.configurations
    if isinstance(configuration[0], (int, np.integer)):  # single configuration
        configuration = [[c] for c in configuration]
    layertypes = Counter()
    for i, L in enumerate(net.components):
        input_choices = [net.in_channels] if i == 0 else configuration[i - 1]
        output_choices = ([net.out_channels] if i == len(net.components) - 1 
                          else configuration[i])
        for I in input_choices:
            for O in output_choices:
                var = Vartype(**L._asdict(), in_channels=I, out_channels=O)
                layertypes[var] += 1
    return layertypes


def sample_file_iterator(samples_file):
    with open(samples_file, 'r') as f:
        for line in f:
            yield tuplify(json.loads(line))


def generate(device, dtype, batch_size, iterations, warmup, reduction, biased,
             count, repetitions, samples_file=os.devnull, save_every=10):
    os.makedirs(osp.dirname(samples_file), exist_ok=True)

    net = get_model()
    combinations = collect_repetitions(net)
    logger.info(f"{len(net.configurations)} modulers")
    logger.debug(f"search space: {net.configurations}")
    logger.debug(f"components: {net.components}")
    logger.info(f"Latency model has {len(combinations)} parameters")

    repeats = Counter()
    for c in combinations:
        repeats[c] = 0

    samples = []
    if osp.isfile(samples_file):
        for sample in sample_file_iterator(samples_file):
            samples.append(sample)
            repeats.update(collect_repetitions(net, sample[0]))
        logger.info(f"Loaded {samples_file}, "
                    f"min_repetition={min(repeats.values())} "
                    f"count={len(samples)} ")
    logger.info(f"Writing new samples to {samples_file}")
    new_samples = []
    while (len(samples) + len(new_samples) < count 
           or min(repeats.values()) < repetitions):
        configuration = gen_configuration(net, repeats, biased=biased)
        ms = benchmark(device, dtype, batch_size, iterations, warmup, reduction, configuration, silent=True)
        repeats.update(collect_repetitions(net, configuration))
        logger.info(f"{configuration}: {ms:.04f}ms, "
                    f"min_repetition={min(repeats.values())} "
                    f"count={len(samples) + len(new_samples)} ")
        new_samples.append([[int(d) for d in configuration], ms])
        if (len(new_samples) % save_every) == 0:
            with open(samples_file, 'a') as f:
                for sample in new_samples:
                    dump = json.dumps(sample) + '\n'
                    with DelayedKeyboardInterrupt():
                        f.write(dump)
            samples.extend(new_samples)
            new_samples = []

    samples.extend(new_samples)
    return samples


def build_equation(samples):
    """
    Samples can be iterator
    """
    net = get_model()
    variables = {}
    ivariables = {}
    Mcoord = []
    y = []
    for (i, sample) in enumerate(samples):
        y.append(sample[1])
        local_repeats = collect_repetitions(net, sample[0])
        for (L, r) in local_repeats.items():
            if L not in variables:
                j = len(variables)
                variables[L] = j
                ivariables[j] = L
            Mcoord.append((i, variables[L], r))
    y = np.array(y)
    M = lil_matrix((len(y), len(variables)))
    for (i, j, r) in Mcoord:
        M[i, j] = r
    return M, y, variables, ivariables


def solve_lsq(M, y, regularize=0.0, K=None):
    n = M.shape[1]
    x = cp.Variable(n)
    t = cp.Variable(K.shape[0])
    M_cp = cp.Constant(M)
    obj = cp.sum_squares(M_cp @ x - y)
    constraints = [x >= 0]
    if regularize:
        K_cp = cp.Constant(K)
        obj += regularize * cp.sum_squares(t)
        constraints += [t >= 0, K_cp @ x <= t]
    objective = cp.Minimize(obj)
    prob = cp.Problem(objective, constraints)
    prob.solve(cp.SCS, verbose=True)
    return x.value


def get_inequalities(variables):
    def other(L, *args):
        props = L._asdict()
        for k in args:
            del props[k]
        return tuple(props.values())
    buckets = defaultdict(list)
    for order in ['in_channels', 'out_channels', 'in_size']:
        for V in variables:
            buckets[other(V, order)].append(V)
    inequalities = []
    for bucket in buckets.values():
        bucket = sorted(bucket)
        for i in range(len(bucket) - 1):
            inequalities.append((bucket[i], bucket[i + 1]))
    K = lil_matrix((len(inequalities), len(variables)))
    for i, (C1, C2) in enumerate(inequalities):
        K[i, variables[C1]] = 1
        K[i, variables[C2]] = -1
    return K


def fit_model(samples, regularize=0.0):
    M, y, variables, ivariables = build_equation(samples)
    K = get_inequalities(variables)
    x = solve_lsq(M, y, regularize, K)
    model = []
    for i, ms in enumerate(x):
        model.append((ivariables[i], ms))
    return model


def dump_model(model, model_file):
    with open(model_file, 'w') as f:
        for m in model:
            var, ms = m
            dump = json.dumps([var._asdict(), ms]) + '\n'
            f.write(dump)


def load_model(model_file):
    with open(model_file, 'r') as f:
        for line in f:
            var, ms = tuplify(json.loads(line))
            var = Vartype(**var)
            yield (var, ms)


def fit(samples_file, model_file, regularize=0.0):
    os.makedirs(osp.dirname(model_file), exist_ok=True)
    samples = sample_file_iterator(samples_file)
    model = fit_model(samples, regularize)
    dump_model(model, model_file)
    return model


def validate(samples_file, model_file, plot_file):
    os.makedirs(osp.dirname(plot_file), exist_ok=True)
    model = load_model(model_file)
    model_dict = dict(model)
    samples = sample_file_iterator(samples_file)
    M, y, variables, ivariables = build_equation(samples)
    x = [model_dict[ivariables[i]] for i in range(len(variables))]
    yhat = M @ x
    rmse = np.sqrt(((y - yhat) ** 2).mean())
    title = f"RMSE {rmse:.04f}, NRMSE {100 * rmse / y.mean():.02f}%"
    print(title)
    plt.plot(y, yhat, 'o')
    plt.xlabel("ground truth (ms)")
    plt.ylabel("predicted (ms)")
    plt.title(title)
    plt.savefig(plot_file)


if __name__ == "__main__":
    logger = logging.getLogger(__file__)
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                        format='%(name)s: %(message)s')
    args = parse_args().__dict__

    globals()[args.pop('mode')](**args)
