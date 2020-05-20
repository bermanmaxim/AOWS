import itertools
import numpy as np
from numba import jit
import sys, logging
logging.getLogger('numba').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def complete(unary, pairwise, fill=-np.inf):
    """
    Convert lists of unaries and pairwises into tensors by filling blanks
    """
    N = len(unary)
    states = np.array(list(map(len, unary)))
    max_S = max(states)
    un = np.full((N, max_S), fill)
    pair = np.full((N - 1, max_S, max_S), fill)
    for i in range(N):
        un[i, :len(unary[i])] = unary[i]
        if i < N - 1:
            pair[i, :len(unary[i]), :len(unary[i + 1])] = pairwise[i]

    return un, pair, states


@jit(nopython=True)
def maxsum(unary, pairwise, states):
    N, S = unary.shape
    partial = unary[0]
    selected = np.zeros((N - 1, S), np.int64)
    for s in range(N - 1):
        new_partial = np.full((S,), -np.inf)
        for j in range(states[s + 1]):
            best_ = -np.inf
            best_i = 0
            for i in range(states[s]):
                candidate = partial[i] + pairwise[s, i, j]
                if candidate > best_:
                    best_ = candidate
                    best_i = i
            selected[s, j] = best_i
            new_partial[j] = unary[s + 1, j] + best_
        partial = new_partial

    path = np.zeros((N,), np.int64)
    score = -np.inf
    best_j = 0
    for j in range(states[N - 1]):
        candidate = partial[j]
        if candidate > score:
            score = candidate
            best_j = j
    path[N - 1] = best_j
    for i in range(N - 2, -1, -1):
        best_j = selected[i, best_j]
        path[i] = best_j
    return score, path


def score(path, unary, pairwise, detail=False):
    if not len(path): return 0.0
    S = unary[0][path[0]]
    Sp = 0.0
    prev = path[0]
    for i, p in enumerate(path[1:]):
        Sp += pairwise[i][prev, p]
        S += unary[i + 1][p]
        prev = p
    if detail:
        return S, Sp
    return S + Sp


def maxsum_brute(unary, pairwise, states, K=3):
    """
    Brute-force max-sum (for debugging)
    """
    best_path = None
    best_score = -float('inf')
    for path in itertools.product(*[range(s) for s in states]):
        sc = score(path, unary, pairwise)
        if sc > best_score:
            best_path = path
            best_score = sc
    return best_score, best_path


@jit(nopython=True)
def sumprod_log(unary, pairwise, states, logspace=False):
    N, max_s = unary.shape
    alpha = np.zeros((N, max_s), dtype=unary.dtype)
    beta = np.zeros((N, max_s), dtype=unary.dtype)
    alpha[0, :states[0]] = unary[0][:states[0]]
    alpha[0, states[0]:] = -np.inf
    for s in range(N - 1):
        for k2 in range(states[s + 1]):
            M = -np.inf
            for k1 in range(states[s]):
                C = alpha[s, k1] + pairwise[s, k1, k2]
                if C > M:
                    M = C
            for k1 in range(states[s]):
                alpha[s + 1, k2] += np.exp(alpha[s, k1] + pairwise[s, k1, k2] - M)
            alpha[s + 1, k2] = unary[s + 1, k2] + np.log(alpha[s + 1, k2]) + M
        for k2 in range(states[s + 1], max_s):
            alpha[s + 1, k2] = -np.inf
        M = alpha[s + 1, :states[s + 1]].max()
        Z = np.log(np.exp(alpha[s + 1, :states[s + 1]] - M).sum()) + M
        alpha[s + 1, :states[s + 1]] = alpha[s + 1, :states[s + 1]] - Z
    for s in range(N - 2, -1, -1):
        for k1 in range(states[s]):
            M = -np.inf
            for k2 in range(states[s + 1]):
                C = beta[s + 1, k2] + pairwise[s, k1, k2] + unary[s + 1, k2]
                if C > M:
                    M = C
            for k2 in range(states[s + 1]):
                beta[s, k1] += np.exp(beta[s + 1, k2] + pairwise[s, k1, k2] + unary[s + 1, k2] - M)
            beta[s, k1] = np.log(beta[s, k1]) + M
        for k1 in range(states[s], max_s):
            beta[s, k1] = -np.inf
        M = beta[s, :states[s]].max()
        Z = np.log(np.exp(beta[s, :states[s]] - M).sum()) + M
        beta[s, :states[s]] = beta[s, :states[s]] - Z
    marg = alpha + beta
    for s in range(N):
        marg[s] = marg[s] - marg[s].max()
    if not logspace:
        marg = np.exp(marg)
        marg = marg / marg.sum(1).reshape(-1, 1)
    return marg


