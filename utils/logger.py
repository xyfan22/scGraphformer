import torch
import numpy as np
from collections import defaultdict
from numpy import *

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin

            print_str=f'Run {run + 1:02d}:'+\
                f'Highest Train: {result[:, 0].max():.2f} '+\
                f'Highest Valid: {result[:, 1].max():.2f} '+\
                f'Highest Test: {result[:, 2].max():.2f} '+\
                f'Chosen epoch: {ind+1}\n'+\
                f'Final Train: {result[ind, 0]:.2f} '+\
                f'Final Test: {result[ind, 2]:.2f}'
            print(print_str)

        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test = r.mean()
            # Stoing best in Cross_platforms
            # return best_result[:, 4]
            if best_result[:, 1].mean() >= best_result[:, 4].mean():
                return best_result[:, 1]
            if best_result[:, 4].mean() > best_result[:, 1].mean():
                return best_result[:, 4]

    def output(self, out_path, info):
        with open(out_path, 'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')

import os
def save_result(args, results, TIME, cross_platforms = False, CP_acc = False):
    if cross_platforms:
        path = 'cache/results/CP_merge_knn'
        if CP_acc:
            CP_result = 100 * np.array(CP_acc)
        filename = f'{path}/CP_all_merge_knn.csv'
    else:
        path = 'cache/results'
        if args.use_knn:
            filename = f'{path}/base_results.csv'
        else:
            filename = f'{path}/base_results_withoutKNN.csv'
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    print(f"Saving results to {filename}")

    if cross_platforms:
        print(f"Ref->Query: {args.dataset}->{args.query_dataset}: {results.mean():.2f} ± {results.std():.2f}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(
                "scGraphformer,"
                +
                f"{args.dataset},"
                +
                f"{args.query_dataset},"
                +
                f"{results.mean():.2f},"
                +
                f"{results.std():.2f},"
                +
                f"{mean(TIME):.2f},"
                +
                f"{std(TIME):.2f},")
        for i in range(args.runs):
            if i != args.runs:
                with open(f"{filename}", 'a+') as write_obj:
                    write_obj.write(
                        f"{results.numpy()[i]:.2f},")
            if i == args.runs:
                with open(f"{filename}", 'a+') as write_obj:
                    write_obj.write(
                        f"{results.numpy()[i]:.2f}\n")
    else:
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(
                "scGraphformer,"
                +
                f"{args.dataset},"
                +
                f"{results.mean():.2f},"
                +
                f"{results.std():.2f},"
                +
                f"{mean(TIME):.2f},"
                +
                f"{std(TIME):.2f},")
        for i in range(args.runs):
            if i != args.runs:
                with open(f"{filename}", 'a+') as write_obj:
                    write_obj.write(
                        f"{results.numpy()[i]:.2f},")
            if i == args.runs:
                with open(f"{filename}", 'a+') as write_obj:
                    write_obj.write(
                        f"{results.numpy()[i]:.2f}\n")
