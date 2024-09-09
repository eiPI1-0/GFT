import os
import csv
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn as nn


def draw_lpd(
    model1, model2,
    save_dir='./compare_result',
    eps=1e-6,
):
    named_parameters1 = list(model1.named_parameters())
    named_parameters2 = list(model2.named_parameters())
    groups = [{}, {}, {}, {}]

    def compute(t):
        count = (t > eps).sum()
        if count > 0:
            avg = ((t[t > eps]).mean()).item()
            m = t.max().item()
        else:
            avg = 0
            m = 0
        return count, avg, m, t[t > eps].sum().item()

    for (name, params1), (_, params2) in tqdm(zip(named_parameters1, named_parameters2), total=len(named_parameters1)):
        group, type = name.split('.')[-2:]
        if 'bias' in type or 'norm' in group:
            continue
        t = torch.abs(params1 - params2)
        t = t*200 / torch.clamp(torch.abs(params1) + torch.abs(params2), min=eps)
        compare = compute(t)
        for v, g in zip(compare, groups):
            if group not in g:
                g[group] = []
            g[group].append(v)

    for g in groups:
        for k, v in g.items():
            if len(v) > 1:
                g[k] = np.array(v)

    sum_avg_array = None
    for k, v in groups[1].items():
        if len(v) > 1:
            if sum_avg_array is None:
                sum_avg_array = v
            else:
                sum_avg_array += v

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'result.csv'), 'w', newline='') as outfile:
            spamwriter = csv.writer(outfile)
            for i, g in zip(('Count', 'Avg', 'Max', 'Sum'), groups):
                for name, v in g.items():
                    if len(v) > 1:
                        spamwriter.writerow([f'{name}_{i}'] + v.tolist())
                        plt.plot(v, label=name)
                    else:
                        spamwriter.writerow([f'{name}_{i}'] + v)
                plt.xlabel('Layer ID')
                plt.ylabel(i)
                plt.legend()
                plt.savefig(os.path.join(save_dir, f'{i}.pdf'))
                plt.clf()
            spamwriter.writerow(['SUMAVG'] + sum_avg_array.tolist())
        plt.plot(sum_avg_array)
        plt.xlabel('Layer ID')
        plt.ylabel('LPD')
        plt.savefig(os.path.join(save_dir, 'LPD.pdf'))
        plt.clf()
    return groups


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    model_a_path = input('Model A path: ')
    model_b_path = input('Model B path: ')
    eps = input('eps(1e-6): ')
    eps = eps or 1e-6
    eps = float(eps)

    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_path,
        device_map="cpu",
        trust_remote_code=True,
    )
    model_b = AutoModelForCausalLM.from_pretrained(
        model_b_path,
        device_map="cpu",
        trust_remote_code=True
    )
    print('Comparing.......')
    draw_lpd(model_a, model_b, eps=eps)
    print('Done!')
