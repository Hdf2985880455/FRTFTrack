import os
import sys
import importlib
import argparse
import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.models import build_mmtrack


def count_params_m(module):
    if module is None:
        return 0.0
    return sum(p.numel() for p in module.parameters()) / 1e6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='yaml name under experiments/mmtrack, without .yaml')
    return parser.parse_args()


def main():
    args = parse_args()

    yaml_path = f'experiments/mmtrack/{args.config}.yaml'
    config_module = importlib.import_module('lib.config.mmtrack.config')
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_path)

    model = build_mmtrack(cfg, is_training=False)
    model.eval()

    total = count_params_m(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    backbone = count_params_m(getattr(model, 'backbone', None))
    text_encoder = getattr(model, 'text_encoder', None)
    text_total = count_params_m(text_encoder)
    text_embeddings = count_params_m(getattr(text_encoder, 'embeddings', None)) if text_encoder is not None else 0.0
    text_encoder_layers = count_params_m(getattr(text_encoder, 'encoder', None)) if text_encoder is not None else 0.0
    text_pooler = count_params_m(getattr(text_encoder, 'pooler', None)) if text_encoder is not None else 0.0
    decoder = count_params_m(getattr(model, 'decoder', None))
    predictor = count_params_m(getattr(model, 'predictor', None))
    bottleneck = count_params_m(getattr(model, 'bottleneck', None))
    text_adj = count_params_m(getattr(model, 'text_adj', None))
    vl_fusion = count_params_m(getattr(model, 'vl_fusion', None))

    # Paper-style count that matches the published MMTrack table more closely.
    paper_style = backbone + text_encoder_layers + decoder + predictor

    print(f'Config: {args.config}')
    print(f'Total Params (M): {total:.3f}')
    print(f'Trainable Params (M): {trainable:.3f}')
    print(f'Paper-style Params (M): {paper_style:.3f}')
    print('')
    print('Breakdown (M):')
    print(f'  backbone: {backbone:.3f}')
    print(f'  text_encoder_total: {text_total:.3f}')
    print(f'  text_embeddings: {text_embeddings:.3f}')
    print(f'  text_encoder_layers: {text_encoder_layers:.3f}')
    print(f'  text_pooler: {text_pooler:.3f}')
    print(f'  decoder: {decoder:.3f}')
    print(f'  predictor: {predictor:.3f}')
    print(f'  bottleneck: {bottleneck:.3f}')
    print(f'  text_adj: {text_adj:.3f}')
    print(f'  vl_fusion: {vl_fusion:.3f}')


if __name__ == '__main__':
    main()
