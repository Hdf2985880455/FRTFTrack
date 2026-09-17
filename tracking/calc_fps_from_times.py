import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    all_times = []
    for fn in os.listdir(args.result_dir):
        if fn.endswith('_time.txt'):
            path = os.path.join(args.result_dir, fn)
            arr = np.loadtxt(path, delimiter='\t')
            arr = np.atleast_1d(arr).astype(float)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size > 0:
                all_times.append(arr)

    if not all_times:
        print('No valid *_time.txt found.')
        return

    all_times = np.concatenate(all_times)
    mean_time = all_times.mean()
    fps = 1.0 / mean_time

    print(f'Result dir: {args.result_dir}')
    print(f'Frames: {len(all_times)}')
    print(f'Mean time per frame (s): {mean_time:.6f}')
    print(f'FPS: {fps:.3f}')

if __name__ == '__main__':
    main()