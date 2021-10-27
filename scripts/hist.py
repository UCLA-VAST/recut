#!/usr/bin/env python
from __future__ import print_function
import argparse
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize histogram output')
    parser.add_argument(
        '-f', metavar='FILE', type=argparse.FileType('r'), default=sys.stdin,
        dest='file', help='path to file containing the csv benchmark data')
    parser.add_argument(
        '--xlabel', type=str, default='input size', help='label of the x-axis')
    parser.add_argument(
        '--ylabel', type=str, help='label of the y-axis')
    parser.add_argument(
        '--title', type=str, default='', help='title of the plot')
    parser.add_argument(
        '--logx', action='store_true', help='plot x-axis on a logarithmic scale')
    parser.add_argument(
        '--logy', action='store_true', help='plot y-axis on a logarithmic scale')

    args = parser.parse_args()
    if args.ylabel is None:
        args.ylabel = 'Cumulative %'
    return args


def read_data(args):
    """Read and process dataframe using commandline args"""
    try:
        data = pd.read_csv(args.file)
    except ValueError:
        msg = 'Could not parse the benchmark data. Did you forget "--benchmark_format=csv"?'
        logging.error(msg)
        exit(1)
    return data


def plot_hist(data, args):
    """Display the processed data"""
    plt.plot(data['range'], data['cumulative %'], label='% of total image', marker='.')
    if args.logx:
        plt.xscale('log')
    if args.logy:
        plt.yscale('log')
    plt.xlabel('Pixel value')
    c = 'k'
    plt.axvline(x=256, label='8-bit max', c='r')
    plt.axvline(x=996, label='99.9992 % background', c=c)
    plt.axvline(x=65535, label='16-bit max', c='g')
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.legend()
    plt.show()

def main():
    """Entry point of the program"""
    args = parse_args()
    data = read_data(args)
    plot_hist(data, args)


if __name__ == '__main__':
    main()
