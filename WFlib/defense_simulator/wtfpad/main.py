import sys
import argparse
import configparser
import numpy as np

from os import mkdir, listdir
from os.path import join, isdir
import os

from time import strftime

import adaptive as ap
import constants as ct
import overheads as oh
from pparser import Trace, parse, dump

import logging
import glob
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


logger = logging.getLogger('wtfpad')
#parameter

MON_SITE_NUM = 10
MON_INST_NUM = 1
UNMON_SITE_NUM = 10
OPEN_WORLD = 1

def process_trace_wrapper(trace_path, config, output_dir):
    """
    单个 Trace 处理函数，用于多进程调用
    """
    if not os.path.isfile(trace_path):
        return None
    
    try:
        fname = os.path.basename(trace_path)
        trace = parse(trace_path)
        
        # 在每个进程中独立实例化模拟器，避免状态冲突
        wtfpad = ap.AdaptiveSimulator(config)
        
        simulated = wtfpad.simulate(Trace(trace))
        dump(simulated, join(output_dir, fname))

        # calculate overheads
        total_len = sum(getattr(pkt, 'length', 0) for pkt in trace)
        bw_ovhd = 0
        if total_len > 0:
            bw_ovhd = oh.bandwidth_ovhd(simulated, trace)
        
        lat_ovhd = oh.latency_ovhd(simulated, trace)
        
        return {
            'success': True,
            'fname': fname,
            'bw_ovhd': bw_ovhd,
            'lat_ovhd': lat_ovhd,
            'total_len': total_len
        }
    except Exception as e:
        return {
            'success': False,
            'fname': os.path.basename(trace_path),
            'error': str(e)
        }

def init_directories(name):
    # Create a results dir if it doesn't exist yet
    if not isdir(ct.RESULTS_DIR):
        mkdir(ct.RESULTS_DIR)

    # Define output directory
    timestamp = strftime('%m%d_%H%M')
    output_dir = join(ct.RESULTS_DIR, 'wtfpad' + '_' + timestamp)
    logger.info("Creating output directory: %s" % output_dir)

    # make the output directory
    mkdir(name)

    return name


def main():
    # parser config and arguments
    args, config = parse_arguments()
    logger.info("Arguments: %s, Config: %s" % (args, config))

    # Init run directories
    output_dir = init_directories(args.name)

    # 遍历 traces_path 下所有文件
    trace_files = glob.glob(os.path.join(args.traces_path, '*'))
    latencies, bandwidths = [], []

    logger.info(f"Starting simulation with {args.workers} workers processing {len(trace_files)} files.")

    # 准备多进程参数
    worker_func = partial(process_trace_wrapper, config=config, output_dir=output_dir)

    # 使用进程池进行并行处理
    with Pool(processes=args.workers) as pool:
        # 使用 imap_unordered 可以让结果乱序返回，提高进度条响应速度
        results_iter = pool.imap_unordered(worker_func, trace_files)
        
        # 使用 tqdm 包装迭代器
        pbar = tqdm(results_iter, total=len(trace_files), desc="Simulating", unit="trace")
        
        for res in pbar:
            if res is None:
                continue
            
            if not res['success']:
                tqdm.write(f"Error processing {res['fname']}: {res['error']}")
                continue

            fname = res['fname']
            bw_ovhd = res['bw_ovhd']
            lat_ovhd = res['lat_ovhd']
            
            # 警告信息使用 tqdm.write 输出
            if res['total_len'] == 0:
                tqdm.write(f"Warning: Trace {fname} has total packet length 0, bandwidth overhead set to 0.")
            
            bandwidths.append(bw_ovhd)
            latencies.append(lat_ovhd)
            
            # 更新进度条描述（可选，如果刷新太快可以去掉）
            # pbar.set_description(f"Finished {fname}")

    logger.info("Latency overhead: %s" % np.median([l for l in latencies if l > 0.0]))
    logger.info("Bandwidth overhead: %s" % np.median([b for b in bandwidths if b > 0.0]))

    # if OPEN_WORLD:
    #     for i in range(UNMON_SITE_NUM):
    #         fname = str(i)
    #         if os.path.exists(join(args.traces_path,fname)):
    #             trace = parse(join(args.traces_path, fname))
    #             logger.info("Simulating trace: %s" % fname)
    #             simulated = wtfpad.simulate(Trace(trace))
    #             dump(simulated, join(output_dir, fname))
    
    #             total_len = sum(getattr(pkt, 'length', 0) for pkt in trace)
    #             if total_len > 0:
    #                 bw_ovhd = oh.bandwidth_ovhd(simulated, trace)
    #             else:
    #                 bw_ovhd = 0
    #                 logger.warning("Trace %s has total packet length 0, bandwidth overhead set to 0." % fname)
    #             bandwidths.append(bw_ovhd)
    #             logger.debug("Bandwidth overhead: %s" % bw_ovhd)
    
    #             lat_ovhd = oh.latency_ovhd(simulated, trace)
    #             latencies.append(lat_ovhd)
    #             logger.debug("Latency overhead: %s" % lat_ovhd)
    #         else:
    #             logger.warning('File %s does not exist!'%fname)

    logger.info("Latency overhead: %s" % np.median([l for l in latencies if l > 0.0]))
    logger.info("Bandwidth overhead: %s" % np.median([b for b in bandwidths if b > 0.0]))


def parse_arguments():
    # Read configuration file
    conf_parser = configparser.RawConfigParser()
    conf_parser.read(ct.CONFIG_FILE)

    parser = argparse.ArgumentParser(description='It simulates adaptive padding on a set of web traffic traces.')

    parser.add_argument('traces_path',
                        metavar='<traces path>',
                        help='Path to the directory with the traffic traces to be simulated.')

    parser.add_argument('-c', '--config',
                        dest="section",
                        metavar='<config name>',
                        help="Adaptive padding configuration.",
                        choices=conf_parser.sections(),
                        default="normal_rcv")

    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    parser.add_argument('--log-level',
                        type=str,
                        dest="loglevel",
                        metavar='<log level>',
                        help='logging verbosity level.')
    
    parser.add_argument('--name',
                        type=str,
                        dest="name",
                        metavar='<name>',
                        default='null',)

    parser.add_argument('--workers',
                        type=int,
                        dest="workers",
                        metavar='<num_workers>',
                        default=4,
                        help='Number of worker processes to use for simulation.')

    # Parse arguments
    args = parser.parse_args()

    # Get section in config file
    config = conf_parser._sections[args.section]

    # Use default values if not specified
    config = dict(config, **conf_parser._sections['normal_rcv'])

    # logging config
    config_logger(args)

    return args, config


def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    # Set logging format
    ch.setFormatter(logging.Formatter(ct.LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(-1)
