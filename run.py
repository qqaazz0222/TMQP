import os
import json
import torch
import pickle
import multiprocessing
from argparse import ArgumentParser
from rich.console import Console
from model.model import init_model, predict
from utils.logger import log, log_execution_time, log_execution_time_with_dist, log_progress, console_banner, console_args

console = Console()

def init_args():
    """
    인자 초기화 함수

    Returns:
        args (ArgumentParser): 인자
    """
    parser = ArgumentParser()
    parser.add_argument('-P', '--patient_id', type=str, default="10003382", help='환자 아이디(단일 환자만 분석)')
    parser.add_argument('-O', '--overlap', action='store_true', help='근육 오버랩 여부')
    parser.add_argument('-C', '--calculate', action='store_true', help='IoU 계산 여부')
    parser.add_argument('-D', '--data_dir', type=str, default='data', help='데이터 디렉토리')
    parser.add_argument('-V', '--visualize', type=bool, default=False, help='시각화 여부')
    parser.add_argument('--batch_size', type=int, default=1, help='한번에 분석할 슬라이드 개수')
    parser.add_argument('--input_dir', type=str, default='data/input', help='입력(원본데이터) 디렉토리')
    parser.add_argument('--output_dir', type=str, default='data/output', help='출력 디렉토리')
    parser.add_argument('--working_dir', type=str, default='data/working', help='작업 디렉토리')
    parser.add_argument('--label_dir', type=str, default='data/label', help='라벨(정답데이터) 디렉토리')
    parser.add_argument('--weight', type=str, default='model/checkpoint.pth', help='모델 가중치 파일 경로')
    parser.add_argument('--threshold', type=float, default=0.9, help='CNN 모델의 확률 임계값')
    args = parser.parse_args()
    dict_args = vars(args)
    console_args(console, dict_args)
    return parser.parse_args()


def main():
    args = init_args()
    if args.data_dir != 'data':
        args.input_dir = os.path.join(args.data_dir, 'input')
        args.output_dir = os.path.join(args.data_dir, 'output')
        args.working_dir = os.path.join(args.data_dir, 'working')
        args.label_dir = os.path.join(args.data_dir, 'label')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = init_model(args.weight, device)
    
    
    
if __name__ == '__main__':
    main()