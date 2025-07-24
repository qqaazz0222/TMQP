import os
from utils.logger import log
from argparse import ArgumentParser

def init_dir(args: ArgumentParser):
    """
    디렉토리 구조를 초기화하는 함수

    Args:
        args (ArgumentParser): 인자 파서

    Returns:
        bool: 디렉토리가 이미 존재하는지 여부
    """
    flag = True
    if not os.path.exists(args.input_dir):
        os.makedirs(args.input_dir)
        flag = False
    if not os.path.exists(args.label_dir):
        os.makedirs(args.label_dir)
        flag = False
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        flag = False
    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)
        flag = False
    if not flag:
        raise Exception("Directory not found. Created new directories.")
    return flag

def create_working_dir_structure(patient_id: str, working_dir: str, date_list: list):
    """
    전처리 디렉터리 구조를 생성하는 함수

    Args:
        patient_id (str): 환자 ID
        input_dir (str): 입력 디렉터리 경로
    """
    working_dir_with_patient_id = os.path.join(working_dir, patient_id)
    working_dir_list = []
    os.makedirs(working_dir_with_patient_id, exist_ok=True)
    for date in date_list:
        date_dir = os.path.join(working_dir_with_patient_id, date)
        os.makedirs(date_dir, exist_ok=True)
        in_dir = os.path.join(date_dir, "in")
        ex_dir = os.path.join(date_dir, "ex")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(ex_dir, exist_ok=True)
        working_dir_list.append((date_dir, in_dir, ex_dir))
    return working_dir_with_patient_id, working_dir_list
def check_checkpoint(target_path: str):
    """
    이전에 처리한 데이터가 존재하는지 확인하는 함수

    Args:
        target_path (str): 대상 파일명

    Returns:
        bool: 파일이 존재하는지 여부
    """
    flag = os.path.exists(target_path)
    if flag:
        log("info", f"A checkpoint exists. Use the checkpoint. ({target_path})")
    return flag

def create_fig_dir(dir: str, target: str):
    """
    그래프 디렉터리를 생성하는 함수

    Args:
        dir (str): 디렉터리 경로
        target (str): convex, masking

    Returns:
        str: 그래프 디렉터리 경로
    """
    target_dir = os.path.join(dir, target)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir