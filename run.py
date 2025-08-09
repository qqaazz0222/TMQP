import os
import sys
import json
import torch
import pickle
import numpy as np
import multiprocessing
from argparse import ArgumentParser
from rich.console import Console
from pathlib import Path
from typing import List, Tuple
try:
    from model.model import init_model, predict
    from modules.classifier import classify
    from modules.postprocess import post_processing
    from modules.extractor import extract, calc_boundary
    from modules.analyzer import overlap, counting
    from utils.directoryHandler import check_checkpoint
    from utils.dicomHandler import *
    from utils.imageHandler import *
    from utils.logger import log, log_execution_time, log_execution_time_with_dist, log_progress, console_banner, console_args
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

@log_execution_time
def init_args(console: Console) -> ArgumentParser:
    """
    인자 초기화 함수

    Args:
        console (Console): 콘솔 객체

    Returns:
        ArgumentParser: 파싱된 인자
    """
    try:
        parser = ArgumentParser(description='Rib Muscle Segmentation Tool')
        
        # 기본 인자들
        parser.add_argument('-P', '--patient_id', type=str, default="10003382", 
                          help='환자 아이디(단일 환자만 분석)')
        parser.add_argument('-O', '--overlap', action='store_true', 
                          help='근육 오버랩 시각화 여부')
        parser.add_argument('-C', '--calculate', action='store_true', 
                          help='IoU 계산 여부')
        parser.add_argument('-D', '--data_dir', type=str, default='data', 
                          help='데이터 디렉토리')
        parser.add_argument('-V', '--visualize', type=bool, default=False, 
                          help='3D 시각화 여부')
        
        # 성능 관련 인자
        parser.add_argument('--batch_size', type=int, default=1, 
                          help='한번에 분석할 슬라이드 개수')
        
        # 디렉토리 경로 인자들
        parser.add_argument('--input_dir', type=str, default='data/input', 
                          help='입력(원본데이터) 디렉토리')
        parser.add_argument('--output_dir', type=str, default='data/output', 
                          help='출력 디렉토리')
        parser.add_argument('--working_dir', type=str, default='data/working', 
                          help='작업 디렉토리')
        parser.add_argument('--label_dir', type=str, default='data/label', 
                          help='라벨(정답데이터) 디렉토리')
        
        # 모델 관련 인자
        parser.add_argument('--weight', type=str, default='model/checkpoint.pth', 
                          help='모델 가중치 파일 경로')
        parser.add_argument('--threshold', type=float, default=0.9, 
                          help='CNN 모델의 확률 임계값')
        parser.add_argument('--device', type=int, default=0, 
                          help='CNN 모델의 디바이스 ID')

        args = parser.parse_args()
        
        # 경로 유효성 검사
        if not Path(args.weight).exists():
            log("warning", f"Model weight file not found: {args.weight}")
        
        # 배치 크기 유효성 검사
        if args.batch_size < 1:
            args.batch_size = 1
            log("warning", "Batch size set to minimum value: 1")
        
        # 임계값 유효성 검사
        if not (0.0 <= args.threshold <= 1.0):
            args.threshold = 0.9
            log("warning", "Threshold set to default value: 0.9")
        
        dict_args = vars(args)
        console_args(console, dict_args)
        return args
        
    except Exception as e:
        log("error", f"Failed to initialize arguments: {e}")
        sys.exit(1)

def extract_helper(params: Tuple) -> Tuple:
    """
    근육 추출 헬퍼 함수 (멀티프로세싱용)
    
    Args:
        params (Tuple): (idx, working_file, method, working_file_num, category, sn, boundary, visualize_dir)
    
    Returns:
        Tuple: (working_file, original_image, extracted_muscle_image, metadata)
    """
    try:
        idx, working_file, method, working_file_num, category, sn, boundary, visualize_dir = params
        
        # 인덱스 유효성 확인
        if working_file_num <= 0:
            norm_idx = 0.0
        else:
            norm_idx = max(0.0, min(1.0, idx / working_file_num))  # 0~1 범위로 제한
            
        # 파일 존재 여부 확인
        if not os.path.exists(working_file):
            raise FileNotFoundError(f"Working file not found: {working_file}")
            
        original_image, extracted_muscle_image, metadata = extract(
            idx, norm_idx, category, sn, method, boundary, working_file, visualize_dir)
        
        return working_file, original_image, extracted_muscle_image, metadata
        
    except Exception as e:
        log("error", f"Error in extract_helper for {working_file}: {e}")
        empty_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 안전한 메타데이터 생성
        safe_metadata = {
            "idx": idx,
            "error": str(e),
            "abs": {
                "base_point": (256, 256),  # 중앙점으로 기본값 설정
                "left_point": (-1, -1),
                "right_point": (-1, -1),
                "left_y_intercept": 0,
                "right_y_intercept": 0
            }
        }
        return working_file, empty_image, empty_image, safe_metadata

def extract_muscle(working_list: List[str], pred_list: List[str], category: str, sn: str, 
                  checkpoint_dir: str, visualize_dir: str, batch_size: int = 1) -> Tuple[List, List, List]:
    """
    근육 추출 함수

    Args:
        working_list (List[str]): 작업 파일 리스트
        pred_list (List[str]): 예측된 클래스 리스트
        category (str): 분류 카테고리 ('in' 또는 'ex')
        sn (str): 시리즈 번호
        checkpoint_dir (str): 체크포인트 디렉토리
        visualize_dir (str): 시각화 디렉토리
        batch_size (int): 배치 크기

    Returns:
        Tuple[List, List, List]: (원본 이미지 리스트, 추출된 근육 이미지 리스트, 메타데이터 리스트)
    """
    try:
        st = log_execution_time_with_dist("start", "extract_muscle")
        
        # 입력 유효성 검사
        if not working_list:
            log("warning", f"Empty working list for category: {category}, series: {sn}")
            return [], [], []
            
        if len(working_list) != len(pred_list):
            log("warning", f"Mismatch between working_list and pred_list lengths")
            min_len = min(len(working_list), len(pred_list))
            working_list = working_list[:min_len]
            pred_list = pred_list[:min_len]
        
        checkpoint = f"checkpoint_extract_{category}_{sn}.pkl"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        
        # 체크포인트 확인
        # TODO: 임시로 비활성화
        if check_checkpoint(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                return pickle.load(f)

        original_image_list = []
        extracted_muscle_image_list = []
        metadata_list = []
        working_file_num = max(1, len(working_list) - 1)  # 0으로 나누기 방지
            
        # 경계 계산
        try:
            boundary = calc_boundary(working_list)
        except Exception as e:
            log("warning", f"Failed to calculate boundary: {e}. Using empty boundary.")
            boundary = []
        
        # 멀티프로세싱 매개변수 준비
        param_list = []
        for idx, working_file in enumerate(working_list):
            if idx < len(pred_list):
                method = pred_list[idx]
            else:
                method = 'lower'  # 기본값
            param_list.append((idx, working_file, method, working_file_num, category, sn, boundary, visualize_dir))

        # 안전한 배치 크기 설정
        safe_batch_size = min(batch_size, multiprocessing.cpu_count(), len(param_list))
        
        # 멀티프로세싱 실행
        try:
            with multiprocessing.Pool(processes=safe_batch_size) as pool:
                results = []
                for i, result in enumerate(pool.imap(extract_helper, param_list)):
                    working_file, original_image, extracted_muscle_image, metadata = result
                    log_progress(i + 1, len(working_list), f"Extracting Muscle ({os.path.basename(working_file)})")
                    
                    # 에러 체크
                    if "error" in metadata:
                        log("warning", f"Error in processing {working_file}: {metadata['error']}")
                    
                    original_image_list.append(original_image)
                    extracted_muscle_image_list.append(extracted_muscle_image)
                    metadata_list.append(metadata)
                    
        except Exception as e:
            log("error", f"Multiprocessing failed: {e}. Falling back to sequential processing.")
            # 순차 처리로 폴백
            for i, params in enumerate(param_list):
                result = extract_helper(params)
                working_file, original_image, extracted_muscle_image, metadata = result
                log_progress(i + 1, len(working_list), f"Extracting Muscle ({os.path.basename(working_file)})")
                
                original_image_list.append(original_image)
                extracted_muscle_image_list.append(extracted_muscle_image)
                metadata_list.append(metadata)

        # 체크포인트 저장
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump([original_image_list, extracted_muscle_image_list, metadata_list], f)
        except Exception as e:
            log("warning", f"Failed to save checkpoint: {e}")
            
        log_execution_time_with_dist("end", "extract_muscle", st)
        return original_image_list, extracted_muscle_image_list, metadata_list
        
    except Exception as e:
        log("error", f"Critical error in extract_muscle: {e}")
        return [], [], []

def main():
    """
    메인 함수
    """
    try:
        # 콘솔 설정
        console = Console()
        console = console.__class__(log_time=False)
        console_banner(console)
        
        # 인자 초기화
        args = init_args(console)
        
        # 데이터 디렉토리 설정
        if args.data_dir != 'data':
            args.input_dir = os.path.join(args.data_dir, 'input')
            args.output_dir = os.path.join(args.data_dir, 'output')
            args.working_dir = os.path.join(args.data_dir, 'working')
            args.label_dir = os.path.join(args.data_dir, 'label')
        
        # 필수 디렉토리 생성
        for directory in [args.output_dir, args.working_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # GPU/CPU 설정
        # GPU/CPU 설정
        if torch.cuda.is_available():
            if args.device >= 0 and args.device < torch.cuda.device_count():
                device = torch.device(f"cuda:{args.device}")
                log("info", f"Using GPU device: {device}")
            else:
                device = torch.device("cuda:0")
                log("warning", f"Invalid device ID {args.device}, using default GPU: {device}")
        else:
            device = torch.device("cpu")
            log("info", f"CUDA not available, using CPU: {device}")
        
        # 모델 초기화
        try:
            model, transform = init_model(args.weight, device)
            log("success", "Model initialized successfully")
        except Exception as e:
            log("error", f"Failed to initialize model: {e}")
            return

        # DICOM 파일 분류
        try:
            classified_list = classify(args.patient_id, args.input_dir, args.working_dir, args.output_dir, console)
            if not classified_list:
                log("warning", f"No data found for patient {args.patient_id}")
                return
            log("success", f"Found {len(classified_list)} datasets to process")
        except Exception as e:
            log("error", f"Failed to classify DICOM files: {e}")
            return
        
        # 각 데이터셋 처리
        for dataset_idx, (cur_sub_input_dir, working_dir, output_dir, classify_dict, slide_location_dict, size) in enumerate(classified_list):
            try:
                date = working_dir.split('/')[-2] if '/' in working_dir else 'unknown'
                log("info", f"Processing Patient ID: {args.patient_id}, Date: {date} ({dataset_idx+1}/{len(classified_list)})", upper_div=True)
                
                # classify_dict 구조 검증
                if not isinstance(classify_dict, dict):
                    log("error", f"classify_dict is not a dictionary: {type(classify_dict)}")
                    continue
                
                # 필수 키 확인
                required_keys = ["in", "ex"]
                for key in required_keys:
                    if key not in classify_dict:
                        log("warning", f"Missing key '{key}' in classify_dict")
                        classify_dict[key] = {}
                    elif not isinstance(classify_dict[key], dict):
                        log("warning", f"classify_dict['{key}'] is not a dictionary, converting to empty dict")
                        classify_dict[key] = {}
                
                log("info", f"classify_dict structure: IN={len(classify_dict['in'])} series, EX={len(classify_dict['ex'])} series")
                
                # 디렉토리 설정
                label_dir = os.path.join(args.label_dir, args.patient_id)
                checkpoint_dir = os.path.join(working_dir, 'checkpoint')
                visualize_dir = os.path.join(working_dir, 'visualize')
                muscle_dir = os.path.join(output_dir, 'muscle')
                abs_dir = os.path.join(output_dir, 'abs')
                overlap_dir = os.path.join(output_dir, 'overlap')
                
                # 결과 저장용 딕셔너리 초기화
                original_image_dict = {"in": {}, "ex": {}}
                extracted_muscle_image_dict = {"in": {}, "ex": {}}
                metadata_dict = {"in": {}, "ex": {}}
                
                # 각 카테고리별 처리
                for category in ["in", "ex"]:
                    try:
                        # classify_dict[category]가 dict인지 확인 후 keys() 호출
                        if not isinstance(classify_dict[category], dict):
                            log("warning", f"classify_dict[{category}] is not a dictionary: {type(classify_dict[category])}")
                            continue
                            
                        sn_list = list(classify_dict[category].keys())
                        if not sn_list:
                            log("warning", f"No series found for category {category}")
                            continue
                            
                        log("info", f"Processing {category.upper()} category with {len(sn_list)} series")
                        
                        for sn in sn_list:
                            try:
                                working_list = classify_dict[category][sn]
                                if not working_list:
                                    log("warning", f"Empty working list for series {sn}")
                                    continue
                                
                                # 예측 수행
                                pred_list = predict(model, transform, device, checkpoint_dir, working_list, category, sn)
                                
                                if pred_list is None:
                                    log("warning", f"Prediction failed for series {sn}")
                                    continue
                                
                                # 정렬 처리
                                if len(pred_list) >= 2 and pred_list[0] == 'lower' and pred_list[-1] == 'upper':
                                    working_list.sort(reverse=True)
                                    pred_list.sort(reverse=True)
                                
                                # 근육 추출
                                original_image_list, extracted_muscle_image_list, metadata_list = extract_muscle(
                                    working_list, pred_list, category, sn, checkpoint_dir, visualize_dir, 
                                    batch_size=args.batch_size)
                                
                                if not original_image_list:
                                    log("warning", f"No images extracted for series {sn}")
                                    continue
                                
                                original_image_dict[category][sn] = original_image_list
                                extracted_muscle_image_dict[category][sn] = extracted_muscle_image_list
                                metadata_dict[category][sn] = metadata_list
                                
                                log("success", f"Processed series {sn}: {len(original_image_list)} images")
                                
                            except Exception as e:
                                log("error", f"Error processing series {sn} in category {category}: {e}")
                                continue
                                
                    except Exception as e:
                        log("error", f"Error processing category {category}: {e}")
                        continue
                
                # 메타데이터 검증 및 보완
                def validate_and_fix_metadata(metadata_dict):
                    """메타데이터를 검증하고 누락된 'abs' 정보를 추가"""
                    for category in ["in", "ex"]:
                        if category in metadata_dict:
                            for sn in metadata_dict[category]:
                                for i, metadata in enumerate(metadata_dict[category][sn]):
                                    if "abs" not in metadata:
                                        # 기본 abs 정보 추가
                                        metadata["abs"] = {
                                            "base_point": (256, 256),
                                            "left_point": (-1, -1),
                                            "right_point": (-1, -1),
                                            "left_y_intercept": 0,
                                            "right_y_intercept": 0
                                        }
                                        log("warning", f"Added missing 'abs' metadata for {category} series {sn} index {i}")
                    return metadata_dict
                
                # 메타데이터 검증
                metadata_dict = validate_and_fix_metadata(metadata_dict)
                
                # 메타데이터 저장
                try:
                    metadata_path = os.path.join(working_dir, 'metadata.json')
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata_dict, f, indent=4, ensure_ascii=False)
                    log("success", f"Metadata saved to {metadata_path}")
                except Exception as e:
                    log("warning", f"Failed to save metadata: {e}")
                
                # 후처리
                try:
                    post_processed_muscle_image_dict = post_processing(extracted_muscle_image_dict, metadata_dict)
                    log("success", "Post-processing completed")
                except Exception as e:
                    log("error", f"Post-processing failed: {e}")
                    post_processed_muscle_image_dict = extracted_muscle_image_dict  # 폴백
                
                # 픽셀 카운팅
                try:
                    counting(post_processed_muscle_image_dict, slide_location_dict, cur_sub_input_dir, 
                            output_dir, args.output_dir, size, args.visualize)
                    log("success", "Counting completed")
                except Exception as e:
                    log("error", f"Counting failed: {e}")
                
                # 오버랩 시각화
                if args.overlap:
                    try:
                        overlap(original_image_dict, post_processed_muscle_image_dict, muscle_dir, abs_dir, overlap_dir)
                        log("success", "Overlap visualization completed")
                    except Exception as e:
                        log("error", f"Overlap visualization failed: {e}")
                    
            except Exception as e:
                log("error", f"Error processing dataset {dataset_idx+1}: {e}")
                continue
        
        log("success", "Processing Finished")
        
    except KeyboardInterrupt:
        log("warning", "Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        log("error", f"Critical error in main function: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()