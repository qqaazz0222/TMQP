import os
import pydicom
import shutil
from collections import defaultdict
from utils.orderChecker import check_order
from utils.logger import log, log_execution_time, console_classify

def copy_to_working_dir(target_dir: str, target_list: list, reversed: bool = False):
    """
    DICOM 파일 리스트를 작업 디렉토리로 복사하는 함수

    Args:
        target_dir (str): 작업 디렉토리 경로
        target_list (list): DICOM 파일 정보 리스트
        reversed (bool): 순서를 뒤집을지 여부
    
    Returns:
        tuple: (sorted_dict, sorted_sl_dict) - 시리즈별 파일 딕셔너리와 슬라이스 위치 딕셔너리
    """
    if reversed:
        target_list = target_list[::-1]
    
    sorted_dict = defaultdict(list)
    sorted_sl_dict = defaultdict(list)
    count = defaultdict(int)
    
    # 필요한 디렉토리들을 미리 생성
    series_dirs = set()
    for target in target_list:
        _, sn, _, _ = target
        series_dirs.add(str(sn))
    
    for series_dir in series_dirs:
        os.makedirs(os.path.join(target_dir, series_dir), exist_ok=True)
    
    for target in target_list:
        filepath, sn, sl, _ = target
        sn_str = str(sn)
        cur_dir = os.path.join(target_dir, sn_str)
        cur_file = os.path.join(cur_dir, f"{count[sn]:03d}.dcm")
        
        shutil.copy(filepath, cur_file)
        sorted_dict[sn].append(cur_file)
        sorted_sl_dict[sn].append(sl)
        count[sn] += 1
    
    return dict(sorted_dict), dict(sorted_sl_dict)

def matching_keyword(dicom_data: pydicom.dataset.FileDataset):
    """
    DICOM 데이터셋에서 특정 키워드를 찾는 함수

    Args:
        dicom_data (pydicom.dataset.FileDataset): DICOM 데이터셋

    Returns:
        str: 'in', 'ex', 또는 'unknown' 카테고리
    """
    series_description = getattr(dicom_data, 'SeriesDescription', None)
    if not series_description:
        return 'in'  # 기본값
    
    description = series_description.lower()
    
    # 더 구체적인 키워드를 먼저 검사
    ex_keywords = ['expiration', 'exhalation', 'exp', 'ex', 'exh', 
                   'post', 'out', 'exhale', 'expiratory',
                   'forced exp', 'end exp']
    
    for keyword in ex_keywords:
        if keyword in description:
            return 'ex'
    
    # 흡기 키워드 검사
    in_keywords = ['inspiration', 'inhalation', 'insp', 'in', 'ins', 'pre']
    for keyword in in_keywords:
        if keyword in description:
            return 'in'
    
    return 'in'  # 기본값
    
def check_axial(dicom_file_1: str, dicom_file_2: str):
    """
    두 DICOM 파일이 Axial 이미지인지 확인하는 함수

    Args:
        dicom_file_1 (str): 첫 번째 DICOM 파일 경로
        dicom_file_2 (str): 두 번째 DICOM 파일 경로

    Returns:
        bool: 두 파일이 Axial 이미지인 경우 True, 그렇지 않으면 False
    """
    try:
        ds1 = pydicom.dcmread(dicom_file_1)
        ds2 = pydicom.dcmread(dicom_file_2)
        
        position_1 = ds1.ImagePositionPatient
        position_2 = ds2.ImagePositionPatient
        
        # X, Y 좌표가 같은지만 확인 (Z축은 다를 수 있음)
        return (position_1[0] == position_2[0] and 
                position_1[1] == position_2[1])
    except (AttributeError, IndexError, Exception):
        return False

def _process_dicom_files(files, cur_working_sub_dir, cur_output_sub_dir):
    """
    DICOM 파일들을 처리하는 헬퍼 함수
    
    Args:
        files (list): DICOM 파일 경로 리스트
        cur_working_sub_dir (str): 작업 디렉토리 경로
        cur_output_sub_dir (str): 출력 디렉토리 경로
    
    Returns:
        tuple: (classify_dict, slide_location_dict, summary, size)
    """
    summary = {'in': 0, 'ex': 0, 'skipped': 0, 'error': 0}
    classify_dict = {"series_description": [], "in": [], "ex": []}
    slide_location_dict = {"in": [], "ex": []}
    size = None
    
    for file in files:
        if not file.endswith('.dcm'):
            continue
            
        try:
            dicom_data = pydicom.dcmread(file)
            
            # Modality 확인
            modality = getattr(dicom_data, 'Modality', '').lower()
            if modality != 'ct':
                summary['skipped'] += 1
                continue
            
            # 필요한 DICOM 정보 추출
            sn = str(dicom_data.SeriesNumber)
            
            # SliceLocation 또는 ImagePositionPatient[2] 사용
            try:
                sl = dicom_data.SliceLocation
            except AttributeError:
                try:
                    sl = float(dicom_data.ImagePositionPatient[2])
                except (AttributeError, IndexError, TypeError):
                    sl = 0.0  # 기본값
            
            # PixelSpacing과 SliceThickness 정보
            ps = getattr(dicom_data, 'PixelSpacing', [1.0, 1.0])
            st = getattr(dicom_data, 'SliceThickness', 1.0)
            
            if size is None:
                size = (float(ps[0]), float(ps[1]), float(st))
            
            # 카테고리 분류
            category = matching_keyword(dicom_data)
            if category in ['in', 'ex']:
                summary[category] += 1
                classify_dict[category].append((file, sn, sl, ps))
            else:
                summary['skipped'] += 1
                
        except Exception:
            summary['error'] += 1
            continue
    
    return classify_dict, slide_location_dict, summary, size

@log_execution_time
def classify(patient_id: str, input_dir: str, working_dir: str, output_dir: str, console):
    """
    입력 디렉토리의 DICOM 파일들을 분류하는 함수

    Args:
        patient_id (str): 환자 ID
        input_dir (str): 입력 디렉터리 경로
        working_dir (str): 작업 디렉터리 경로
        output_dir (str): 출력 디렉터리 경로
        console: 콘솔 객체

    Returns:
        list: 분류된 DICOM 파일들의 정보 리스트
    """
    classified_list = []
    _date_list = []
    _sub_list = []
    _summary_list = []

    cur_patient_dir = os.path.join(input_dir, patient_id)
    if not os.path.exists(cur_patient_dir):
        return classified_list
    
    date_list = [date for date in os.listdir(cur_patient_dir) 
                 if os.path.isdir(os.path.join(cur_patient_dir, date))]
    date_list.sort()
    
    cur_working_dir = os.path.join(working_dir, patient_id)
    cur_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(cur_working_dir, exist_ok=True)
    os.makedirs(cur_output_dir, exist_ok=True)

    for date in date_list:
        cur_date_dir = os.path.join(cur_patient_dir, date)
        cur_working_date_dir = os.path.join(cur_working_dir, date)
        cur_output_date_dir = os.path.join(cur_output_dir, date)
        os.makedirs(cur_working_date_dir, exist_ok=True)
        os.makedirs(cur_output_date_dir, exist_ok=True)
        
        sub_dir_list = [d for d in os.listdir(cur_date_dir) 
                       if os.path.isdir(os.path.join(cur_date_dir, d))]
        
        for sub_dir in sub_dir_list:
            cur_sub_input_dir = os.path.join(cur_date_dir, sub_dir)
            files = [os.path.join(cur_sub_input_dir, f) 
                    for f in os.listdir(cur_sub_input_dir) 
                    if f.endswith('.dcm')]
            
            # 파일이 충분하지 않으면 스킵
            if len(files) < 2:
                continue
            
            # Axial 이미지 확인
            if not check_axial(files[0], files[1]):
                continue
            
            cur_working_sub_dir = os.path.join(cur_working_date_dir, sub_dir)
            cur_output_sub_dir = os.path.join(cur_output_date_dir, sub_dir)
            os.makedirs(cur_working_sub_dir, exist_ok=True)
            os.makedirs(cur_output_sub_dir, exist_ok=True)
            
            # DICOM 파일 처리
            classify_dict, slide_location_dict, summary, size = _process_dicom_files(
                files, cur_working_sub_dir, cur_output_sub_dir)
            
            # 정렬 및 복사
            for category in ['in', 'ex']:
                if classify_dict[category]:
                    sorted_target_list = sorted(classify_dict[category], key=lambda x: x[2])
                    order = check_order(sorted_target_list)
                    sort_flag = order != 'top'
                    classify_dict[category], slide_location_dict[category] = copy_to_working_dir(
                        cur_working_sub_dir, sorted_target_list, sort_flag)
            
            _date_list.append(date)
            _sub_list.append(sub_dir)
            _summary_list.append(summary)
            classified_list.append((
                cur_sub_input_dir, cur_working_sub_dir, cur_output_sub_dir, 
                classify_dict, slide_location_dict, size))
    
    console_classify(console, _date_list, _sub_list, _summary_list)
    return classified_list