import os
from pathlib import Path
from typing import Dict, Optional
from utils.directoryHandler import check_checkpoint
from utils.dicomHandler import get_dicom_files
from utils.lungmask import generate_lungmask
from utils.logger import log_execution_time, log

@log_execution_time
def create_lung_mask(working_dir: str) -> Dict[str, Dict[str, str]]:
    """
    전처리를 수행하는 함수

    Args:
        working_dir (str): 작업 디렉터리 경로(./data/working/{환자 ID})

    Returns:
        Dict[str, Dict[str, str]]: 전처리 메타데이터
    """
    try:
        # 입력 유효성 검사
        if not working_dir or not Path(working_dir).exists():
            log("error", f"Invalid working directory: {working_dir}")
            return {}
        
        metadata = {
            "in": {
                "working_dir": os.path.join(working_dir, "in"),
                "lungmask_path": os.path.join(working_dir, "output_HU_in.dcm")
            },
            "ex": {
                "working_dir": os.path.join(working_dir, "ex"),
                "lungmask_path": os.path.join(working_dir, "output_HU_ex.dcm")
            }
        }

        for folder_type in ['in', 'ex']:
            try:
                lungmask_path = os.path.join(working_dir, f"output_HU_{folder_type}.dcm")
                
                if not check_checkpoint(lungmask_path):
                    log("info", f"Generating lung mask for {folder_type}")
                    
                    # 작업 디렉토리 존재 확인
                    folder_dir = metadata[folder_type]["working_dir"]
                    if not Path(folder_dir).exists():
                        log("warning", f"Working directory not found: {folder_dir}")
                        continue
                    
                    try:
                        generate_lungmask(working_dir, lungmask_path)
                        log("success", f"Generated lung mask: {lungmask_path}")
                    except Exception as e:
                        log("error", f"Failed to generate lung mask for {folder_type}: {e}")
                        continue
                else:
                    log("info", f"Using existing lung mask: {lungmask_path}")
                    
            except Exception as e:
                log("error", f"Error processing {folder_type}: {e}")
                continue
        
        return metadata
        
    except Exception as e:
        log("error", f"Critical error in create_lung_mask: {e}")
        return {}