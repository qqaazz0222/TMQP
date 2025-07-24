import subprocess

def generate_lungmask(input_dir: str, output_path: str):
    """
    폐 마스크를 생성하는 함수입니다.

    Args:
        input_dir (str): 입력 디렉터리 경로
        output_path (str): 출력 파일 경로
    """
    try:
        subprocess.run([
            "lungmask",  # lungmask CLI 명령어
            input_dir,  # 입력 폴더 (DICOM)
            output_path,  # 출력 DICOM 파일 경로
            "--modelname", "R231"  # 기본 모델 지정
        ], check=True)
        return True
    except Exception as e:
        raise e