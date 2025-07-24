import os
import pydicom
import numpy as np
import cv2 as cv
from PIL import Image
from torchvision import transforms
from sklearn.cluster import DBSCAN
from skimage.morphology import convex_hull_image
from collections import Counter

def get_dicom_files(dir: str):
    """
    디렉토리 내 DICOM 파일들을 반환하는 함수

    Args:
        dir (str): 디렉터리 경로

    Returns:
        list: DICOM 파일 경로 리스트
    """
    files = os.listdir(dir)
    file_list = []
    if len(files) > 0:
        file_list = [os.path.join(dir, file) for file in files if file.endswith('.dcm')]
    file_list.sort()
    return file_list

def get_dicom_data(file_path: str):
    """
    DICOM 파일 데이터를 반환하는 함수

    Args:
        file_path (str): DICOM 파일 경로

    Returns:
        pydicom.dataset.FileDataset: DICOM 데이터셋
    """
    return pydicom.dcmread(file_path)

def get_dicom_data_list(dir: str):
    """
    디렉토리 내 DICOM 데이터셋들을 반환하는 함수

    Args:
        dir (str): 디렉터리 경로

    Returns:
        list: DICOM 데이터셋 리스트
    """
    files = get_dicom_files(dir)
    files.sort()

    data_list = [get_dicom_data(file) for file in files]
    data_list.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    return data_list

def get_dicom_series(data_list: list, is_adjusted: bool = False):
    """
    DICOM 시리즈를 반환하는 함수

    Args:
        dir (str): 디렉터리 경로

    Returns:
        pydicom.dataset.FileDataset: DICOM 데이터셋
    """
    pixel_array_list = np.stack([s.pixel_array for s in data_list], axis=-1)

    for i, d in enumerate(data_list):
        pixel_array_list[:, :, i] = get_pixels(d, from_DICOM_data = True, adjusted = is_adjusted)

    return pixel_array_list

def get_body_mask_series_with_DBSCAN(data_list: list):
    """
    DBSCAN을 이용한 Body Mask 시리즈를 반환하는 함수

    Args:
        data_list (list): DICOM 데이터셋 리스트

    Returns:
        numpy.ndarray: Body Mask 시리즈
    """

    pixel_array_list = np.stack([s.pixel_array for s in data_list], axis=-1)

    for i, d in enumerate(data_list):     
        pixel_array_list[:, :, i] = get_body_mask_with_DBSAN(d, from_DICOM_data = True)

    return pixel_array_list

def get_dicom_idx(file: str):
    """
    DICOM 파일의 인덱스를 반환하는 함수

    Args:
        file (str): DICOM 파일 경로

    Returns:
        int: DICOM 파일 인덱스
    """
    data = get_dicom_data(file) 
    idx_data = data.get((0x0020, 0x0013), "Unknown")
    idx = idx_data.value
    return file, idx
    

def get_pixels(target, adjusted: bool = False, from_DICOM_data: bool = False,):
    """
    DICOM 데이터셋으로부터 픽셀 데이터를 반환하는 함수

    Args:
        data (pydicom.dataset.FileDataset): DICOM 데이터셋
        from_DICOM_data (bool): DICOM 데이터셋에서 픽셀 데이터를 가져올지 여부
        adjusted (bool): 픽셀 데이터를 조정할지 여부

    Returns:
        numpy.ndarray: 픽셀 데이터
    """
    if from_DICOM_data == False:
        ds = pydicom.dcmread(target)
    else:
        ds = target

    pixels = ds.pixel_array.astype(np.int16)
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept
    pixels = slope * pixels + intercept

    if adjusted == True:
        if('WindowCenter' in ds):
            if(type(ds.WindowCenter) == pydicom.multival.MultiValue):
                window_center = float(ds.WindowCenter[0])
                window_width = float(ds.WindowWidth[0])    
            else:    
                window_center = float(ds.WindowCenter)
                window_width = float(ds.WindowWidth)
            
            lwin = window_center - (window_width / 2.0)
            rwin = window_center + (window_width / 2.0)
        else:
            lwin = np.min(pixels)
            rwin = np.max(pixels)
        
        pixels[np.where(pixels < lwin)] = lwin
        pixels[np.where(pixels > rwin)] = rwin
        pixels = pixels - lwin


    # 이미지 방향 확인
    if 'ImageOrientationPatient' in ds:
        orientation = ds.ImageOrientationPatient
        
        # 일반적인 상하 또는 좌우 뒤집힘의 경우 처리
        if orientation == [1, 0, 0, 0, -1, 0]:
            # 상하 뒤집힘: 수직 반전
            pixels = np.flipud(pixels)
        elif orientation == [-1, 0, 0, 0, 1, 0]:
            # 좌우 뒤집힘: 수평 반전
            pixels = np.fliplr(pixels)

    return pixels

def get_pixels_with_window(target, window_center, window_width, from_DICOM_data = False):
    """
    DICOM 데이터셋으로부터 픽셀 데이터를 반환하는 함수

    Args:
        target (pydicom.dataset.FileDataset): DICOM 데이터셋
        from_DICOM_data (bool): DICOM 데이터셋에서 픽셀 데이터를 가져올지 여부
        window_center (int): 윈도우 중심값
        window_width (int): 윈도우 폭

    Returns:
        numpy.ndarray: 픽셀 데이터
    """
    if from_DICOM_data == False:
        ds = pydicom.dcmread(target)
    else:
        ds = target
    
    # Extract the pixel data and convert to Hounsfield units
    pixels = ds.pixel_array.astype(np.int16)
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept
    pixels = slope * pixels + intercept
    
    lwin = window_center - (window_width / 2.0)
    rwin = window_center + (window_width / 2.0)
    
    pixels[np.where(pixels < lwin)] = lwin
    pixels[np.where(pixels > rwin)] = rwin
    pixels = pixels - lwin
            

    # 이미지 방향 확인
    if 'ImageOrientationPatient' in ds:
        orientation = ds.ImageOrientationPatient
        
        # 일반적인 상하 또는 좌우 뒤집힘의 경우 처리
        if orientation == [1, 0, 0, 0, -1, 0]:
            # 상하 뒤집힘: 수직 반전
            pixels = np.flipud(pixels)
        elif orientation == [-1, 0, 0, 0, 1, 0]:
            # 좌우 뒤집힘: 수평 반전
            pixels = np.fliplr(pixels)

    return pixels

def get_body_mask_with_DBSAN(target, from_DICOM_data: bool = False):
    """
    DBSCAN을 이용한 Body Mask를 반환하는 함수

    Args:
        target (pydicom.dataset.FileDataset): DICOM 데이터셋
        from_DICOM_data (bool): DICOM 데이터셋에서 픽셀 데이터를 가져올지 여부

    Returns:
        numpy.ndarray: Body Mask
    """
    pixels = get_pixels_with_window(target, 20, 1000, from_DICOM_data)
    
    _, thresh = cv.threshold(pixels, 600, 4000, cv.THRESH_TOZERO)
    
    kernel = np.ones((3,3),np.uint8)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    pixelsXor = cv.bitwise_xor(pixels, closing)
    opening = cv.morphologyEx(pixelsXor, cv.MORPH_OPEN, kernel)
    
    pixels_normalized = cv.normalize(opening, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    _, binary_image = cv.threshold(pixels_normalized, 127, 255, cv.THRESH_BINARY)

    points = np.column_stack(np.where(binary_image > 0))

    dbscan = DBSCAN(eps=5, min_samples=10)
    labels = dbscan.fit_predict(points)

    label_counts = Counter(labels)
    if -1 in label_counts:
        del label_counts[-1]
    most_common_label = label_counts.most_common(1)[0][0]

    target_label = most_common_label
    mask = (labels == target_label)

    clustered_image = np.zeros_like(binary_image)
    clustered_image[points[mask][:, 0], points[mask][:, 1]] = 255
    cvxHull = convex_hull_image(clustered_image)

    return cvxHull

def convert_dicom_to_tensor(path: str):
    """
    DICOM 파일을 Tensor로 변환하는 함수

    Args:
        path (str): DICOM 파일 경로

    Returns:
        torch.Tensor: 전처리된 DICOM 이미지
    """
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    weight = np.sum(img < 200) / img.size
    img = np.stack([img] * 3, axis=-1)
    img = (img / np.max(img) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    return transform(img).unsqueeze(0), weight