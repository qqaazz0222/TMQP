import os
import cv2
import math
import numpy
import pydicom
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, cKDTree
import time
from typing import List, Tuple, Optional, Union
from pathlib import Path
import warnings
from utils.logger import log

# 경고 메시지 억제
warnings.filterwarnings('ignore', category=RuntimeWarning)

def split_by_low_regions(arr: numpy.ndarray, threshold: int = 128) -> List[Tuple]:
    """
    주어진 1D NumPy 배열을 threshold 이하 구간을 경계로 하여 분할하는 함수
    
    Args:
        arr (numpy.ndarray): 1D NumPy array
        threshold (int): 경계로 사용할 값 (기본값: 128)
    
    Returns: 
        List[Tuple]: threshold 이상 구간의 (시작인덱스, 배열) 튜플 리스트
    """
    try:
        if arr is None or len(arr) == 0:
            return []
            
        mask = arr > threshold
        segments = []
        N = arr.shape[0]
        i = 0
        
        while i < N:
            if mask[i]:
                start = i
                while i < N and mask[i]:
                    i += 1
                end = i
                segment = arr[start:end]
                if len(segment) > 6:  # 최소 길이 체크
                    segments.append((start, segment))
            else:
                i += 1
                
        return segments
        
    except Exception as e:
        print(f"Error in split_by_low_regions: {e}")
        return []

def calc_boundary(dicom_list: List[str]) -> List[int]:
    """
    DICOM 파일 리스트로부터 경계를 계산하는 함수
    
    Args:
        dicom_list (List[str]): DICOM 파일 경로 리스트
        
    Returns:
        List[int]: 각 x 좌표에 대한 경계 y 좌표 리스트
    """
    try:
        boundary = []
        
        if not dicom_list:
            return boundary
            
        distance_list = []
        pixel_array_shape = None
        
        # 거리 계산
        for file_path in dicom_list:
            try:
                if not Path(file_path).exists():
                    continue
                    
                dcm = pydicom.dcmread(file_path)
                pixel_array = dcm.pixel_array
                
                if pixel_array is None:
                    continue
                    
                h, w = pixel_array.shape
                pixel_array_shape = (h, w)
                
                target_x = w // 2
                target_col = pixel_array[:, target_x]
                segmented_col = split_by_low_regions(target_col)
                
                if not segmented_col:
                    distance = h
                else:
                    segmented_col = sorted(segmented_col, reverse=True, key=lambda x: len(x[1]))
                    distance = h
                    
                    for start_idx, segment_arr in segmented_col:
                        if len(segment_arr) < 48:
                            distance = start_idx
                        else:
                            break
                
                distance_list.append(distance)
                
            except Exception as e:
                print(f"Error processing DICOM file {file_path}: {e}")
                continue
        
        if not distance_list or pixel_array_shape is None:
            return boundary
            
        # 평균 거리 계산
        distance = numpy.mean(distance_list)
        distance = max(8, int(distance) - 8)  # 최소값 보장
        
        h, w = pixel_array_shape
        
        # 마스크 생성
        mask = numpy.zeros((h, w), dtype=numpy.uint8)
        shift = 148
        center = (w // 2, -shift)
        radius = distance + shift
        
        cv2.circle(mask, center, radius, 255, -1)
        
        # 경계 계산
        for x in range(w):
            boundary_y = h - 8  # 기본값
            for y in range(h):
                if mask[y, x] != 255:
                    boundary_y = max(0, y - 8)
                    break
            boundary.append(boundary_y)
    
        return boundary
        
    except Exception as e:
        print(f"Error in calc_boundary: {e}")
        return []

def remove_bottom_arc(pixel_array: numpy.ndarray, boundary: list):
    """
    CT 이미지 하단의 아치형 구조를 제거하는 함수

    Args:
        pixel_array (numpy.ndarray): CT 이미지 배열

    Returns:
        numpy.ndarray: 아치형 구조가 제거된 CT 이미지 배열
    """
    h, w = pixel_array.shape
    for x in range(w):
        boundary_y = boundary[x]
        pixel_array[boundary_y:, x] = 0
    
    return pixel_array

def check_sternum(pixel_array: numpy.ndarray):
    """
    CT 이미지에서 흉골가 있는지 확인하는 함수

    Args:
        pixel_array (numpy.ndarray): CT 이미지 배열

    Returns:
        bool: 흉골이 있는 경우 True, 없는 경우 False
    """
    _, w = pixel_array.shape
    target_range = [[w//2 - 32, 0], [w//2 + 32, 250]]
    x1, y1, x2, y2 = target_range[0][0], target_range[0][1], target_range[1][0], target_range[1][1]
    target_area = pixel_array[y1:y2, x1:x2]
    bone_hu_value = [[1200, 3000]]
    count = 0
    for hu_range in bone_hu_value:
        count += numpy.sum((target_area >= hu_range[0]) & (target_area <= hu_range[1]))
    return count > 64

def find_heart(pixel_array: numpy.ndarray):
    """
    CT 이미지에서 심장을 찾는 함수

    Args:
        pixel_array (numpy.ndarray): CT 이미지 배열

    Returns:
        numpy.ndarray: 심장 마스크
    """
    hu = [1000, 1100]
    target_area = [[150, 100], [412, 356]] # LTRB
    heart_mask = numpy.zeros_like(pixel_array, dtype=numpy.uint8)
    heart_mask[(pixel_array >= hu[0]) & (pixel_array <= hu[1])] = 1
    reversed_heart_mask = numpy.zeros_like(pixel_array, dtype=numpy.uint8)
    reversed_heart_mask[(pixel_array < hu[0]) | (pixel_array > hu[1])] = 1

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(reversed_heart_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= 64:
            heart_mask[labels == i] = 0
    x1, y1, x2, y2 = target_area[0][0], target_area[0][1], target_area[1][0], target_area[1][1]
    cropped_mask = heart_mask[y1:y2, x1:x2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cropped_mask, connectivity=8)
    largest_label = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(numpy.uint8)
    largest_component_size = stats[largest_label, cv2.CC_STAT_AREA]
    heart_mask[:, :] = 0
    heart_mask[y1:y2, x1:x2] = largest_component
    contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heart_mask, contours, -1, 1000, -1)
    for contour in contours:
        for point in contour:
            point[0][0] += x1
            point[0][1] += y1
    heart_mask = numpy.zeros_like(pixel_array, dtype=numpy.uint8)
    if largest_component_size > 5000:
        cv2.drawContours(heart_mask, contours, -1, 1, -1)
        cv2.drawContours(heart_mask, contours, -1, 1, 3)
    return heart_mask

def masking_heart(pixel_array: numpy.ndarray, normalized_pixel_array: numpy.ndarray, heart_mask: numpy.ndarray):
    """
    심장을 마스킹하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        normalized_pixel_array (numpy.ndarray): 정규화된 이미지 배열
        heart_mask (numpy.ndarray): 심장 마스크

    Returns:
        numpy.ndarray: 심장이 마스킹된 이미지 배열
        numpy.ndarray: 심장이 마스킹된 정규화된 이미지 배열
    """
    mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    mask[heart_mask == 1] = 255
    pixel_array[mask == 255] = 0
    normalized_pixel_array[mask == 255] = [0, 0, 0]
    return pixel_array, normalized_pixel_array

def find_liver(pixel_array: numpy.ndarray):
    """
    CT 이미지에서 간을 찾는 함수

    Args:
        pixel_array (numpy.ndarray): CT 이미지 배열

    Returns:
        numpy.ndarray: 간 마스크
    """
    hu = [1000, 1100]
    target_area = [[150, 100], [464, 420]] # LTRB
    liver_mask = numpy.zeros_like(pixel_array, dtype=numpy.uint8)
    liver_mask[(pixel_array >= hu[0]) & (pixel_array <= hu[1])] = 1
    reversed_liver_mask = numpy.zeros_like(pixel_array, dtype=numpy.uint8)
    reversed_liver_mask[(pixel_array < hu[0]) | (pixel_array > hu[1])] = 1

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(reversed_liver_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= 64:
            liver_mask[labels == i] = 0
    x1, y1, x2, y2 = target_area[0][0], target_area[0][1], target_area[1][0], target_area[1][1]
    cropped_mask = liver_mask[y1:y2, x1:x2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cropped_mask, connectivity=8)
    largest_label = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(numpy.uint8)
    largest_component_size = stats[largest_label, cv2.CC_STAT_AREA]
    liver_mask[:, :] = 0
    liver_mask[y1:y2, x1:x2] = largest_component
    contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(liver_mask, contours, -1, 1000, -1)
    for contour in contours:
        for point in contour:
            point[0][0] += x1
            point[0][1] += y1
    liver_mask = numpy.zeros_like(pixel_array, dtype=numpy.uint8)
    if largest_component_size > 5000:
        cv2.drawContours(liver_mask, contours, -1, 1, -1)
        cv2.drawContours(liver_mask, contours, -1, 1, 3)
    return liver_mask

def masking_liver(pixel_array: numpy.ndarray, normalized_pixel_array: numpy.ndarray, liver_mask: numpy.ndarray):
    """
    간을 마스킹하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        normalized_pixel_array (numpy.ndarray): 정규화된 이미지 배열
        liver_mask (numpy.ndarray): 간 마스크

    Returns:
        numpy.ndarray: 간이 마스킹된 이미지 배열
        numpy.ndarray: 간이 마스킹된 정규화된 이미지 배열
    """
    mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    mask[liver_mask == 1] = 255
    pixel_array[mask == 255] = 0
    normalized_pixel_array[mask == 255] = [0, 0, 0]
    return pixel_array, normalized_pixel_array

def find_contours(pixel_array: numpy.ndarray):
    """
    이미지에서 최외곽 윤곽선을 찾는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열

    Returns:
        list: 윤곽선 리스트
    """
    if len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
        gray_pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2GRAY)
    else:
        gray_pixel_array = pixel_array
    gray_pixel_array = cv2.copyMakeBorder(gray_pixel_array, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    low_value_areas = gray_pixel_array <= 32
    binary_image = numpy.zeros_like(gray_pixel_array)
    binary_image[low_value_areas] = 255
    binary_image = binary_image.astype(numpy.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 256]
    contours = [contour for contour in contours if not any((point[0][0] == 0 and point[0][1] == 0) or (point[0][0] == 511 and point[0][1] == 511) for point in contour)]
    for contour in contours:
        for point in contour:
            point[0][0] -= 10
            point[0][1] -= 10
    return contours

def fill_contours(pixel_array: numpy.ndarray, contours: list):
    """
    이미지 윤곽선을 채우는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        contours (list): 윤곽선 리스트

    Returns:
        numpy.ndarray: 윤곽선이 채워진 이미지 배열
    """
    try:
        inner_contour = contours[1:]
        # cv2.drawContours(pixel_array, contours, 0, (0, 0, 0), 8, cv2.LINE_8)
        cv2.drawContours(pixel_array, inner_contour, -1, (0, 0, 0), -1)
        left_inner_contour = [contour for contour in inner_contour if cv2.boundingRect(contour)[0] < pixel_array.shape[1] // 2]
        largest_left_contour = max(left_inner_contour, key=cv2.contourArea)
        right_inner_contour = [contour for contour in inner_contour if cv2.boundingRect(contour)[0] >= pixel_array.shape[1] // 2]
        largest_right_contour = max(right_inner_contour, key=cv2.contourArea)
        left_rect = cv2.boundingRect(largest_left_contour)
        right_rect = cv2.boundingRect(largest_right_contour)
        left_point = (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3] // 2)
        right_point = (right_rect[0], right_rect[1] + right_rect[3] // 2)
        mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
        cv2.drawinner_contour(mask, [largest_left_contour], -1, 255, -1)
        cv2.drawinner_contour(mask, [largest_right_contour], -1, 255, -1)
        cv2.line(mask, left_point, right_point, 255, thickness=2)
        pixel_array[mask == 255] = [0, 0, 0]
    except: # 컨투어가 없을때 예외처리
        pass
    return pixel_array

def remove_skin(pixel_array: numpy.ndarray, contours: list, method: str, thickness: int = 16):
    """
    이미지에서 피부를 제거하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        contours (list): 윤곽선 리스트
        method (str): 처리 방법
        thickness (int): 선 두께

    Returns:
        numpy.ndarray: 피부가 제거된 이미지 배열
    """
    try:
        if not contours or len(contours) == 0:
            log("warning", "No contours found for skin removal")
            return pixel_array
            
        if method == "upper":
            thickness = 20
        
        h, w = pixel_array.shape[:2]
        body_outline_contour = contours[0]
        
        # 백 라인 포인트 찾기
        back_line_points = [point for point in body_outline_contour if len(point) > 0 and len(point[0]) >= 2 and point[0][1] >= h // 2 + 96]
        if len(back_line_points) < 2:
            back_line_points = [point for point in body_outline_contour if len(point) > 0 and len(point[0]) >= 2 and point[0][1] >= h // 2 + 48]
        
        if len(back_line_points) < 2:
            log("warning", "Insufficient back line points for skin removal")
            return pixel_array
        
        # 피부 제거
        cv2.drawContours(pixel_array, [body_outline_contour], -1, 0, 8, cv2.LINE_8)
        
        # 등 피부 제거
        for i in range(len(back_line_points) - 1):
            try:
                cv2.line(pixel_array, tuple(back_line_points[i][0]), tuple(back_line_points[i + 1][0]), 0, thickness)
            except Exception as e:
                log("warning", f"Error drawing back line {i}: {e}")
                continue
        
        # 포인트 추출
        left_point = min(back_line_points, key=lambda point: point[0][0])[0]
        right_point = max(back_line_points, key=lambda point: point[0][0])[0]
        middle_point = min(back_line_points, key=lambda point: abs(point[0][0] - w // 2))[0].copy()
        middle_point[1] += 12
        
        bottom_points = []
        left_x = [left_point[0], middle_point[0]]
        right_x = [middle_point[0], right_point[0]]
        total_x = [left_point[0], right_point[0]]
        
        # 거리 계산
        l_dx = middle_point[0] - left_point[0]
        l_dy = middle_point[1] - left_point[1]
        r_dx = right_point[0] - middle_point[0]
        r_dy = right_point[1] - middle_point[1]
        
        # 안전한 제곱근 계산
        def safe_sqrt(value):
            return math.sqrt(max(0, value))
        
        l_length = safe_sqrt(l_dx**2 + l_dy**2)
        r_length = safe_sqrt(r_dx**2 + r_dy**2)
        
        sin60 = math.sqrt(3) / 2
        cos60 = 0.5
            
        if method == "over":
            # 왼쪽 삼각 포인트 계산
            l_third_point_x1 = left_point[0] + l_dx * cos60 - l_dy * sin60
            l_third_point_y1 = left_point[1] + l_dx * sin60 + l_dy * cos60
            l_third_point1 = (int(l_third_point_x1), int(l_third_point_y1))
            l_third_point_x2 = left_point[0] + l_dx * cos60 + l_dy * sin60
            l_third_point_y2 = left_point[1] - l_dx * sin60 + l_dy * cos60
            l_third_point2 = (int(l_third_point_x2), int(l_third_point_y2))
            l_third_point = l_third_point1 if l_third_point1[1] < l_third_point2[1] else l_third_point2
            
            # 오른쪽 삼각 포인트 계산
            r_third_point_x1 = middle_point[0] + r_dx * cos60 - r_dy * sin60
            r_third_point_y1 = middle_point[1] + r_dx * sin60 + r_dy * cos60
            r_third_point1 = (int(r_third_point_x1), int(r_third_point_y1))
            r_third_point_x2 = middle_point[0] + r_dx * cos60 + l_dy * sin60
            r_third_point_y2 = middle_point[1] - r_dx * sin60 + r_dy * cos60
            r_third_point2 = (int(r_third_point_x2), int(r_third_point_y2))
            r_third_point = r_third_point1 if r_third_point1[1] < r_third_point2[1] else r_third_point2
            
            # 왼쪽 영역 처리
            for x in range(left_x[0], left_x[1] + 1):
                dx = x - l_third_point[0]
                # 안전한 제곱근 계산
                discriminant = l_length**2 - dx**2
                dy = safe_sqrt(discriminant) if discriminant >= 0 else 0
                bottom_points.append((x, int(l_third_point[1] + dy)))

            # 오른쪽 영역 처리
            for x in range(right_x[0], right_x[1] + 1):
                dx = x - r_third_point[0]
                # 안전한 제곱근 계산
                discriminant = r_length**2 - dx**2
                dy = safe_sqrt(discriminant) if discriminant >= 0 else 0
                bottom_points.append((x, int(r_third_point[1] + dy)))
        else:
            # 원 계산 방법
            A = left_point[0] - middle_point[0]
            B = left_point[1] - middle_point[1]
            C = right_point[0] - middle_point[0]
            D = right_point[1] - middle_point[1]
            E = A * (left_point[0] + middle_point[0]) + B * (left_point[1] + middle_point[1])
            F = C * (right_point[0] + middle_point[0]) + D * (right_point[1] + middle_point[1])
            G = 2 * (A * (right_point[1] - middle_point[1]) - B * (right_point[0] - middle_point[0]))

            if abs(G) > 1e-10:  # 0에 가까운 값 체크
                circle_center_x = int((D * E - B * F) / G)
                circle_center_y = int((A * F - C * E) / G)
                circle_center = (circle_center_x, circle_center_y)

                # 안전한 반지름 계산
                radius_squared = (circle_center_x - left_point[0])**2 + (circle_center_y - left_point[1])**2
                radius = int(safe_sqrt(radius_squared))
                
                for x in range(total_x[0], total_x[1] + 1):
                    dx = x - circle_center[0]
                    # 안전한 제곱근 계산
                    discriminant = radius**2 - dx**2
                    dy = safe_sqrt(discriminant) if discriminant >= 0 else 0
                    bottom_points.append((x, int(circle_center[1] + dy)))
            else:
                log("warning", "Invalid circle calculation, using fallback")
                # 폴백: 직선으로 연결
                for x in range(total_x[0], total_x[1] + 1):
                    y = int(left_point[1] + (right_point[1] - left_point[1]) * (x - left_point[0]) / max(1, right_point[0] - left_point[0]))
                    bottom_points.append((x, y))
        
        # 바닥 포인트 완성
        bottom_points.append((w, h))
        bottom_points.append((0, h))
        
        # 폴리곤 채우기
        try:
            cv2.fillPoly(pixel_array, [numpy.array(bottom_points, dtype=numpy.int32)], 0)
        except Exception as e:
            log("warning", f"Error filling polygon: {e}")
        
        return pixel_array
        
    except Exception as e:
        log("error", f"Critical error in remove_skin: {e}")
        return pixel_array

def find_muscle(pixel_array: numpy.ndarray, is_lung: bool):
    """
    HU 값으로 근육을 찾는 함수(폐가 있는 경우)

    Args:
        pixel_array (numpy.ndarray): 이미지 배열

    Returns:
        numpy.ndarray: 근육 마스크
    """
    if is_lung:
        muscle_hu_value = [[1000, 1200]]
    else:
        muscle_hu_value = [[1020, 1200]]
    muscle = muscle_hu_value[0]
    muscle_mask = numpy.zeros(pixel_array.shape)
    for i in range(pixel_array.shape[0]):
        for j in range(pixel_array.shape[1]):
            if muscle[0] <= pixel_array[i, j] <= muscle[1]:
                muscle_mask[i, j] = 1
    muscle_mask = cv2.normalize(muscle_mask, None, 0, 1, cv2.NORM_MINMAX)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(muscle_mask.astype(numpy.uint8), connectivity=8)
    back_skin_idx = -1
    flag = True
    for y in range(labels.shape[0] - 1, -1, -1):
        for x in range(labels.shape[1]):
            if back_skin_idx == -1:
                if labels[y, x] != 0:
                    back_skin_idx = labels[y, x]
                    if stats[back_skin_idx, cv2.CC_STAT_AREA] > 3000:
                        flag = False
                        break
                    muscle_mask[y, x] = 0
            else:
                if labels[y, x] == back_skin_idx:
                    muscle_mask[y, x] = 0
        if not flag:
            break
    
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            muscle_mask[labels == i] = 0
    return muscle_mask

def fill_muscle(pixel_array: numpy.ndarray, muscle_mask: numpy.ndarray):
    """
    근육을 채우는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        muscle_mask (numpy.ndarray): 근육 마스크

    Returns:
        numpy.ndarray: 근육이 채워진 이미지 배열
    """
    mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    mask[muscle_mask == 1] = 255
    pixel_array[mask == 255] = [255, 0, 0]
    return pixel_array

def find_bone(pixel_array: numpy.ndarray, muscle_mask: numpy.ndarray):
    """
    HU 값으로 뼈를 찾는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        muscle_mask (numpy.ndarray): 근육 마스크

    Returns:
        numpy.ndarray: 뼈 마스크
    """
    bone_hu_value = [[300, 400], [1130, 3000]]
    # cancellous_bone = bone_hu_value[0]
    cortical_bone = bone_hu_value[1]
    bone_mask = numpy.zeros(pixel_array.shape)
    for i in range(pixel_array.shape[0]):
        for j in range(pixel_array.shape[1]):
            if cortical_bone[0] <= pixel_array[i, j] <= cortical_bone[1]:
                bone_mask[i, j] = 1
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bone_mask.astype(numpy.uint8), connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 64:
            bone_mask[labels == i] = 0
    contours, _ = cv2.findContours(bone_mask.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 64]
    
    filtered_contours = []
    for contour in contours:
        contour_center = numpy.mean(contour, axis=0).astype(int)
        if contour_center[0][1] < 200:
            filtered_contours.append(contour)
        else:
            cur_pixels = muscle_mask.copy()
            contour_size = cv2.contourArea(contour)
            contour_pixels = numpy.zeros_like(cur_pixels)
            cur_pixels[contour_pixels == 0] = 0
            active_pixel_count = numpy.sum(cur_pixels)
            if active_pixel_count / contour_size < 1:
                filtered_contours.append(contour)

    filtered_contours = contours
    cv2.drawContours(bone_mask, filtered_contours, -1, 1, -1)
    cv2.drawContours(bone_mask, filtered_contours, -1, 1, 1)
    return bone_mask, filtered_contours

def masking_bone(pixel_array: numpy.ndarray, bone_mask: numpy.ndarray, bone_contours: list = None):
    """
    뼈를 마스킹하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        bone_mask (numpy.ndarray): 뼈 마스크

    Returns:
        numpy.ndarray: 뼈가 마스킹된 이미지 배열
    """
    if bone_contours is not None:
        cv2.drawContours(pixel_array, bone_contours, -1, (0, 0, 0), 3)
        cv2.drawContours(pixel_array, bone_contours, -1, (0, 0, 0), -1)
    mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    mask[bone_mask == 1] = 255
    pixel_array[mask == 255] = [0, 0, 0]
    return pixel_array

def masking_bone_inner_area(pixel_array: numpy.ndarray, bone_inner_points: numpy.ndarray):
    """
    갈비뼈 및 척추를 기준으로 내부를 마스킹하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        bone_inner_points (numpy.ndarray): 내부 윤곽선 포인트

    Returns:
        numpy.ndarray: 내부가 마스킹된 이미지 배열
    """
    if len(bone_inner_points) > 2:
        try:
            mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
            cv2.drawContours(mask, [bone_inner_points], -1, 255, -1)
        
            pixel_array[mask == 255] = [0, 0, 0]

            red_pixels = (pixel_array[:, :, 0] == 255) & (pixel_array[:, :, 1] == 0) & (pixel_array[:, :, 2] == 0)
            red_coords = numpy.argwhere(red_pixels)
            
            if len(red_coords) == 0:
                log("warning", "No red pixels found for bone center calculation")
                return pixel_array
                
            bone_center_point = numpy.mean(red_coords, axis=0).astype(int)
            
            # 안전한 거리 계산을 위한 포인트 변환
            try:
                bone_points_2d = numpy.array(bone_inner_points)[:, 0, :]  # (N, 2) 형태로 변환
                bone_center_2d = numpy.array([bone_center_point[1], bone_center_point[0]])  # (y,x) -> (x,y)
                
                distances = numpy.linalg.norm(bone_points_2d - bone_center_2d, axis=1)
                distance = numpy.mean(distances)
                dispersion = numpy.std(distances)
                distance = distance + dispersion

                target_bone_inner_points = [point for i, point in enumerate(bone_inner_points) 
                                          if i < len(distances) and distances[i] < distance]
                                          
                if len(target_bone_inner_points) > 2:
                    try:
                        hull = ConvexHull(numpy.array(target_bone_inner_points)[:, 0, :])
                        mask_convex = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
                        hull_points = bone_inner_points[hull.vertices]
                        cv2.fillPoly(mask_convex, [hull_points], 255)
                        center = pixel_array.shape[1] // 2
                        center_point = min(hull_points, key=lambda point: abs(point[0][0] - center))
                    except Exception as e:
                        log("warning", f"Error in ConvexHull calculation: {e}")
                        return pixel_array
            except Exception as e:
                log("warning", f"Error in distance calculation: {e}")
                return pixel_array
                
        except Exception as e:
            log("warning", f"Error in bone inner area masking: {e}")
            return pixel_array
            center = pixel_array.shape[1] // 2
            center_point = min(hull_points, key=lambda point: abs(point[0][0] - center))
            left_points = [point for point in hull_points if point[0][0] < center_point[0][0]]
            right_points = [point for point in hull_points if point[0][0] > center_point[0][0]]
            if len(left_points) > 0 and len(right_points) > 0:    
                cv2.drawContours(mask_convex, [hull_points], -1, 255, -1)
                left_bottom_point = max(left_points, key=lambda point: point[0][0])
                right_bottom_point = min(right_points, key=lambda point: point[0][0])
                exclude_points = numpy.array([center_point, left_bottom_point, right_bottom_point])
                cv2.fillPoly(mask_convex, [exclude_points], 0)
            pixel_array[mask_convex == 255] = [0, 0, 0]
    return pixel_array

def find_vertebrae_contour(bone_contours: list):
    """
    척추 윤곽선을 찾는 함수

    Args:
        bone_contours (numpy.ndarray): 뼈 윤곽선

    Returns:
        numpy.ndarray: 척추 윤곽선
    """
    vertebrae_area = [[200, 250], [306, 450]] #LTRB
    vertebrae_points = []
    vertebrea_contour = []
    for contour in bone_contours:
        for point in contour:
            x, y = point[0]
            if (vertebrae_area[0][0] <= x <= vertebrae_area[1][0]) and (vertebrae_area[0][1] <= y <= vertebrae_area[1][1]):
                vertebrae_points.append(point)
    total_x, total_y = [], []
    for point in vertebrae_points:
        total_x.append(point[0][0])
        total_y.append(point[0][1])
    try:
        center_x = (max(total_x) + min(total_x)) // 2
        center_y = (max(total_y) + min(total_y)) // 2
        center = (center_x, center_y)
    except:
        center = (256, 320)
    quadrant = [[], [], [], []]
    for point in vertebrae_points:
        x, y = point[0]
        if x >= center[0] and y < center[1]:
            quadrant[0].append(point)
        elif x < center[0] and y < center[1]:
            quadrant[1].append(point)
        elif x < center[0] and y >= center[1]:
            quadrant[2].append(point)
        else:
            quadrant[3].append(point)
    quadrant[0] = sorted(quadrant[0], key=lambda x: x[0][0], reverse=True) # quadrant 1
    quadrant[1] = sorted(quadrant[1], key=lambda x: x[0][0], reverse=True) # quadrant 2
    quadrant[2] = sorted(quadrant[2], key=lambda x: x[0][0]) # quadrant 3
    quadrant[3] = sorted(quadrant[3], key=lambda x: x[0][0]) # quadrant 4
    vertebrea_contour = numpy.array(quadrant[0] + quadrant[1] + quadrant[2] + quadrant[3])
    return numpy.array(vertebrea_contour)

def find_bone_inner_contours(bone_contours: list):
    """
    갈비뼈 및 척추 윤곽선 중 몸 중심을 바라보는 윤곽선 포인트들을 찾는 함수

    Args:
        bone_contours (numpy.ndarray): 뼈 윤곽선

    Returns:
        numpy.ndarray: 몸 중심을 바라보는 윤곽선 포인트들
    """
    vertebrae_area = [240, 260]
    total_x, total_y, vertebrae_bottom, bone_bottom = [], [], 0, 0
    for contour in bone_contours:
        for point in contour:
            x, y = point[0]
            total_x.append(x)
            total_y.append(y)
            if vertebrae_area[0] <= x <= vertebrae_area[1]:
                vertebrae_bottom = max(vertebrae_bottom, y)
            else:
                bone_bottom = max(bone_bottom, y)
                
    try:
        center_x = (max(total_x) + min(total_x)) // 2
        center_y = (max(total_y) + min(total_y)) // 2
        center = (center_x, center_y)    
    except:
        center = (256, 256)
    
    flag = vertebrae_bottom - bone_bottom < 0

    if flag: # 날개뼈가 있는 경우
        block_area = [[[200, 330], [300, 450]], [[80, 350], [200, 450]], [[300, 350], [420, 450]]] # LTRB
    else: # 날개뼈가 없는 경우
        block_area = [[[200, 330], [300, 450]]] # LTRB
    block_area = [[[200, 330], [300, 450]]] # LTRB

    quadrant = [[], [], [], []]
    pixel_array = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
    cv2.drawContours(pixel_array, bone_contours, -1, (0, 0, 0), 1)
    for bone_contour in bone_contours:
        contour_size = cv2.contourArea(bone_contour)
        if contour_size > 64:
            cv2.drawContours(pixel_array, [bone_contour], -1, (0, 0, 0), 2)
            distances = [((point[0][0] - center[0])**2 + (point[0][1] - center[1])**2)**0.5 for point in bone_contour]
            closest_indices = numpy.argsort(distances)[:len(distances)//4]
            if len(closest_indices) > 5:
                for idx in closest_indices:
                    in_block_area = False
                    for area in block_area:
                        if area[0][0] < bone_contour[idx][0][0] < area[1][0] and area[0][1] < bone_contour[idx][0][1] < area[1][1]:
                            in_block_area = True
                            break
                    if not in_block_area:
                        if bone_contour[idx][0][0] >= center[0] and bone_contour[idx][0][1] < center[1]:
                            quadrant[0].append(bone_contour[idx]) 
                        elif bone_contour[idx][0][0] < center[0] and bone_contour[idx][0][1] < center[1]:
                            quadrant[1].append(bone_contour[idx])
                        elif bone_contour[idx][0][0] < center[0] and bone_contour[idx][0][1] >= center[1]:
                            quadrant[2].append(bone_contour[idx])
                        else:
                            quadrant[3].append(bone_contour[idx])
    quadrant[0] = sorted(quadrant[0], key=lambda x: x[0][0], reverse=True) # quadrant 1
    quadrant[1] = sorted(quadrant[1], key=lambda x: x[0][0], reverse=True) # quadrant 2
    quadrant[2] = sorted(quadrant[2], key=lambda x: x[0][0]) # quadrant 3
    quadrant[3] = sorted(quadrant[3], key=lambda x: x[0][0]) # quadrant 4
    bone_inner_points = numpy.array(quadrant[0] + quadrant[1] + quadrant[2] + quadrant[3])
    return bone_inner_points

def masking_vertebrae(pixel_array: numpy.ndarray, vertebrae_contour: numpy.ndarray):
    """
    척추뼈와 척수를 마스킹하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        vertebrae_contour (numpy.ndarray): 척추 윤곽선

    Returns:
        numpy.ndarray: 척추뼈와 척수가 마스킹된 이미지 배열
    """
    mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    if len(vertebrae_contour) > 0:
        cv2.drawContours(mask, [vertebrae_contour], -1, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, 2)
    pixel_array[mask == 255] = [0, 255, 0]
    return pixel_array

def masking_breast(pixel_array: numpy.ndarray, bone_contours: list):
    """
    유방을 마스킹하는 함수
    
    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        bone_contours (numpy.ndarray): 뼈 윤곽선
        
    Returns:
        numpy.ndarray: 유방이 마스킹된 이미지 배열
    """
    # bone_contours를 다 포함하는 폴리곤 생성
    temp_pixels = pixel_array.copy()
    w, h = temp_pixels.shape[:2]
    roi = temp_pixels[:, 224:308, 0]
    if numpy.sum(roi == 255) > 0:
        top_y = numpy.where(temp_pixels[:, 224:308, 0] == 255)[0].min()
    else:
        top_y = h // 2 - 32
    edge_points = numpy.array([[(0, h)], [(0, h // 2 + 32)], [(w//2, top_y - 8)],[(w, h // 2 + 32)], [(w, h)]])
    all_points = numpy.concatenate(bone_contours)
    all_points = numpy.concatenate([all_points, edge_points])
    hull = ConvexHull(all_points[:, 0, :])
    
    scaled_hull_points = (all_points[hull.vertices] - numpy.mean(all_points[hull.vertices], axis=0)) * 1.0875 + numpy.mean(all_points[hull.vertices], axis=0)

    cv2.drawContours(temp_pixels, [scaled_hull_points.astype(int)], -1, (0, 0, 0), -1)
    
    yellow_pixels = (temp_pixels[:, :, 0] == 255) & (temp_pixels[:, :, 1] == 255) & (temp_pixels[:, :, 2] == 0)
    
    hull_top_points = []
    for x in range(0, temp_pixels.shape[1], 64):
        weight = 16 - abs(w // 2 - x)**(1/2)
        y_values = numpy.where(yellow_pixels[:, x])[0]
        if len(y_values) > 0:
            min_y = y_values.min()
            hull_top_points.append((x, min_y - weight))
    hull_top_points.append((w, h // 2))
    
    if len(hull_top_points) > 2:
        hull_top_points = numpy.array(hull_top_points)
        x = hull_top_points[:, 0]
        y = hull_top_points[:, 1]
        coefficients = numpy.polyfit(x, y, 2)  # 2차 함수 계수 계산
        polynomial = numpy.poly1d(coefficients)

        x_fit = numpy.linspace(x.min(), x.max(), 500)
        y_fit = polynomial(x_fit)

        for i in range(len(x_fit) - 1):
            cv2.line(temp_pixels, (int(x_fit[i]), int(y_fit[i])), (int(x_fit[i + 1]), int(y_fit[i + 1])), (0, 0, 255), 2)

    blue_pixels = (temp_pixels[:, :, 0] == 0) & (temp_pixels[:, :, 1] == 0) & (temp_pixels[:, :, 2] == 255)
    blue_pixels = blue_pixels.T
    black_polygon_points = numpy.concatenate([numpy.argwhere(blue_pixels), [(w, h), (0, h),]])
    cv2.fillPoly(temp_pixels, [black_polygon_points], (0, 0, 0))
    
    breast_mask = numpy.zeros(temp_pixels.shape[:2], dtype=numpy.uint8)
    red_pixels = (temp_pixels[:, :, 0] == 255) & (temp_pixels[:, :, 1] == 0) & (temp_pixels[:, :, 2] == 0)
    breast_mask[red_pixels] = 1
    
    pixel_array[breast_mask == 1] = [0, 0, 0]

    return pixel_array

def find_abs(norm_idx: float, pixel_array: numpy.ndarray, bone_contours: list, vertebrae_contour: list):
    """
    앞쪽 갈비뼈를 기준으로 복근을 찾는 함수

    Args:
        norm_idx (float): 정규화 인덱스
        pixel_array (numpy.ndarray): 이미지 배열
        bone_contours (list): 뼈 윤곽선
        vertebrae_contour (list): 척추 윤곽선

    Returns:
        dict: 복근 정보
    """
    try:
        h, w = pixel_array.shape[:2]
        
        # 안전한 기본값 설정
        default_center = (w // 2, h // 2)
        default_abs_info = {
            "center": default_center,
            "left": (-1, -1),
            "right": (-1, -1),
            "left_y_intercept": h // 2,
            "right_y_intercept": h // 2
        }
        
        # 빨간 픽셀 찾기
        red_pixels = (pixel_array[:, :, 0] == 255) & (pixel_array[:, :, 1] == 0) & (pixel_array[:, :, 2] == 0)
        red_coords = numpy.where(red_pixels)
        
        if len(red_coords[0]) == 0:
            log("warning", "No red pixels found for abs detection")
            return default_abs_info
            
        top_y = red_coords[0].min()
        bottom_y = red_coords[0].max()
        
        # 안전한 norm_idx 범위 확인
        norm_idx = max(0.0, min(1.0, norm_idx))
        cur_max_y = int(top_y + norm_idx * (top_y - bottom_y + ((bottom_y - top_y) * 0.4)))
        
        # 척추 중심점 계산
        if len(vertebrae_contour) > 0:
            try:
                vertebrae_center = numpy.mean(vertebrae_contour, axis=0).astype(int)
                center_point = (int(vertebrae_center[0][0]), int(vertebrae_center[0][1]))
            except (IndexError, TypeError):
                center_point = default_center
        else:
            center_point = default_center
        
        # 갈비뼈 윤곽선 처리
        if not bone_contours:
            log("warning", "No bone contours found for abs detection")
            return default_abs_info
        
        try:
            # 좌측 갈비뼈 윤곽선
            front_left_rib_contour = [contour for contour in bone_contours 
                                    if len(contour) > 0 and all(len(point) > 0 and len(point[0]) >= 2 and 
                                                              point[0][0] < w // 2 for point in contour)]
            
            # 우측 갈비뼈 윤곽선  
            front_right_rib_contour = [contour for contour in bone_contours
                                     if len(contour) > 0 and all(len(point) > 0 and len(point[0]) >= 2 and
                                                               point[0][0] >= w // 2 for point in contour)]
            
            if not front_left_rib_contour or not front_right_rib_contour:
                log("warning", "Insufficient rib contours for abs detection")
                return default_abs_info
            
            # 좌측/우측 점 찾기
            front_left_rib_contour = sorted(front_left_rib_contour, 
                                          key=lambda x: min(point[0][1] for point in x if len(point[0]) >= 2))[:1]
            front_right_rib_contour = sorted(front_right_rib_contour,
                                           key=lambda x: min(point[0][1] for point in x if len(point[0]) >= 2))[:1]
            
            left_point = max(front_left_rib_contour[0], key=lambda point: point[0][0])[0]
            right_point = min(front_right_rib_contour[0], key=lambda point: point[0][0])[0]
            
            left_point = (int(left_point[0]), int(left_point[1]))
            right_point = (int(right_point[0]), int(right_point[1]))
            
            # y절편 계산 (0으로 나누기 방지)
            if center_point[0] != left_point[0]:
                left_y_intercept = float(left_point[1] + (center_point[1] - left_point[1]) * 
                                       (0 - left_point[0]) / (center_point[0] - left_point[0]))
            else:
                left_y_intercept = float(left_point[1])
                
            if center_point[0] != right_point[0]:
                right_y_intercept = float(right_point[1] + (center_point[1] - right_point[1]) * 
                                        (w - right_point[0]) / (center_point[0] - right_point[0]))
            else:
                right_y_intercept = float(right_point[1])
                
        except Exception as e:
            log("warning", f"Error in rib contour processing: {e}")
            center_point = default_center
            left_point = (-1, -1)
            right_point = (-1, -1) 
            left_y_intercept = float(cur_max_y)
            right_y_intercept = float(cur_max_y)
        
        abs_info = {
            "center": center_point,
            "left": left_point,
            "right": right_point,
            "left_y_intercept": left_y_intercept,
            "right_y_intercept": right_y_intercept
        }
        
        return abs_info
        
    except Exception as e:
        log("error", f"Critical error in find_abs: {e}")
        h, w = pixel_array.shape[:2] if pixel_array is not None else (512, 512)
        return {
            "center": (w // 2, h // 2),
            "left": (-1, -1),
            "right": (-1, -1),
            "left_y_intercept": h // 2,
            "right_y_intercept": h // 2
        }

def masking_inner_area_from_over_lung(pixel_array: numpy.ndarray, contours: list, bone_contours: list):
    """
    전체 영역 중 폐 상단 영역에서 내부를 마스킹하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        contours (numpy.ndarray): 윤곽선 배열
        bone_contours (numpy.ndarray): 뼈 윤곽선 배열

    Returns:
        numpy.ndarray: 내부가 마스킹된 이미지
    """
    h, w = pixel_array.shape[:2]
    mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    bone_contours = sorted(bone_contours, key=lambda contour: cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] / 2)
    sorted_left_bone_contours = sorted(bone_contours, key=lambda contour: cv2.boundingRect(contour)[0])
    sorted_right_bone_contours = sorted(bone_contours, key=lambda contour: cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2], reverse=True)
    temp_contours = []
    if len(bone_contours) > 7:
        left_bone_1 = sorted_left_bone_contours[0]
        left_bone_2 = sorted_left_bone_contours[1]
        right_bone_1 = sorted_right_bone_contours[0]
        right_bone_2 = sorted_right_bone_contours[1]
        for contour in bone_contours:
            if not numpy.array_equal(contour, left_bone_1) and not numpy.array_equal(contour, left_bone_2) and not numpy.array_equal(contour, right_bone_1) and not numpy.array_equal(contour, right_bone_2):
                temp_contours.append(contour)
    else:
        left_bone_1 = sorted_left_bone_contours[0]
        right_bone_1 = sorted_right_bone_contours[0]
        for contour in bone_contours:
            if not numpy.array_equal(contour, left_bone_1) and not numpy.array_equal(contour, right_bone_1):
                temp_contours.append(contour)
    bone_contours = temp_contours    
    cv2.drawContours(mask, bone_contours, -1, 1, -1)
    if len(bone_contours) > 0:
        all_points = numpy.concatenate(bone_contours)
        convex_points = []
        center_point = numpy.mean(all_points, axis=0, dtype=int)
        center_point = (w // 2, int((center_point[0][1] + h // 2) / 2))
        
        h, w = mask.shape
        for angle in range(0, 360, 4):
            rad = math.radians(angle)
            farthest_point = None
            max_distance = 0
            for r in range(max(h, w)):
                x = int(center_point[0] + r * math.cos(rad))
                y = int(center_point[1] + r * math.sin(rad))
                if 0 <= x < w and 0 <= y < h and mask[y, x] == 1:
                    distance = math.sqrt(max(0, (x - center_point[0])**2 + (y - center_point[1])**2))
                    if distance < 48:
                        continue
                    if distance > max_distance:
                        max_distance = distance
                        farthest_point = (x, y)
            if farthest_point and farthest_point[0] > w // 2 - 64 and farthest_point[0] < w // 2 + 64:
                convex_points.append(farthest_point)
        
        if convex_points:
            convex_points = numpy.array(convex_points, dtype=numpy.int32).reshape((-1, 1, 2))
            cv2.drawContours(pixel_array, [convex_points], -1, (0, 0, 0), -1)
    return pixel_array

def masking_inner_area_from_lung(pixel_array: numpy.ndarray, contours: list, bone_contours: list, vertebrae_contour: numpy.ndarray):
    """
    전체 영역 중 폐 영역에서 내부를 마스킹하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        contours (numpy.ndarray): 윤곽선 배열
        vertebrae_contour (numpy.ndarray): 척추 윤곽선 배열

    Returns:
        numpy.ndarray: 내부가 마스킹된 이미지
    """
    h, w = pixel_array.shape[:2]
    mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    bone_points = numpy.concatenate(bone_contours)
    bone_bbox = cv2.boundingRect(bone_points)
    bone_center_point = (bone_bbox[0] + bone_bbox[2] // 2, bone_bbox[1] + bone_bbox[3] // 2)
    bone_w = int(bone_bbox[2] * 0.45)
    bone_h = int(bone_bbox[3] * 0.45)
    cv2.ellipse(mask, bone_center_point, (bone_w, bone_h), 0, 0, 360, 1, 1)
    boundary_points = numpy.argwhere(mask == 1)
    
    bone_inner_points = []
    quadrants = [[], [], [], []]
    for point in boundary_points:
        y, x = point
        if x >= bone_center_point[0] and y < bone_center_point[1]:
            quadrants[0].append((x, y))  # Top-right
        elif x < bone_center_point[0] and y < bone_center_point[1]:
            quadrants[1].append((x, y))  # Top-left
        elif x < bone_center_point[0] and y >= bone_center_point[1]:
            if x > 180:
                y -= 56
            quadrants[2].append((x, y))  # Bottom-left
        else:
            if x < 338:
                y -= 56
            quadrants[3].append((x, y))  # Bottom-right
    
    quadrants[0] = sorted(quadrants[0], key=lambda x: x[0], reverse=True)  # Top-right
    quadrants[1] = sorted(quadrants[1], key=lambda x: x[0], reverse=True)  # Top-left
    quadrants[2] = sorted(quadrants[2], key=lambda x: x[0])  # Bottom-left
    quadrants[3] = sorted(quadrants[3], key=lambda x: x[0])  # Bottom-right
    sorted_boundary_points = numpy.array(quadrants[0] + quadrants[1] + quadrants[2] + quadrants[3])
    bone_coords = bone_points[:, 0, :]  # Shape (n, 2)
    tree = cKDTree(bone_coords)
    indices = tree.query(sorted_boundary_points, k=1, workers=-1)[1]
    bone_inner_points = bone_points[indices]
    
    cv2.drawContours(pixel_array, [numpy.array(bone_inner_points)], -1, (0, 0, 0), -1)
    if len(contours) >= 3:
        target_lung_contour = contours[1:]
        target_points = numpy.concatenate(target_lung_contour)
        hull = ConvexHull(target_points[:, 0, :])
        cv2.drawContours(pixel_array, [target_points[hull.vertices]], -1, (0, 0, 0), -1)

    try:
        x, y, w, h = cv2.boundingRect(vertebrae_contour)
        center_point = (pixel_array.shape[1] // 2, y + h // 5)
    except:
        center_point = (pixel_array.shape[1] // 2, pixel_array.shape[0] // 2)
    cv2.ellipse(pixel_array, center_point, (w // 3 * 4, h // 2), 0, 0, 360, (0, 0, 0), -1)
    return pixel_array

def masking_inner_area_from_under_lung(pixel_array: numpy.ndarray, contours: list, bone_contours: list):
    """
    전체 영역 중 폐 하단 영역에서 내부를 마스킹하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        contours (numpy.ndarray): 윤곽선 배열
        bone_contours (numpy.ndarray): 뼈 윤곽선 배열

    Returns:
        numpy.ndarray: 내부가 마스킹된 이미지
    """
    green_pixels = (pixel_array[:, :, 0] == 0) & (pixel_array[:, :, 1] == 255) & (pixel_array[:, :, 2] == 0)
    red_pixels = (pixel_array[:, :, 0] == 255) & (pixel_array[:, :, 1] == 0) & (pixel_array[:, :, 2] == 0)
    x, y, w, h = cv2.boundingRect(red_pixels.astype(numpy.uint8))
    xx, yy, ww, hh = cv2.boundingRect(red_pixels.astype(numpy.uint8))
    center_point = numpy.array([x + w//2, y + h//2])
    step = 2
    found_points = []
    under_vertebrae_points = []
    for ang in range(0, 360, step):
        rad = math.radians(ang)
        flag = False
        for r in range(1, max(pixel_array.shape)):
            x = int(center_point[1] + r * math.cos(rad))
            y = int(center_point[0] + r * math.sin(rad))
            if 0 <= x < pixel_array.shape[0] and 0 <= y < pixel_array.shape[1]:
                if green_pixels[x, y]:
                    flag = True
                if red_pixels[x, y]:
                    found_points.append((y, x))
                    if flag:
                        under_vertebrae_points.append((y, x))
                    break
    try:
        distance = numpy.mean(numpy.linalg.norm(numpy.array(found_points) - center_point, axis=1))
        dispersion = numpy.std(numpy.linalg.norm(numpy.array(found_points) - center_point, axis=1))
        distance = distance - dispersion // 5 * 2
        filtered_points = [point for point in found_points if numpy.linalg.norm(point - center_point) > distance]
        filtered_points = numpy.array(filtered_points)
        filtered_under_vertebrae_points = [point for point in under_vertebrae_points if 206 <= point[0] <= 306]
        filtered_under_vertebrae_points = numpy.array(filtered_under_vertebrae_points)
        if len(filtered_under_vertebrae_points) == 0:
            all_points = filtered_points
        else:
            all_points = numpy.concatenate([filtered_points, filtered_under_vertebrae_points])
        sorted_points = sorted(all_points, key=lambda point: (math.atan2(point[1] - center_point[0], point[0] - center_point[1]) + 2 * math.pi) % (2 * math.pi))

        shift = (distance + dispersion)

        try:
            left_point = min([point for point in all_points if point[1] >= 256 and point[0] <= 100], key=lambda point: point[0])
        except:
            left_point = (int(center_point[0] - shift), int(center_point[1]))
        try:
            right_point = max([point for point in all_points if point[1] >= 256 and point[0] >= 412], key=lambda point: point[0])
        except:
            right_point = (int(center_point[0] + shift), int(center_point[1]))
        bottom_points = [point for point in all_points if 200 <= point[0] <= 306]
        if len(bottom_points) != 0:
            bottom_point = max(bottom_points, key=lambda point: point[1])
        else:
            bottom_point = (center_point[0], center_point[1] + 64)
        left_point = (left_point[0] + 4, left_point[1])
        right_point = (right_point[0] - 4, right_point[1])
        
        center_left_top_point = left_point
        center_right_top_point = right_point
        
        bottom_point = (bottom_point[0], bottom_point[1] - 64)
        vertebrae_left_bottom_points = [point for point in numpy.argwhere(green_pixels) if point[1] < center_point[0] - 16 and point[0] > center_point[1] + 32]
        vertebrae_right_bottom_points = [point for point in numpy.argwhere(green_pixels) if point[1] > center_point[0] + 16 and point[0] > center_point[1] + 32]

        if len(vertebrae_left_bottom_points) != 0:
            vertebrae_left_bottom_point = max(vertebrae_left_bottom_points, key=lambda point: -point[1])
            vertebrae_left_bottom_point = (vertebrae_left_bottom_point[1], vertebrae_left_bottom_point[0])
        else:
            vertebrae_left_bottom_point = (center_point[1], center_point[0] - 64)
        if len(vertebrae_right_bottom_points) != 0:
            vertebrae_right_bottom_point = max(vertebrae_right_bottom_points, key=lambda point: point[1])
            vertebrae_right_bottom_point = (vertebrae_right_bottom_point[1], vertebrae_right_bottom_point[0])
        else:
            vertebrae_right_bottom_point = (center_point[1], center_point[0] + 64)
        
        ratio = red_pixels.sum() / (512 * 512)
        if ratio >= 0.1:
            shift = distance + dispersion // 5 * 2
            if shift < 100:
                shift += dispersion // 3 + 100
        else:
            shift = distance - dispersion
        if shift > 120:
            shift *= 0.725
        fixed_ratio = [0.9, 0.9, 0.22, 0.225]
            
        left_bottom_point = (int(center_point[0] - shift), int(center_point[1] + hh * 0.35))
        right_bottom_point = (int(center_point[0] + shift), int(center_point[1] + hh * 0.35))
        left_point = (int(xx + 24), center_point[0] - 16)
        left_point_alpha = (int(xx + 16), center_point[0] + 8)
        right_point = (int(xx + ww - 24), center_point[0] - 16)
        right_point_alpha = (int(xx + ww - 16), center_point[0] + 8)
        
        center_left_bottom_point = left_point
        center_right_bottom_point = right_point
        
        left_bottom_x_l = left_bottom_point[0] - ww // 14
        left_bottom_x_r = left_bottom_point[0] + ww // 14
        right_bottom_x_l = right_bottom_point[0] - ww // 14
        right_bottom_x_r = right_bottom_point[0] + ww // 14
        
        left_bottom_y_l = int(left_bottom_point[1] * 0.95)
        left_bottom_y_r = int(left_bottom_point[1] * 0.98)
        right_bottom_y_l = int(right_bottom_point[1] * 0.98)
        right_bottom_y_r = int(right_bottom_point[1] * 0.95)
        
        left_bottom_point_l = (left_bottom_x_l, left_bottom_y_l)
        left_bottom_point_r = (left_bottom_x_r, left_bottom_y_r)
        right_bottom_point_l = (right_bottom_x_l, right_bottom_y_l)
        right_bottom_point_r = (right_bottom_x_r, right_bottom_y_r)
        
        points = [left_point, left_point_alpha, left_bottom_point_l, left_bottom_point, left_bottom_point_r, vertebrae_left_bottom_point, vertebrae_right_bottom_point, right_bottom_point_l, right_bottom_point, right_bottom_point_r, right_point_alpha, right_point]
        
        outline_points = []
        for i in range(len(points) - 1):
            outline_points.append(points[i])
            pos1 = points[i]
            pos2 = points[i + 1]
            distance_pos = numpy.linalg.norm(numpy.array(pos1) - numpy.array(pos2))
            if distance_pos <= 64 or abs(pos1[1] - pos2[1]) <= 64:
                cv2.line(pixel_array, pos1, pos2, (0, 0, 0), 2)
            elif abs(pos1[0] - pos2[0]) >= 128:
                for t in numpy.linspace(0, 1, 100):
                    x = int((1 - t) * pos1[0] + t * pos2[0])
                    y = int((1 - t) * pos1[1] + t * pos2[1] + 20 * (math.sin(t * math.pi) * 2) )
                
                    outline_points.append((x, y))
            else:
                for t in numpy.linspace(0, 1, 100):
                    x = int((1 - t) * pos1[0] + t * pos2[0])
                    y = int((1 - t) * pos1[1] + t * pos2[1] + 20 * math.sin(t * math.pi))
                    outline_points.append((x, y))
        
        outline_points.append(right_point)
        outline_points = numpy.array(outline_points)

        cv2.fillPoly(pixel_array, [numpy.array(sorted_points)], (0, 0, 0))
        cv2.fillPoly(pixel_array, [numpy.array(outline_points)], (0, 0, 0))
        
        red_contours, _ = cv2.findContours(red_pixels.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if red_contours:
            all_red_points = numpy.concatenate(red_contours)
            hull = ConvexHull(all_red_points[:, 0, :])
            hull_area = all_red_points[hull.vertices]
        # outline = contours[0]
        outline = hull_area
        outline_center_point = numpy.mean(outline, axis=0, dtype=numpy.float32)

        outline[:, 0, 0] = (outline[:, 0, 0] - outline_center_point[0][0]) * fixed_ratio[0] + outline_center_point[0][0]
        outline[:, 0, 1] = (outline[:, 0, 1] - outline_center_point[0][1]) * fixed_ratio[1] + outline_center_point[0][1]
        outline = outline.astype(numpy.int32)
        outline = numpy.array([point for point in outline if point[0][1] <= center_point[0]])
        if outline is not None and len(outline) > 0:
            outline = outline.astype(numpy.int32).reshape((-1, 1, 2))
            cv2.drawContours(pixel_array, [outline], -1, (0, 0, 0), -1)    
        center_points = numpy.array([center_left_top_point, center_left_bottom_point, center_right_bottom_point, center_right_top_point])
        cv2.fillPoly(pixel_array, [center_points], (0, 0, 0))
    except:
        pass
    return pixel_array

def extract_muscle_only(pixel_array: numpy.ndarray, remove_noise: bool = True):
    """
    근육 영역만 추출하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        remove_noise (bool): 잡음 제거 여부

    Returns:
        numpy.ndarray: 근육 영역만 추출된 이미
    """
    red_pixels = (pixel_array[:, :, 0] == 255) & (pixel_array[:, :, 1] == 0) & (pixel_array[:, :, 2] == 0)
    black_pixels = (red_pixels[:, :] == 0)
    mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    reversed_mask = numpy.zeros(pixel_array.shape[:2], dtype=numpy.uint8)
    mask[red_pixels] = 255
    reversed_mask[black_pixels] = 255
    result = numpy.zeros_like(pixel_array)
    result[mask == 255] = [255, 0, 0]
    
    if remove_noise:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] <= 128:
                result[labels == i] = [0, 0, 0]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(reversed_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= 128:
            result[labels == i] = [255, 0, 0]
            
    # closing_size = 16
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            
    return result

def preprocess(idx: int, category: str, sn: str, dicom_path: str, visualize_dir: str, boundary: list):
    """
    경로 및 CT 이미지 전처리 함수

    Args:
        idx (int): 이미지 인덱스
        category (str): 이미지 카테고리
        sn (str): 시리즈 번호
        dicom_path (str): DICOM 파일 경로
        visualize_dir (str): 시각화 디렉토리

    Returns:
        str: 시각화된 이미지 경로
        numpy.ndarray: CT 이미지 배열
        numpy.ndarray: 정규화된 CT 이미지 배열
    """
    output_dir = os.path.join(visualize_dir, category)
    os.makedirs(output_dir, exist_ok=True)
    cur_output_dir = os.path.join(output_dir, sn)
    os.makedirs(cur_output_dir, exist_ok=True)
    cur_output_path = os.path.join(cur_output_dir, f"{idx:03d}.png")

    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array
    pixel_array = remove_bottom_arc(pixel_array, boundary)
    pixel_array[pixel_array <= 96] = 0
        
    normalized_pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    normalized_pixel_array = normalized_pixel_array.astype(numpy.uint8)
    normalized_pixel_array = cv2.cvtColor(normalized_pixel_array, cv2.COLOR_GRAY2BGR)

    return cur_output_path, pixel_array, normalized_pixel_array

def show(pixel_array):
    """
    이미지를 시각화하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열

    Returns:
        None
    """
    plt.imshow(pixel_array, cmap='gray')
    plt.show()

def save(pixel_array, path):
    """
    이미지를 저장하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        path (str): 이미지 저장 경로
    
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    ax.imshow(pixel_array)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
def save_temp(pixel_array, idx = 0):
    """
    이미지를 저장하는 함수

    Args:
        pixel_array (numpy.ndarray): 이미지 배열
        path (str): 이미지 저장 경로
    
    Returns:
        None
    """
    filename = f"temp_{idx}.png"
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    ax.imshow(pixel_array)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return idx + 1

def extract_muscle_from_over_lung(
        method: str,
        norm_idx: float, 
        pixel_array: numpy.ndarray, 
        normalized_pixel_array: numpy.ndarray
    ):
    """
    CT 이미지 중 폐 상부 영역에서 타겟 영역(Rib Muscle)을 추출하는 함수

    Args:
        norm_idx(float): 이미지 정규화 인덱스
        pixel_array (numpy.ndarray): 원본 이미지 배열
        normalized_pixel_array (numpy.ndarray): 정규화된 이미지 배열

    Returns:
        numpy.ndarray: 마스킹된 이미지 배열
    """
    contours = find_contours(normalized_pixel_array)
    normalized_pixel_array = fill_contours(normalized_pixel_array, contours)
    pixel_array = remove_skin(pixel_array, contours, method)
    muscle_mask = find_muscle(pixel_array, False)
    normalized_pixel_array = fill_muscle(normalized_pixel_array, muscle_mask)
    bone_mask, bone_contours = find_bone(pixel_array, muscle_mask)
    
    normalized_pixel_array = masking_bone(normalized_pixel_array, bone_mask, bone_contours)
    normalized_pixel_array = extract_muscle_only(normalized_pixel_array)
    normalized_pixel_array = masking_inner_area_from_over_lung(normalized_pixel_array, contours, bone_contours)
    return normalized_pixel_array

def extract_muscle_from_lung(
        method: str,
        norm_idx: float, 
        pixel_array: numpy.ndarray, 
        normalized_pixel_array: numpy.ndarray
    ):
    """
    CT 이미지 중 폐 영역에서 타겟 영역(Rib Muscle)을 추출하는 함수

    Args:
        norm_idx(float): 이미지 정규화 인덱스
        pixel_array (numpy.ndarray): 원본 이미지 배열
        normalized_pixel_array (numpy.ndarray): 정규화된 이미지 배열

    Returns:
        numpy.ndarray: 마스킹된 이미지 배열
    """
    if 0.35 <= norm_idx <= 0.8:
        heart_mask = find_heart(pixel_array)
        pixel_array, normalized_pixel_array = masking_heart(pixel_array, normalized_pixel_array, heart_mask)
    contours = find_contours(normalized_pixel_array)
    normalized_pixel_array = fill_contours(normalized_pixel_array, contours)
    pixel_array = remove_skin(pixel_array, contours, method)
    muscle_mask = find_muscle(pixel_array, True)
    normalized_pixel_array = fill_muscle(normalized_pixel_array, muscle_mask)
    bone_mask, bone_contours = find_bone(pixel_array, muscle_mask)
    bone_inner_points = find_bone_inner_contours(bone_contours)
    vertebrae_contour = find_vertebrae_contour(bone_contours)

    normalized_pixel_array = masking_bone(normalized_pixel_array, bone_mask, bone_contours)
    normalized_pixel_array = masking_bone_inner_area(normalized_pixel_array, bone_inner_points)
    normalized_pixel_array = masking_vertebrae(normalized_pixel_array, vertebrae_contour)
    normalized_pixel_array = masking_breast(normalized_pixel_array, bone_contours)
    normalized_pixel_array = masking_inner_area_from_lung(normalized_pixel_array, contours, bone_contours, vertebrae_contour) # <- 1.62초
    normalized_pixel_array = extract_muscle_only(normalized_pixel_array)
    return normalized_pixel_array

def extract_muscle_from_under_lung(
        method: str,
        norm_idx: float, 
        pixel_array: numpy.ndarray, 
        normalized_pixel_array: numpy.ndarray
    ):
    """
    CT 이미지 중 폐 하부 영역에서 타겟 영역(Rib Muscle)을 추출하는 함수

        norm_idx(float): 이미지 정규화 인덱스
    Args:
        pixel_array (numpy.ndarray): 원본 이미지 배열
        normalized_pixel_array (numpy.ndarray): 정규화된 이미지 배열

    Returns:
        numpy.ndarray: 마스킹된 이미지 배열
        dict: 복근 정보
    """
    contours = find_contours(normalized_pixel_array)    
    normalized_pixel_array = fill_contours(normalized_pixel_array, contours)
    pixel_array = remove_skin(pixel_array, contours, method)
    muscle_mask = find_muscle(pixel_array, False)
    normalized_pixel_array = fill_muscle(normalized_pixel_array, muscle_mask)
    bone_mask, bone_contours = find_bone(pixel_array, muscle_mask)
    bone_inner_points = find_bone_inner_contours(bone_contours)
    vertebrae_contour = find_vertebrae_contour(bone_contours)
    
    normalized_pixel_array = masking_bone(normalized_pixel_array, bone_mask, bone_contours)
    normalized_pixel_array = masking_bone_inner_area(normalized_pixel_array, bone_inner_points)
    normalized_pixel_array = masking_vertebrae(normalized_pixel_array, vertebrae_contour)
    normalized_pixel_array = masking_inner_area_from_under_lung(normalized_pixel_array, contours, bone_contours)
    abs_info = find_abs(norm_idx, normalized_pixel_array, bone_contours, vertebrae_contour)
    normalized_pixel_array = extract_muscle_only(normalized_pixel_array)
    return normalized_pixel_array, abs_info

def extract(idx: int, norm_idx: float, category: str, sn: str, method: str, boundary: list, dicom_path: str, visualize_dir: str):
    """
    CT 이미지에서 타겟 영역(Rib Muscle)을 추출하는 함수

    Args:
        idx(int): 이미지 인덱스
        norm_idx (float): 정규화된 이미지 인덱스 (0.0~1.0)
        category (str): 이미지 카테고리
        sn (str): 시리즈 번호
        method (str): 추출 방법
        boundary (list): 경계 정보
        dicom_path (str): DICOM 파일 경로
        visualize_dir (str): 시각화 디렉토리

    Returns:
        numpy.ndarray: 원본 이미지 배열
        numpy.ndarray: 마스킹된 이미지 배열
        dict: 메타데이터
    """
    try:
        # 입력 유효성 검사
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
        
        # norm_idx 유효성 검사 및 보정
        norm_idx = max(0.0, min(1.0, float(norm_idx)))
        
        cur_output_path, pixel_array, normalized_pixel_array = preprocess(idx, category, sn, dicom_path, visualize_dir, boundary)
        original = normalized_pixel_array.copy()
        
        # 안전한 메타데이터 초기화
        metadata = {
            "idx": idx,
            "norm_idx": float(norm_idx),
            "position": method,
            "sternum": False,
            "abs": {
                "base_point": (256, 256),  # 안전한 기본값
                "left_point": (-1, -1),
                "right_point": (-1, -1),
                "left_y_intercept": 0,
                "right_y_intercept": 0,
            }
        }
        
        metadata["sternum"] = True if method in ["upper", "lung"] else False
        
        try:
            if method == "upper":
                result = extract_muscle_from_over_lung(method, norm_idx, pixel_array, normalized_pixel_array)
            elif method == "lung":
                result = extract_muscle_from_lung(method, norm_idx, pixel_array, normalized_pixel_array)
            elif method == "lower":
                result, abs_info = extract_muscle_from_under_lung(method, norm_idx, pixel_array, normalized_pixel_array)
                
                # abs_info 안전성 검사
                if abs_info and isinstance(abs_info, dict):
                    metadata["abs"]["base_point"] = abs_info.get("center", (256, 256))
                    metadata["abs"]["left_point"] = abs_info.get("left", (-1, -1))
                    metadata["abs"]["right_point"] = abs_info.get("right", (-1, -1))
                    metadata["abs"]["left_y_intercept"] = abs_info.get("left_y_intercept", 0)
                    metadata["abs"]["right_y_intercept"] = abs_info.get("right_y_intercept", 0)
            else:
                # 알 수 없는 method의 경우 기본 처리
                log("warning", f"Unknown method '{method}', using default processing")
                result = normalized_pixel_array.copy()
                
        except Exception as e:
            log("warning", f"Error in muscle extraction for method '{method}': {e}")
            result = normalized_pixel_array.copy()  # 원본을 결과로 사용
        
        # 결과 저장
        try:
            save(result, cur_output_path)
        except Exception as e:
            log("warning", f"Failed to save result to {cur_output_path}: {e}")
        
        return original, result, metadata
        
    except Exception as e:
        log("error", f"Critical error in extract function: {e}")
        # 안전한 폴백 반환
        import numpy as np
        empty_image = np.zeros((512, 512, 3), dtype=np.uint8)
        safe_metadata = {
            "idx": idx,
            "norm_idx": float(norm_idx) if norm_idx is not None else 0.0,
            "position": method,
            "sternum": False,
            "error": str(e),
            "abs": {
                "base_point": (256, 256),
                "left_point": (-1, -1),
                "right_point": (-1, -1),
                "left_y_intercept": 0,
                "right_y_intercept": 0,
            }
        }
        return empty_image, empty_image, safe_metadata