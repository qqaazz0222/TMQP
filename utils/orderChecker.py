import os
import cv2
import numpy
import pydicom

def find_bone_boundary(pixel_array: numpy.ndarray):
    bone_hu_value = [1120, 3000]
    mask = numpy.zeros(pixel_array.shape)
    for i in range(pixel_array.shape[0]):
        for j in range(pixel_array.shape[1]):
            if bone_hu_value[0] <= pixel_array[i, j] <= bone_hu_value[1]:
                mask[i, j] = 1
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(numpy.uint8), connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 256:
            mask[labels == i] = 0
    contours, _ = cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 64]
    
    return mask, contours
    

def check_order(dicom_list: list):
    """
    Dicom 파일의 상단과 하단을 비교하여 시작점을 결정하는 함수
    
    Args:
        # first_dircom (str): 첫 번째 DICOM 파일 경로
        # last_dicom (str): 마지막 DICOM 파일 경로
        
    Returns:
        str: 시작점 ("top" 또는 "bottom")
    """
    try:
        selected_file_list = [dicom_list[0][0], dicom_list[-1][0]]
        num_contours = []
        for file in selected_file_list:
            filename = os.path.basename(file)
            filename = filename.split(".")[0]
            dcm = pydicom.dcmread(file)
            pixel_array = dcm.pixel_array
            _, contours = find_bone_boundary(pixel_array)
            contours = [contour for contour in contours if any(156 <= point[0][0] <= 356 for point in contour)]
            num_contours.append(len(contours))
        if num_contours[0] > num_contours[1]:
            start_point = "top"
        else:
            start_point = "bottom"
        return start_point
    except:
        return "top"