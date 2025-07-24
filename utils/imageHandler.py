import os
import io
import cv2
import numpy
from PIL import Image
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from utils.logger import log_execution_time

@log_execution_time
def convert_image(filtered_convex_info: list, filtered_masking_info: list):
    """
    필터링된 Convex와 Masking 정보를 이미지로 변환하는 함수

    Args:
        filtered_convex_info (list): 필터링된 Convex 정보 리스트
        filtered_masking_info (list): 필터링된 Masking 정보 리스트

    Returns:
        list: 변환된 이미지 리스트
    """

    def convert_png_bytes_to_numpy(png_bytes):
        """
        PNG 바이트를 NumPy 배열로 변환하는 함수

        Args:
            png_bytes (bytes): PNG 바이트

        Returns:
            numpy.ndarray: NumPy 배열
        """
        image = Image.open(io.BytesIO(png_bytes))
        return numpy.array(image)
    
    def convert_matplotlib_figure_to_numpy(fig):
        """
        Figure를 NumPy 배열로 변환하는 함수

        Args:
            fig (matplotlib.figure.Figure): Figure 객체

        Returns:
            numpy.ndarray: NumPy 배열
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        return numpy.array(image)
    
    converted_convex_images = [(info[0], convert_png_bytes_to_numpy(info[1])) for info in filtered_convex_info]
    converted_masking_images = [(info[0], convert_matplotlib_figure_to_numpy(info[1])) for info in filtered_masking_info]
    combined_images_with_index = converted_convex_images + converted_masking_images
    sorted_combined_images_with_index = sorted(combined_images_with_index, key=lambda x: x[0])

    return sorted_combined_images_with_index

def expand_image_to_12bit(image_array):
    """
    이미지를 12비트로 확장하는 함수

    Args:
        image_array (numpy.ndarray): 이미지 배열

    Returns:
        numpy.ndarray: 확장된 이미지 배열
    """
    if len(image_array.shape) > 2:
        image_array = image_array[:,:,0]
    return (image_array.astype(numpy.float32) * (4095/255)).astype(numpy.uint16)
    
def numpy_to_hu(image_array, min_pixel_value=0, max_pixel_value=4095, min_hu_value=-1000, max_hu_value=1000):
    """
    이미지를 HU 단위로 변환하는 함수

    Args:
        image_array (numpy.ndarray): 이미지 배열
        min_pixel_value (int): 최소 픽셀 값
        max_pixel_value (int): 최대 픽셀 값
        min_hu_value (int): 최소 HU 값
        max_hu_value (int): 최대 HU 값

    Returns:
        numpy.ndarray: HU 이미지 배열
    """
    actual_min = image_array.min()
    actual_max = image_array.max()
    normalized_array = (image_array - actual_min) / (actual_max - actual_min) * (max_pixel_value - min_pixel_value) + min_pixel_value
    hu_array = (normalized_array - min_pixel_value) * (max_hu_value - min_hu_value) / (max_pixel_value - min_pixel_value) + min_hu_value
    return hu_array

def extract_points_in_hu_range(hu_image, lower_bound, upper_bound):
    """
    HU 범위 내의 점을 추출하는 함수

    Args:
        hu_image (numpy.ndarray): HU 이미지
        lower_bound (int): 하한값
        upper_bound (int): 상한값

    Returns:
        numpy.ndarray: HU 범위 내의 점
    """
    mask = (hu_image >= lower_bound) & (hu_image <= upper_bound)
    points = numpy.column_stack(numpy.nonzero(mask))
    return points

def cluster_points(points, eps=5, min_samples=50):
    """
    점을 클러스터링하는 함수

    Args:
        points (numpy.ndarray): 점
        eps (int): 클러스터 범위
        min_samples (int): 최소 샘플

    Returns:
        numpy.ndarray: 클러스터링된 레이블
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels

def find_cluster_y_min(points, cluster_labels):
    """
    클러스터의 최소 y값을 찾는 함수

    Args:
        points (numpy.ndarray): 점
        cluster_labels (numpy.ndarray): 클러스터 레이블

    Returns:
        int: 최소 y값
    """
    unique_labels = numpy.unique(cluster_labels)
    cluster_y_min = []
    for label in unique_labels:
        if label != -1:
            cluster_points = points[cluster_labels == label]
            y_min = numpy.min(cluster_points[:, 0])
            cluster_y_min.append(y_min)
    return min(cluster_y_min) if cluster_y_min else None

def check_clusters_intersect_with_rectangle(hu_image, processed_hu_image, width=70, height=100):
    """
    클러스터가 사각형과 교차하는지 확인하는 함수

    Args:
        hu_image (numpy.ndarray): HU 이미지
        processed_hu_image (numpy.ndarray): 처리된 HU 이미지
        width (int): 너비
        height (int): 높이

    Returns:
        bool: 교차 여부
    """
    def is_any_point_within_rectangle(points, x_mid, y_min, width=70, height=100):
        """
        점이 사각형 내에 있는지 확인하는 함수

        Args:
            points (numpy.ndarray): 점
            x_mid (int): 중심 x값
            y_min (int): 최소 y값
            width (int): 너비
            height (int): 높이

        Returns:
            bool: 사각형 내에 있는지 여부
        """
        x_min = x_mid - width // 2
        x_max = x_mid + width // 2
        y_max = y_min + height
        within_rectangle = numpy.logical_and(
            numpy.logical_and(points[:, 1] >= x_min, points[:, 1] <= x_max),
            numpy.logical_and(points[:, 0] >= y_min, points[:, 0] <= y_max)
        )
        return numpy.any(within_rectangle)
    
    points = extract_points_in_hu_range(hu_image, lower_bound=300, upper_bound=1900)
    labels = cluster_points(points)
    x_mid, y_min = find_x_mid_and_y_min(processed_hu_image)
    is_inside = any(
        is_any_point_within_rectangle(points[labels == label], x_mid, y_min, width, height)
        for label in numpy.unique(labels) if label != -1
    )
    return is_inside

def create_and_fill_contours(hu_image, points, labels):
    """
    컨투어를 생성하고 채우는 함수

    Args:
        hu_image (numpy.ndarray): HU 이미지
        points (numpy.ndarray): 점
        labels (numpy.ndarray): 레이블

    Returns:
        numpy.ndarray: 채워진 이미지
    """
    filled_image = cv2.cvtColor(hu_image, cv2.COLOR_GRAY2BGR)
    mask = numpy.zeros(hu_image.shape, dtype=numpy.uint8)
    unique_labels = numpy.unique(labels)

    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label]
        cluster_mask = numpy.zeros(hu_image.shape, dtype=numpy.uint8)
        cluster_mask[cluster_points[:, 0], cluster_points[:, 1]] = 255
        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled_image, contours, -1, (0, 255, 255), thickness=cv2.FILLED)
        cv2.drawContours(filled_image, contours, -1, (0, 255, 255), thickness=2)

    return filled_image

def process_image(image):
    """
    이미지를 HU 이미지로 변환하고 처리하는 함수

    Args:
        image (numpy.ndarray): 이미지

    Returns:
        tuple: 변환된 이미지(hu_image), 처리된 이미지(processed_hu_image)
    """
    expanded_image = expand_image_to_12bit(image)
    hu_image = numpy_to_hu(expanded_image).astype(numpy.float32)
    points = extract_points_in_hu_range(hu_image, lower_bound=300, upper_bound=1900)
    labels = cluster_points(points)
    processed_hu_image = create_and_fill_contours(hu_image, points, labels)
    return hu_image, processed_hu_image

def find_x_mid_and_y_min(processed_hu_image):
    """
    이미지의 중심점과 최소 y값을 찾는 함수

    Args:
        processed_hu_image (numpy.ndarray): 처리된 HU 이미지

    Returns:
        tuple: 중심 x값(x_mid), 최소 y값(y_min)
    """
    non_zero_coords = numpy.column_stack(numpy.nonzero(processed_hu_image != -1000))
    x_mid = numpy.median(non_zero_coords[:, 1]).astype(int)
    y_min = numpy.min(non_zero_coords[:, 0]).astype(int)
    return x_mid, y_min

def store_is_inside_and_y_min(index: int, is_inside:dict, y_min:dict, is_inside_value:bool, y_min_value:int):
    """
    is_inside와 y_min을 저장하는 함수

    Args:
        index (int): 인덱스
        is_inside (dict): is_inside
        y_min (dict): y_min
        is_inside_value (bool): is_inside 값
        y_min_value (int): y_min 값

    Returns:
        tuple: is_inside, y_min
    """
    is_inside[index] = is_inside_value
    y_min[index] = y_min_value
    return is_inside, y_min

def image_process_pipeline(hu_image, points, labels, is_inside, index, y_min_dict):
    """
    이미지 처리 파이프라인 함수

    Args:
        hu_image (numpy.ndarray): HU 이미지
        points (numpy.ndarray): 점
        labels (numpy.ndarray): 레이블
        is_inside (dict): is_inside
        index (int): 인덱스
        y_min (dict): y_min

    Returns:
        tuple: HU 이미지(hu_image), 처리된 HU 이미지(processed_hu_image), 중심 x값(x_mid), 최소 y값(y_min)
    """

    def integrated_image_coloring(hu_image, points, labels, is_inside, index, y_min_dict):
        """
        통합 이미지 색칠 함수

        Args:
            hu_image (numpy.ndarray): HU 이미지
            points (numpy.ndarray): 점
            labels (numpy.ndarray): 레이블
            is_inside (dict): is_inside
            index (int): 인덱스
            y_min_global (dict): y_min

        Returns:
            numpy.ndarray: 색칠된 이미지
        """
        hu_image_normalized = cv2.normalize(hu_image, None, 0, 255, cv2.NORM_MINMAX).astype(numpy.uint8)
        colored_image = cv2.cvtColor(hu_image_normalized, cv2.COLOR_GRAY2RGB)
        cluster_mask = numpy.zeros(hu_image.shape, dtype=numpy.uint8)
        unique_labels = numpy.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points[labels == label]
            cluster_mask[cluster_points[:, 0], cluster_points[:, 1]] = 255
        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored_image, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
        y_min = y_min_dict[index]
        if is_inside:
            for y in range(hu_image.shape[0]):
                for x in range(hu_image.shape[1]):
                    if -90 <= hu_image[y, x] <= 150:
                        if y >= y_min:
                            colored_image[y, x] = [0, 255, 0]
                        else:
                            colored_image[y, x] = [255, 0, 0]
        else:
            mask = (hu_image >= -90) & (hu_image <= 150)
            colored_image[mask] = [0, 255, 0]
        return colored_image
    
    def shift_hull_toward_center(hull_points, image_shape, shift_amount=5):
        """
        중심으로 볼록체를 이동하는 함수

        Args:
            hull_points (numpy.ndarray): 볼록체 점
            image_shape (tuple): 이미지 모양
            shift_amount (int): 이동량

        Returns:
            numpy.ndarray: 이동된 볼록체 점
        """
        center_y, center_x = numpy.array(image_shape[:2]) // 2
        shifted_hull_points = []
        for point in hull_points:
            y, x = point
            direction = numpy.array([center_y - y, center_x - x])
            direction_norm = direction / numpy.linalg.norm(direction)
            shifted_point = numpy.array([y, x]) + shift_amount * direction_norm
            shifted_hull_points.append(shifted_point)
        shifted_hull_points = numpy.array(shifted_hull_points).astype(int)  
        return shifted_hull_points
    
    def remove_skin_muscle_colored(hu_image, colored_image, shift_amount=10):
        """
        피부 및 근육을 제거하는 함수

        Args:
            hu_image (numpy.ndarray): HU 이미지
            colored_image (numpy.ndarray): 색칠된 이미지
            shift_amount (int): 이동량

        Returns:
            numpy.ndarray: 피부 및 근육이 제거된 이미지
        """
        subcutaneous_mask = (hu_image >= -800) & (hu_image <= -700)
        subcutaneous_coords = numpy.column_stack(numpy.nonzero(subcutaneous_mask))
        if len(subcutaneous_coords) > 2:
            hull = ConvexHull(subcutaneous_coords)
            hull_points = subcutaneous_coords[hull.vertices]
            shifted_hull_points = shift_hull_toward_center(hull_points, hu_image.shape, shift_amount)
            hull_mask = numpy.zeros_like(hu_image, dtype=numpy.uint8)
            shifted_hull_points_int = numpy.flip(shifted_hull_points.astype(numpy.int32), axis=1)
            cv2.fillConvexPoly(hull_mask, shifted_hull_points_int, 1)
            masked_colored_image = numpy.zeros_like(colored_image)
            for i in range(3):
                masked_colored_image[:, :, i] = numpy.where(hull_mask == 1, colored_image[:, :, i], 0)
            return masked_colored_image
        else:
            return colored_image
        
    def apply_zero_pixels_for_hu_range(hu_image, colored_image, lower_bound=-1000, upper_bound=-700):
        """
        HU 범위에 대해 픽셀을 0으로 적용하는 함수

        Args:
            hu_image (numpy.ndarray): HU 이미지
            colored_image (numpy.ndarray): 색칠된 이미지
            lower_bound (int): 하한값
            upper_bound (int): 상한값

        Returns:
            numpy.ndarray: 픽셀이 0으로 적용된 이미지
        """
        mask = (hu_image >= lower_bound) & (hu_image <= upper_bound)
        colored_image[mask] = [0, 0, 0]
        return colored_image
    
    colored_image = integrated_image_coloring(hu_image, points, labels, is_inside, index, y_min_dict)
    colored_image = remove_skin_muscle_colored(hu_image, colored_image, shift_amount=10)
    colored_image = apply_zero_pixels_for_hu_range(hu_image, colored_image, lower_bound=-1000, upper_bound=-700)
    return colored_image

def save_overlay_image(image, index, output_dir):
    """
    오버레이 이미지를 저장하는 함수

    Args:
        image (numpy.ndarray): 이미지
        index (int): 인덱스
        output_dir (str): 출력 디렉토리

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"overlay_image_{index}.png"
    full_path = os.path.join(output_dir, filename)
    cv2.imwrite(full_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))