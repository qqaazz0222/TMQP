import os
import cv2
import pydicom
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from utils.directoryHandler import check_checkpoint
from utils.logger import log_execution_time, log_progress

def split_by_low_regions(arr, threshold = 128):
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
            segments.append((start, arr[start:end]))
        else:
            i += 1
    segments = [(start, segment) for start, segment in segments if len(segment) > 6]
    return segments

def calc_boundary(dicom_list: list):
    boundary = []
    if len(dicom_list) > 0:
        distance_list = []
        for file in dicom_list:
            dcm = pydicom.dcmread(file)
            pixel_array = dcm.pixel_array
            h, w = pixel_array.shape

            target_x = w // 2
            target_col = pixel_array[:, target_x]
            segmented_col = split_by_low_regions(target_col)
            segmented_col = sorted(segmented_col, reverse=True)
            distance = h
            
            for segment in segmented_col:
                start_idx, segment_arr = segment
                if len(segment_arr) < 48:
                    distance = start_idx
                else:
                    break
            
            distance_list.append(distance)
        distance = np.mean(distance_list)
        distance = int(distance) - 8
        
        mask = np.zeros_like(pixel_array, dtype=np.uint8)
        shift = 148
        center = (target_x, -shift)
        radius = distance + shift
        cv2.circle(mask, center, radius, 255, -1)
    
        for x in range(w):
            flag = False 
            for y in range(h):
                if mask[y, x] != 255:
                    boundary.append(y - 8)
                    flag = True
                    break
            if not flag:
                boundary.append(h - 8)
    
    return boundary

def remove_bottom_structure(pixel_array, boundary):
    h, w = pixel_array.shape
    for x in range(w):
        for y in range(h):
            if pixel_array[y, x] > 0:
                if y > boundary[x]:
                    pixel_array[y, x] = 0
    return pixel_array
            
def extract_muscle(pixel_array, processed_image):
    r = [1000, 1200]
    mask = (pixel_array >= r[0]) & (pixel_array <= r[1])
    processed_image[mask] = [255, 0, 0]  # Red color
    return processed_image
    
def extract_bone(pixel_array, processed_image):
    r = [1200, 2000]
    mask = (pixel_array >= r[0]) & (pixel_array <= r[1])
    processed_image[mask] = [0, 255, 0]
    return processed_image

def preprocess_image(dicom_path: str):
    if os.path.exists(dicom_path):
        boundary = calc_boundary([dicom_path])
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array
        pixel_array = remove_bottom_structure(pixel_array, boundary)
        pixel_array[pixel_array <= 96] = 0
        
        processed_image = np.zeros((*pixel_array.shape, 3), dtype=np.uint8)
        processed_image = extract_muscle(pixel_array, processed_image)
        processed_image = extract_bone(pixel_array, processed_image)
        processed_image = Image.fromarray(processed_image)
        processed_image = processed_image.resize((256, 256))
        processed_image = processed_image.convert("L")
        return processed_image
    else:
        return None
    
def init_model(model_path: str, device: str):
    
    def load_model(model_path, device):
        model = models.resnet18(weights=None)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.fc.in_features, 3)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    model = load_model(model_path, device)
    
    return model, transform

@log_execution_time
def predict(model, transform, device, checkpoint_dir:str, working_list: list, category: str, sn: str):
    label_map = {0: 'upper', 1: 'lung', 2: 'lower'}
    
    checkpoint = f"checkpoint_predict_{category}_{sn}.pkl"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    if check_checkpoint(checkpoint_path):
        return pickle.load(open(checkpoint_path, 'rb'))
    
    result = []
    
    for idx, file_path in enumerate(working_list):
        log_progress(working_list.index(file_path) + 1, len(working_list), f"Predicting({file_path})")
        
        image = preprocess_image(file_path)
        image = transform(image).unsqueeze(0).to(device)  # (1, 1, 256, 256)
        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, dim=1).item()
            pred_class = label_map[pred]
        result.append(pred_class)
    pickle.dump(result, open(checkpoint_path, 'wb'))
    return result
       
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = "resnet_ct_classifier_best.pth"
#     dicom_path = "test.dcm"
    
#     # Initialize model
#     model, transform = init_model(model_path, device)
    
#     # Predict single DICOM file
#     predicted_label = predict(model, transform, device, dicom_path=dicom_path)
#     print(f"Predicted label for {dicom_path}: {predicted_label}")