import os
import shutil

target_dir = "input"

patient_dir_list = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]

for patient_id in patient_dir_list:
    cur_patient_dir = os.path.join(target_dir, patient_id)
    date_list = [date for date in os.listdir(cur_patient_dir) if os.path.isdir(os.path.join(cur_patient_dir, date))]
    date_list.sort()

    for date in date_list:
        cur_date_dir = os.path.join(cur_patient_dir, date)
        dicom_file_list = [os.path.join(cur_date_dir, f) for f in os.listdir(cur_date_dir) if f.endswith('.dcm')]
        
        cur_sub_dir = os.path.join(cur_date_dir, "sub")
        os.makedirs(cur_sub_dir, exist_ok=True)
        
        for file in dicom_file_list:
            shutil.move(file, cur_sub_dir)

