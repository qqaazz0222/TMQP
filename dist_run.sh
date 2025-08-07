#!/bin/bash
num_gpus=$(nvidia-smi -L | wc -l)

# 현재 날짜와 시간을 가져와서 now 변수에 저장
now=$(date +"%Y%m%d_%H%M%S")

# 로그를 저장할 디렉토리를 생성
mkdir -p log
mkdir -p log/$now

# 병렬로 실행할 프로세스 수 설정
num_processes=5
input_dir="data/input"
working_dir="data/working"
output_dir="data/output"

# data/input 디렉토리 아래의 서브디렉토리 목록을 가져와 patient_list 변수에 저장
patient_list=$(find "$input_dir" -mindepth 1 -maxdepth 1 -type d)

# patient_list를 정렬
patient_list=$(echo "$patient_list" | sort)

# 각 환자 디렉토리를 순회
gpu_counter=0
for patient_dir in $patient_list; do
    # 실행 중인 백그라운드 작업 수가 num_processes보다 작아질 때까지 대기
    while [ $(jobs -rp | wc -l) -ge $num_processes ]; do
        sleep 1
    done

    # 디렉토리 이름에서 환자 ID 추출
    patient_id=$(basename "$patient_dir")
    echo "Processing patient $patient_id"

    # GPU 인덱스 계산 (순환)
    gpu_id=$((gpu_counter % num_gpus))
    
    # run.py 스크립트를 실행하고 로그를 파일에 저장 (백그라운드 실행)
    # python run.py --patient_id="$patient_id" --overlap --calculate> "log/${now}/${patient_id}.log" 2>&1 &
    python run.py --patient_id="$patient_id" --input_dir="$input_dir" --working_dir="$working_dir" --output_dir="$output_dir" --device=$gpu_id > "log/${now}/${patient_id}.log" 2>&1 &
    
    # GPU 카운터 증가
    gpu_counter=$((gpu_counter + 1))
done

# 모든 백그라운드 작업이 완료될 때까지 대기
wait
echo "All patients processed."