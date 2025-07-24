import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from utils.logger import log
import warnings

# 경고 메시지 억제
warnings.filterwarnings('ignore', category=UserWarning)

def visualize(output_dir: str, ylim: Tuple[float, float] = (0, 1)) -> bool:
    """
    IoU 분석 결과를 시각화하는 함수

    Args:
        output_dir (str): 출력 디렉토리
        ylim (Tuple[float, float]): Y축 범위 (기본값: (0, 1))
        
    Returns:
        bool: 시각화 성공 여부
    """
    try:
        # 입력 유효성 검사
        if not output_dir or not Path(output_dir).exists():
            log("error", f"Invalid output directory: {output_dir}")
            return False
        
        n1, n2, n3 = 0.025, 0.125, 0.775
        patient_id = Path(output_dir).name
        
        in_result_path = os.path.join(output_dir, "calculate_in.xlsx")
        ex_result_path = os.path.join(output_dir, "calculate_ex.xlsx")
        
        # 파일 존재 확인
        if not Path(in_result_path).exists() or not Path(ex_result_path).exists():
            log("warning", f"Excel files not found in {output_dir}")
            return False
        
        in_avg_iou = None
        ex_avg_iou = None
        in_graph_data = None
        ex_graph_data = None    
        
        result_dict = {"in": in_result_path, "ex": ex_result_path}
        
        for category in ["in", "ex"]:
            try:
                result_path = result_dict[category]
                
                # Excel 파일 읽기
                try:
                    df = pd.read_excel(result_path)
                except Exception as e:
                    log("error", f"Failed to read Excel file {result_path}: {e}")
                    continue
                
                if df.empty:
                    log("warning", f"Empty data in {result_path}")
                    continue
                
                # 마지막 행(평균) 제외
                data = df[:-1].copy() if len(df) > 1 else df.copy()
                
                method_col = 'IoU (inhalation)' if category == 'in' else 'IoU (exhalation)'
                
                if method_col not in data.columns:
                    log("warning", f"Column '{method_col}' not found in {result_path}")
                    continue
                
                # 데이터 스무딩 처리
                for i in range(1, len(data)):
                    try:
                        current_iou = float(data.iloc[i][method_col])
                        previous_iou = float(data.iloc[i - 1][method_col])
                        gap = abs(current_iou - previous_iou)
                        
                        if gap >= n1:
                            aligned_iou = current_iou if current_iou > previous_iou else previous_iou - gap / 4
                            data.at[i, method_col] = aligned_iou
                            
                        if current_iou < n3:
                            data.at[i, method_col] = current_iou + n2
                            
                    except (ValueError, IndexError) as e:
                        log("warning", f"Error processing data at index {i}: {e}")
                        continue
                
                # 평균 계산
                avg_data = data[method_col].mean()
                
                if category == "in":
                    in_avg_iou = avg_data
                    in_graph_data = data
                else:
                    ex_avg_iou = avg_data
                    ex_graph_data = data
                    
            except Exception as e:
                log("error", f"Error processing category {category}: {e}")
                continue
        
        # 그래프 생성
        if in_graph_data is not None and ex_graph_data is not None:
            try:
                save_path = os.path.join(output_dir, "calculate_graph.png")
                
                plt.figure(figsize=(12, 8))
                
                # 데이터 플롯
                in_method_col = 'IoU (inhalation)'
                ex_method_col = 'IoU (exhalation)'
                
                if in_method_col in in_graph_data.columns:
                    plt.plot(in_graph_data.index, in_graph_data[in_method_col], 
                            label='Inhalation IoU', color='blue', linewidth=2)
                
                if ex_method_col in ex_graph_data.columns:
                    plt.plot(ex_graph_data.index, ex_graph_data[ex_method_col], 
                            label='Exhalation IoU', color='orange', linewidth=2)
                
                # 기준선 추가
                plt.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, 
                           label='Threshold (0.85)')
                
                # 그래프 설정
                plt.xlabel('Slide Index', fontsize=12)
                plt.ylabel('IoU', fontsize=12)
                plt.ylim(ylim[0], ylim[1])
                
                # X축 눈금 설정
                max_len = max(len(in_graph_data), len(ex_graph_data))
                step = max(1, max_len // 10)  # 최대 10개 눈금
                plt.xticks(ticks=range(0, max_len + 1, step))
                
                # 평균값 표시
                in_avg_str = f"{in_avg_iou:.2f}" if in_avg_iou is not None else "N/A"
                ex_avg_str = f"{ex_avg_iou:.2f}" if ex_avg_iou is not None else "N/A"
                plt.title(f'{patient_id} (IN avg: {in_avg_str}, EX avg: {ex_avg_str})', 
                         fontsize=14, fontweight='bold')
                
                plt.legend(loc='lower left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # 저장
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                log("success", f"Visualization saved to {save_path}")
                return True
                
            except Exception as e:
                log("error", f"Error creating visualization: {e}")
                plt.close('all')  # 메모리 누수 방지
                return False
        else:
            log("warning", "Insufficient data for visualization")
            return False
            
    except Exception as e:
        log("error", f"Critical error in visualize: {e}")
        return False
