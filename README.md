# TMQP

폐 질환 환자의 CT 기반 흉부 근육 정량화를 위한 완전 자동화된 딥러닝 및 규칙 기반 하이브리드 파이프라인
Fully Automated CT-Based Thoracic Muscle Quantification in Patients with Lung Disease Using a Hybrid Deep Learning and Rule-Based Pipeline

### Note

모든 테스트는 다음 환경에서 진행되었습니다. 일부 환경에서는 버전 호환성 확인이 필요할 수 있습니다.

    CPU: Intel(R) Core(TM) i9-13900KF
    GPU: Nvidia GeForce RTX 4090, CUDA 12.1
    OS: Ubuntu 24.04 LTS
    Conda: 25.5.1

업데이트 내용은 아래 문서를 확인하세요.
[🗒️ 업데이트 내역](UPDATE.md)

## Installation

이 저장소에서 제공하는 모듈을 실행하기 위해 Conda 기반 환경을 구성합니다.

만약, Conda가 설치되어 있지 않다면 아래 링크에 접속하여 설치 후 단계를 진행합니다.

[🔗 아나콘다 다운로드](https://www.anaconda.com/download/success) 또는 [🔗 미니콘다 다운로드](https://www.anaconda.com/docs/getting-started/miniconda/main)

**Step 1**. 저장소 복제

```bash
git clone https://github.com/qqaazz0222/TMQP
cd TMQP
```

**Step 2**. Conda 가상환경 생성 및 활성화

```bash
conda create --name tmqp python=3.12 -y
conda activate tmqp
```

**Step 3**. 라이브러리 설치

```bash
pip install -r requirements.txt
```
