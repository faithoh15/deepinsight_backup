# 검출 → 추적 → 제거 통합 파이프라인

카메라·LiDAR 녹화 하나를 넣으면 명령어 하나로 세 결과를 저장합니다.

1. **검출**: 3D 사람 박스 (v3 선택 누적 검출)
2. **추적**: 박스마다 3D ID와 속도
3. **제거**: 추적된 사람을 모두 지운 최종 정적 맵 (자세 보정 좌표)

마지막 단계(`result`)는 이 결과를 합쳐, 박스마다 3D ID·속도·저속/고속 분류·맵 좌표를 붙인 최종 표 `4_result/boxes.csv`를 만듭니다.

## 폴더 구성

| 위치 | 내용 |
|---|---|
| `run.py` | 실행기: 녹화마다 13단계를 차례로 돌립니다. |
| `view.py` | 결과 3D 보기 |
| `configs/` | `pipeline.yaml`(모든 단계 설정), `local.example.yaml`(이 컴퓨터 경로의 예), `datasets.csv`(녹화 목록) |
| `envs/` | conda 환경 파일 두 개 |
| `stages/` | 단계마다 하나씩 있는 실행 진입점 |
| `pipeline/` | 계산 코드 (준비, 검출, 추적, 맵, 최종 표) |
| `tools/` | 가중치 받기, 설치 확인, 결과 비교, 보기 도구 |
| `tests/` | 테스트 (녹화 데이터 없이 돌아감) |
| `output/`, `models/*.pt`, `configs/local.yaml` | 실행하면 생기는 결과·가중치·이 컴퓨터 설정. git에 올라가지 않습니다. |

## 필요한 것

- **운영체제**: Linux (Ubuntu에서 확인)
- **conda**: Miniconda, Anaconda, Miniforge 중 하나
- **GPU**: NVIDIA GPU와 드라이버 520 이상 (CUDA 11.8용 torch를 씁니다). YOLO 단계에 쓰고, 개발 장비는 RTX 3090입니다.
- **디스크**: 결과가 녹화 하나에 약 1~3 GB입니다. 원본을 로컬로 복사하면 녹화 하나에 1~2 GB가 더 듭니다.
- **녹화 데이터**: DIMENVUE D2 장비로 녹화한 폴더 (아래 "녹화 데이터 준비")

## 설치

```bash
# 1. 코드 받기
git clone git@github.com:faithoh15/deepinsight_backup.git
cd deepinsight_backup

# 2. 실행 환경 두 개 만들기 (처음 한 번, 수 분~수십 분)
conda env create -f envs/environment-main.yml     # 환경 이름 deepinsight
conda env create -f envs/environment-yolo.yml     # 환경 이름 yolo_env (torch + CUDA 포함, 약 3 GB)

# 3. YOLO 가중치 받기 (Ultralytics 공식 파일, sha256 확인)
conda activate deepinsight
python tools/download_models.py                   # -> models/yolo26l.pt, models/yolo26l-seg.pt

# 4. 이 컴퓨터의 경로 설정
cp configs/local.example.yaml configs/local.yaml
#    configs/local.yaml을 열어 nas_root(녹화 폴더 위치)를 고칩니다. 나머지는 보통 그대로 둡니다.

# 5. 돌릴 녹화 목록: configs/datasets.csv (아래 "녹화 데이터 준비")

# 6. 설치 확인: [오류]가 없으면 준비 완료
python tools/check_setup.py
```

- **저장소 접근**: 비공개 저장소라서, 저장소 주인이 GitHub의 Settings → Collaborators에서 받는 사람의 계정을 추가해야 clone할 수 있습니다. SSH 키를 GitHub에 등록하지 않았다면 `https://github.com/faithoh15/deepinsight_backup.git`으로 받습니다.
- **conda 환경 이름**: 이름을 바꿨거나 conda가 기본 위치(`~/miniconda3`, `~/anaconda3`, `~/miniforge3`, `~/mambaforge`)에 없으면, `configs/local.yaml`의 `envs`에 python 경로를 직접 적습니다. 예: `main: /opt/conda/envs/deepinsight/bin/python`
- **설정 파일 두 개**
  - `configs/pipeline.yaml`: 모든 단계의 설정값. 고치지 않습니다.
  - `configs/local.yaml`: 이 컴퓨터의 경로. git에 올라가지 않습니다.
- **가중치를 받을 수 없을 때**: 스크립트가 알려 주는 `curl` 명령으로 직접 받아 `models/`에 넣습니다.

## 녹화 데이터 준비

`nas_root` 아래에 녹화마다 날짜 이름의 폴더가 있어야 합니다.

```
<nas_root>/
  2026-10-12-203411/
    calib/config/camera.yaml        카메라 내부값, cam0 -> cam1
    calib/config/imu_camera0.yaml   cam0 -> IMU
    imu_data.csv                    IMU (Stamp, GyroX, ...)
    lidar/<시각 us>.ply             LiDAR 스캔
    images_cam0/<시각 us>.jpg       어안 카메라 두 대
    images_cam1/<시각 us>.jpg
```

- **`configs/datasets.csv`**: `name,source` 두 열입니다.
  - `name`은 결과 폴더 이름이고, 반드시 `<아무 이름>_<날짜 폴더 이름>` 꼴이어야 합니다. 예: `학연산밤_고속_2026-10-12-203411`
  - `source`는 날짜 폴더 이름입니다. 예: `2026-10-12-203411`
  - 지금 목록은 개발에 쓴 녹화 17개입니다. 자기 녹화로 바꾸거나 줄을 더하면 됩니다.
- **원본 위치**
  - 녹화가 로컬 디스크에 있으면 `nas_root`와 `cache_root`를 같은 폴더로 둡니다. 이때는 복사하지 않습니다.
  - NAS처럼 느린 곳에 있으면 `cache_root`를 로컬 폴더로 두면, 첫 단계가 한 번만 복사합니다.
  - 원본 파일은 읽기만 하고 고치거나 지우지 않습니다.

## 실행

```bash
conda activate deepinsight
python run.py 학연산밤_고속_2026-10-12-203411     # 녹화 하나
python run.py e10복도_정지_2026-10-12-210841 e10복도_저속_2026-10-12-211133   # 여러 개
python run.py --all                               # configs/datasets.csv 전부
python run.py --list                              # 녹화별로 몇 단계까지 끝났는지
```

- **실행기와 환경**: `run.py`는 deepinsight 환경에서 실행합니다. YOLO 단계는 실행기가 알아서 yolo_env로 돌립니다.
- **걸리는 시간**: 녹화 하나(2~4분 분량)에 5~28분입니다(RTX 3090, 검출 작업자 8개). 복도처럼 길고 단조로운 곳은 자세 보정이 오래 걸립니다.
- **결과 위치**: `output/<이름>/`에 저장합니다.
- **이름 오타**: 비슷한 이름을 알려 주고, 아무것도 실행하지 않습니다.
- **다시 돌릴 때**
  - 이미 끝난 단계는 건너뜁니다.
  - 특정 단계부터 다시 하려면 `--from 단계 [--to 단계] --force`를 씁니다.
  - 앞 단계를 다시 돌리면, 그 뒤 단계는 자동으로 다시 돌아야 하는 상태가 됩니다.
- **실패할 때**: 한 녹화가 실패해도 다음 녹화로 넘어갑니다. 끝에 `실패: <이름> (<단계>)`를 출력하고 종료 코드 1로 끝납니다. 원인은 `output/<이름>/logs/<단계>.log`에 남습니다.
- **앞 단계가 안 끝났을 때**: `--from`으로 중간부터 돌릴 때 앞 단계가 끝나 있지 않으면, 그 단계를 실행하지 않고 "earlier stage(s) not done"으로 실패 처리합니다.
- **경고**: 검출 중 일부 프레임에서 오류가 나면 단계는 끝까지 돌지만, 그 단계 줄과 맨 끝에 `경고: <이름> (detection) N frames failed ...`를 출력합니다. 해당 프레임의 사람은 빠져 있으니 로그를 확인하세요.
- **사람이 한 명도 안 잡힌 녹화**: 사람 박스가 있는 프레임에서만 바닥 평면을 구합니다. 그래서 박스가 하나도 없는 녹화는 맵 단계가 "no floor plane"으로 실패합니다. 검출·추적 결과(빈 표)는 정상으로 저장됩니다.

## 실행 환경

| 환경 | 만드는 파일 | 쓰는 단계 |
|---|---|---|
| deepinsight | `envs/environment-main.yml` | 실행기, 복사, 시각 보정, 오도메트리, 마스크, 영상 펼치기, 검출, 추적, 자세 보정, 맵, 최종 표, 보기 도구 |
| yolo_env | `envs/environment-yolo.yml` | YOLO, 운영자 표시, 2D 추적 |

- 오도메트리는 반드시 deepinsight에서 돌립니다. yolo_env에서 돌리면 결과가 달라집니다(1초 이동량이 최대 20 cm 차이).
- 패키지 버전은 공개된 결과를 만든 환경과 같게 고정했습니다.

## 단계

| # | 단계 | 하는 일 | 결과 |
|---|---|---|---|
| 1 | fetch | 원본 녹화(`nas_root`)를 `cache_root`로 복사합니다. 빠졌거나 크기가 다른 파일만 하나씩 복사하고, 아무것도 지우지 않습니다. 두 폴더가 같으면 복사하지 않습니다. | `<cache_root>/<날짜>/` |
| 2 | clock | 영상 시각 보정값 = −(첫 cam0 영상 시각 − 첫 IMU 시각), ms 반올림 | `0_prep/clock.yaml` |
| 3 | odometry | 자이로 회전 + LiDAR 위치(ICP)로 장비 궤적을 구합니다. | `0_prep/odometry/` |
| 4 | masks | 장비·운영자 LiDAR 마스크와 카메라 프레임별 장비 자세를 만듭니다. 예전 1 GB 창 파일은 만들지 않습니다. | `0_prep/masks/` |
| 5 | unwrap | 어안 영상 두 장을 펼쳐 좌우로 붙입니다. | `0_prep/sidebyside/` |
| 6 | yolo | YOLO 사람 박스 | `0_prep/yolo/` |
| 7 | operator | 운영자(장비 든 사람) 픽셀 마스크로 운영자 박스를 표시합니다. | `0_prep/operator_mask/`, `0_prep/yolo/` |
| 8 | track2d | 2D 박스를 polar 방식으로 추적합니다. | `0_prep/track2d/` |
| 9 | **detection** | v3 선택 누적 3D 사람 박스 | `1_detection/` |
| 10 | **tracking** | 3D 추적: 박스마다 ID와 속도 | `2_tracking/` |
| 11 | pose | 맵 전용 자세 보정(루프 폐합) | `3_map/pose_refine/` |
| 12 | **map** | 추적된 사람을 모두 지운 최종 맵 | `3_map/` |
| 13 | **result** | 트랙·프레임·박스의 속도 분류(저속/고속/초고속)와 박스별 최종 표. 앞 단계 결과는 읽기만 합니다. | `4_result/` |

단계 설정값은 모두 `configs/pipeline.yaml` 하나에 있고, 공개된 결과를 만든 값 그대로입니다. 이 컴퓨터의 경로만 `configs/local.yaml`에 따로 씁니다. 앞단 단계의 디버그 그림은 `debug_figures: true`일 때만 만듭니다.

## 결과 폴더 `output/<이름>/`

### 최종적으로 쓰면 되는 것 (★)

| 파일 | 한 줄 단위 | 내용 |
|---|---|---|
| ★ `4_result/boxes.csv` | 박스 1개 | 박스 크기·위치·heading, 3D ID, 속도, 이동 방향, 저속/고속 분류, 맵 좌표 |
| ★ `4_result/tracks.csv` | 사람(3D ID) 1명 | 시작·끝 시각, 속도 중앙값·최고값, 분류 `speed_class` |
| ★ `4_result/track_frames.csv` | 사람 × 프레임 | 궤적 위치, 속도, 이동 방향, 분류(트랙·프레임 기준) |
| ★ `3_map/map_static.ply` | 5 cm 복셀 | 추적된 사람을 모두 지운 최종 정적 맵 (자세 보정 좌표) |

나머지 파일은 확인·진단용이거나 다른 용도의 원자료입니다(아래 "단계별 전체 결과물").

### 좌표계

| 좌표계 | 열·파일 | 언제 쓰나 |
|---|---|---|
| 장비 기준 (수평 맞춤, z = 바닥 위 높이) | `cx`, `cy`, `cz`, `yaw_deg`, `1_detection/clouds/*.ply` | 그 순간 센서에서 본 위치 |
| 오도메트리 (녹화 시작 위치가 원점) | `*_odom`, 트랙 `x`, `y`, `vx`, `vy`, `speed`, `heading_deg` | 한 프레임이나 몇 초 안의 계산 (박스 위치·크기, 속도, heading, 분류) |
| 맵 (자세 보정) | `x_map`, `y_map`, `z_map`, `yaw_map_deg`, `heading_map_deg`, `3_map/*.ply` | 최종 맵 위에 박스 겹치기, 녹화 전체에서 위치 비교 |

- 오도메트리 좌표는 짧은 시간 안에서는 정확하지만 시간이 지날수록 조금씩 어긋납니다. 예를 들어 학연산밤_정지에서 60~120초 떨어진 두 구간은 보정 전 최대 2 m, 보정 후 3.5 cm 어긋납니다.
- 그래서 멀리 떨어진 시간끼리 위치를 비교하거나 맵에 올릴 때만 맵 좌표를 씁니다.
- 속도와 heading은 추적 좌표 값 그대로입니다. 17개 녹화의 확정 트랙 7,063프레임에서 자세 보정을 적용해도 분류가 바뀐 프레임은 0개였습니다(2026-11-17).

### `4_result/boxes.csv`의 열

v3 박스 한 줄마다 아래 값이 들어 있습니다.

- **박스**: 크기 `l`, `w`, `h`(`l`이 heading 방향 길이), 중심 `cx_odom`, `cy_odom`, `cz_odom`, heading `yaw_odom_deg`, `valid`
  - `yaw_source`가 `motion`이면 실제 이동 방향이라 앞뒤까지 맞습니다.
  - `pca`면 몸 모양으로 구한 방향이라 앞뒤(180°)를 모릅니다.
- **추적**: 3D `track_id`(2D ID는 `track_id_2d`), 그 프레임 속도 `speed`(95 % 범위 `speed_lo95`~`speed_hi95`), 이동 방향 `heading_deg`, `track_status`, `segment`
- **분류**: `speed_class`(트랙 대표 속도 = 속도 중앙값 기준, ID마다 하나), `speed_class_frame`(그 프레임 속도 기준)
- **맵 좌표**: `x_map`, `y_map`, `z_map`, `yaw_map_deg`, `heading_map_deg` (`map_static.ply`와 같은 자세 보정 좌표)
- **장비 기준**: `cx`, `cy`, `cz`(바닥 기준 높이), `yaw_deg`

**분류 기준**(`configs/pipeline.yaml`의 `result.speed_class`)
- 저속 < 2.78 m/s(10 km/h) ≤ 고속 ≤ 5.56 m/s(20 km/h) < 초고속
- 속도가 없으면 `미정`입니다.

**권장 사용**: `valid == 1`, `track_confirmed == 1`(한 번이라도 확정된 트랙)
- 확정 전 첫 프레임들은 `track_status`가 `tentative`이지만 속도는 있습니다. 그래서 `track_status == confirmed`로 거르면 이 프레임들이 빠집니다.
- 속도가 없는 박스는 분류가 `미정`입니다. 대부분 관측이 한 번뿐이라 확정되지 않은 트랙입니다(17개 녹화 235개).
- 박스는 모두 관측 프레임이라 `segment`는 항상 `obs`입니다(17개 녹화 8,207개 확인).
- 사람을 놓친 사이 다른 사람에게 잘못 이어 붙은 트랙은 그 구간 속도가 튑니다(예: 개신문화관_저속 T43, 5.9 m/s). 사람별 속도는 `speed_med`나 `speed_class`가 안전합니다.
- 분류 값이 한글이라 Excel에서 바로 열면 글자가 깨질 수 있습니다. 그럴 때는 '데이터 가져오기'에서 UTF-8을 선택하세요.

```python
import pandas as pd
b = pd.read_csv("output/학연산밤_고속_2026-10-12-203411/4_result/boxes.csv")
b = b[(b.valid == 1) & (b.track_confirmed == 1)]        # 쓸 만한 박스만
b.groupby("speed_class").track_id.nunique()             # 분류별 사람 수
```

### 단계별 전체 결과물

**`0_prep/`: 준비 단계 (대부분 내부용)**

| 파일 | 내용 |
|---|---|
| `sidebyside/*.jpg` | 프레임별로 펼친 어안 영상 두 장을 붙인 것 (원본 그림) |
| `yolo/detections.csv` | YOLO 2D 사람 박스 |
| `track2d/tracks.csv` | 2D 추적 결과 (2D ID) |
| `odometry/odometry.csv` | 장비 궤적 (오도메트리 좌표) |
| `masks/frame_poses.csv` | 카메라 프레임마다 장비 자세 (4×4 `T_odom_dev`) |
| `clock.yaml` | 영상 시각 보정값 |

**`1_detection/`: v3 3D 박스**

| 파일 | 내용 |
|---|---|
| `persons.csv` | 박스 원본 표. 열이 많고 보조 박스(`b_*`, `all_*`)까지 있으며, 필요한 열은 `4_result/boxes.csv`에 옮겨 두었습니다. 이 표의 `track_id`는 2D ID입니다. |
| `clouds/NNNN_<key>.ply` | 박스마다 사람 점군 (장비 기준, 스캔 번호·영상 기준 시각 포함) |
| `vis/boxes_img/`, `vis/boxes_bev/` | 프레임별 박스 그림 (영상, 위에서 본 그림), 확인용 |
| `vis/prep/`, `vis/scans/`, `vis/accum/` | 바닥 제거 → 스캔별 조각 → 누적 과정 그림, 디버그용 |
| `frames.csv`, `scans.csv`, `scan_candidates.csv`, `ambiguous.csv`, `summary.json` | 프레임 상태와 후보 처리 기록, 진단용 |

**`2_tracking/`: 3D 추적**

| 파일 | 내용 |
|---|---|
| `tracks.csv`, `track_frames.csv` | 트랙 요약과 프레임별 상태. 분류 열이 붙은 `4_result` 버전을 쓰세요. |
| `rows.csv` | 박스(`frame`, `key`)와 3D ID의 대응 (`4_result/boxes.csv`에 반영됨) |
| `vis_det/img/`, `vis_det/bev/` | 박스에 `T<ID> 속도 m/s`를 표시한 그림, 확인용 |
| `observations.csv`, `events.csv`, `summary.json` | 추적기 입력 관측과 생성·확정·삭제 기록, 진단용 |

**`3_map/`: 최종 맵 (자세 보정 좌표)**

| 파일 | 내용 |
|---|---|
| ★ `map_static.ply` | 최종 맵. 필드는 `x`, `y`, `z`, `h`(바닥 위 높이), `n`(점 수) |
| `map_before.ply` | 사람을 지우기 전 맵 (비교용, 맵 범위 = v3가 볼 수 있던 영역) |
| `map_removed.ply` | 지운 사람 점. `track_id`로 어느 사람인지 알 수 있습니다. |
| `map_protected.ply` | 사람 원기둥 안이지만 벽으로 보고 남긴 점 (`track_id` 포함) |
| `map_residual.ply` | 잔여 진단만 한 점(지우지 않음). `cause` 1 = 사람 근처, 2 = 사람 없음 |
| `removed_by_track.csv`, `top_*.png`, `summary.json` | 사람별 제거 수, 위에서 본 그림, 요약 |
| `pose_refine/pose_correction.csv` | 오도메트리 → 맵 좌표 변환값. 다른 값을 맵에 올릴 때 씁니다. |
| `pose_refine/trajectory_top.png`, `consistency.csv` | 보정 전후 궤적과 정합 확인 |

**`4_result/`: 최종 표**

| 파일 | 내용 |
|---|---|
| ★ `boxes.csv` | 위 "`4_result/boxes.csv`의 열" 참고 |
| ★ `tracks.csv` | `2_tracking/tracks.csv`에 `speed_class` 열을 더한 표 |
| ★ `track_frames.csv` | `2_tracking/track_frames.csv`에 `heading_deg`, `speed_class`, `speed_class_frame` 열을 더한 표 |
| `summary.json` | 분류별 트랙·프레임·박스 수 |

**그 밖**

- `logs/`: 단계별 로그와 완료 표시(`.done`)
- `run.json`: 단계별 상태·시간·명령

## 보기

화면이 있는 컴퓨터에서 실행합니다 (Open3D 창).

```bash
python view.py 학연산밤_고속_2026-10-12-203411 map             # 1 지우기 전 / 2 최종 / 3 겹침 / 4 잔여, C 천장, F 바닥, S 화면 저장
python view.py 학연산밤_고속_2026-10-12-203411 det --track     # 검출 박스와 사람 점, 3D ID 색 (N/P 프레임 이동, C 색 모드)
```

두 실행 결과 폴더를 비교하려면 `python tools/compare_results.py <폴더1> <폴더2>`를 씁니다(처리 시간 열은 무시).

## 테스트

```bash
conda activate deepinsight
for f in tests/test_*.py; do python $f || echo "실패: $f"; done
```

- 실행기, 복사, 설정 읽기, 설치 확인 도구, 이식성(코드에 이 컴퓨터만의 절대 경로가 없는지)을 확인합니다. 녹화 데이터는 필요 없습니다.
- yolo_env 쪽 모듈은 `python tools/check_setup.py`가 실제로 불러와 확인합니다.

## 주의할 점

- **장비 전용 캘리브레이션 보정값**
  - 카메라·IMU 회전 보정값이 코드 상수로 들어 있습니다(`pipeline/common/cam_geom.py`의 `IMU_RPY_DEG`, `CAM1_RPY_DEG`).
  - 개발에 쓴 DIMENVUE D2 한 대에 맞춘 값입니다. 녹화의 yaml 파일은 고치지 않습니다.
  - 다른 장비의 녹화라면 이 값을 그 장비에 맞게 새로 구해야 합니다.
- **영상 시각 보정**: 영상 시각이 LiDAR/IMU보다 약 9.6~9.8초 늦게 찍혀 있습니다. `clock` 단계가 녹화마다 −(첫 cam0 영상 시각 − 첫 IMU 시각)으로 자동 계산합니다(`0_prep/clock.yaml`).
- **자세 보정 재현성**: Open3D 병렬 계산 때문에 실행할 때마다 자세 보정값이 10⁻⁶ 수준으로 다를 수 있습니다. 나머지 단계는 같은 입력이면 같은 결과를 냅니다.
- **복도 녹화**: 길고 단조로운 복도에서는 ICP가 복도 방향으로 미끄러질 수 있습니다. 그래서 자세 보정의 정합 확인(`3_map/pose_refine/consistency.csv`)을 그대로 믿기 어렵습니다.
- **라이선스**: YOLO(Ultralytics 패키지와 가중치)는 AGPL-3.0입니다. 이 저장소를 공개하거나 서비스에 쓰려면 그 조건을 따라야 합니다.
