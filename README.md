# 두 층 DMI 모형: 다중 거리 척도 구조 — 이론 코드와 시뮬레이션 의뢰 묶음

이 묶음은 채팅에서 진행한 해석 연구의 코드, 데이터, 그리고 Claude Code에 넘길 시뮬레이션 의뢰서를 모은 것이다.

| 파일/폴더 | 내용 |
| --- | --- |
| `REQUEST_ClaudeCode.md` | **시뮬레이션 의뢰서** (Claude Code에 이 파일부터 읽히면 된다) |
| `THEORY_SUMMARY.md` | 검증 대상이 되는 해석 결과 요약 |
| `docs/` | Claude Doc 전체를 Markdown으로 내보내 `theory_notes.md`로 넣을 자리 |
| `sim/params.py` | Landau 변수 → 실공간 모델 변수 사상, 실공간 커널 (띠 최소점 검증 포함) |
| `sim/gl_bilayer_reference.py` | 기준 GL 이완 코드 (작은 계에서 검증됨, 운영 계산용 아님) |
| `landau/` | Landau 이론 계산 모듈과 분석 스크립트 |
| `data/` | 채팅에서 계산한 상도·경계·여유 에너지 데이터 |

## 빠른 확인
```bash
cd sim && python params.py                 # 실공간 커널의 띠 최소점 = 이론값 확인
python gl_bilayer_reference.py             # 64x64 이완: 에너지 단조 감소 확인
cd ../landau && python predict_point.py 0.6 2.318 150 4   # 한 점의 Landau 예측 (약 1분)
```
의존성: numpy, scipy (Python 3.10+).

## `landau/` 모듈 지도 (문서 절 번호)
모든 모듈은 같은 폴더에서 이름으로 서로 import한다. `landau/` 안에서 실행할 것.

| 파일 | 역할 | 절 |
| --- | --- | --- |
| `landau.py` | Fourier 정확 평균 ⟨δm_z δm²⟩, ⟨(δm²)²⟩ (모든 계산의 핵심) | 8 |
| `scan.py`, `run2.py` | 후보 상태 기하·다항 계수, 고정 공명비 격자 스캔 | 9 |
| `exact.py`, `fastE.py` | 닫힌 꼴 에너지, 극좌표 최소화 | 9 |
| `trace.py`, `curves.py` | 상경계 근 추적과 곡선 연결 | 9 |
| `tongue.py`, `tscan.py`, `tedge.py` | 파수 어긋남(띠 함수 f(ξ)), 잠금 혀 경계 | 10 |
| `kc.py`, `kcfast.py` | 섬 여유 에너지와 K 의존성 (이전 해석, 17절에서 정정) | 11 |
| `analytic.py` | 보편 함수 φ(τ), 해석적 m∞, g_min, 껍질 띠 에너지 Δf_sh | 12 |
| `rings.py`, `cands.py` | 일반 다중 고리 상태, 확장 후보(6회 준결정, 여러 껍질 육각, 근사결정, 8회) | 13 |
| `newmargin.py`, `ctop.py` | 확장 후보에 대한 여유 에너지, 혀 꼭대기 c_top(K) | 13, 17 |
| `phason.py`, `hess.py`, `elastic.py` | 12모드 정확 에너지, 24×24 헤세 행렬, 포논·위상자 탄성 | 15 |
| `nucleate.py`, `eckhaus.py` | 새 모드 핵생성, 유한 파장(Eckhaus) 안정성 | 16 |
| `predict_point.py` | 임의의 (c/c_L, ρ, K, g)에서 후보 에너지 순위 | — |
| `qc12s_x.npy`, `qc12sym_x.npy` | c/c_L = 0.7, g = 4의 6회/12회 정류점 진폭 (`hess.py`가 읽음) | 15 |

## 회귀 확인값
- `analytic.m_inf(4.0)` → −0.3666
- 6회 준결정 에너지 (c/c_L = 0.7, g = 4, K = ∞) → −2.3540
- ρ*_QC(0.3) = 1.9993, ρ*_QC(0.6) = 2.3184, ρ*_√3(0.3) = 1.7812

## `data/`
`phase.json`(고정 공명비 격자 상도), `bd_*.json`/`curves.json`(정확 상경계), `rows_*.json`(정밀 격자), `te_*.json`/`tg_*.json`/`tdata.json`(잠금 혀), `kc_pts.json`/`marg_g4.json`(K 의존 여유 에너지). 스크립트는 현재 폴더에 결과를 쓰므로, 다시 만들 때는 `landau/`에서 실행한 뒤 옮기면 된다.
