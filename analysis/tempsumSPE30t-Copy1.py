import sys
import os
import re
import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import gamma
import uproot
import pandas as pd

# 별도 모듈에서 변수 및 함수 불러오기 (채널 정보, 에러 계산 함수 등)
from wbls_vars import *
from wbls_functions import *

# Polya(감마) 로그 피팅 함수 정의
def polya_log(x, N, k, theta):
    out = np.full_like(x, -1e10, dtype=np.float64)
    mask = (x > 0)
    out[mask] = (np.log(N)
                 + (k - 1) * np.log(x[mask])
                 - (x[mask] / theta)
                 - np.log(gamma(k))
                 - k * np.log(theta))
    return out

# 전역 변수: 전압 스케일링, 히스토그램 저장 딕셔너리 등
voltageFactor = 2000 / (2**14 - 1)
speHistograms = {}         # 후처리된 (used) 히스토그램 (파란색)
speHistogramsUnused = {}   # 원본 전체 ROI 데이터 (unused, 노란색)

# 스킵할 채널 목록과 사용 채널 선택
skip = ['adc_b1_ch9', 'adc_b4_ch13']
channels = [channel for channel in dataChannelNamesSorted30t if channel not in skip]

# 명령행 인자로부터 날짜 코드를 받아 날짜 및 디렉토리 설정
date_code = sys.argv[1]  # 예: 'YYMMDDTAABB'
date = date_code[:6]      # 'YYMMDD'
dateString = datetime.strptime(date, '%y%m%d').strftime('%d %b %Y')

print('putting together root files for: {}'.format(dateString))

# 진단 플롯 생성: 각 채널에 대해 used (파란색)와 unused (노란색) 히스토그램 그리고 피팅 결과 (빨간 선)
nChannels = len(channels)
plt.figure(figsize=[20, nChannels])
results = {}

p0_dict = {channel: [max(speHistograms[channel]) if max(speHistograms[channel]) > 0 else 50, 1.0, 0.5] for channel in channels}

i = 0
for channel in channels:
    i += 1
    ax = plt.subplot(math.ceil(nChannels / 4), 4, i)
    hist_values = speHistograms[channel]
    bin_edges = np.linspace(-0.25, 6.5, 201)

    # used: 파란색, unused: 노란색
    plt.stairs(hist_values, bin_edges, fill=True, color='blue', label='used')
    plt.stairs(speHistogramsUnused[channel], bin_edges, color='orange', label='unused')
    plt.xlabel('SPE Area [pC]')
    plt.xlim(-0.5, 6.5)
    plt.yscale('log')
    plt.legend(title=channel, title_fontproperties={'weight': 'bold'})

    try:
        # mode (최대 빈도수 bin) 근처 ±0.5 구간에서 피팅 수행
        modeBin = np.argmax(hist_values)
        modeCharge = bin_edges[modeBin]
        xdata = (bin_edges[1:] + bin_edges[:-1]) / 2
        xmask = (xdata > (modeCharge - 0.5)) & (xdata < (modeCharge + 0.5))
        xdata_fit = xdata[xmask]
        ydata_fit = hist_values[xmask]
        valid = ydata_fit > 0
        xdata_fit = xdata_fit[valid]
        ydata_log = np.log(ydata_fit[valid])
        p0 = p0_dict[channel]
        popt, pcov = curve_fit(polya_log, xdata_fit, ydata_log, p0=p0)
        dof = len(xdata_fit) - 3

        # 감마 분포의 모드 (x좌표)
        k, theta = popt[1], popt[2]
        spe_mean = (k - 1) * theta if k > 1 else 0  # 감마 분포의 mode (x좌표)

        # χ² 계산을 위한 mode 주변의 x값 선택 (mode ±0.5 범위)
        chi_mask = (xdata > (spe_mean - 0.5)) & (xdata < (spe_mean + 0.5))
        x_chi = xdata[chi_mask]
        y_chi = hist_values[chi_mask]

        # 피팅된 감마 분포 값 계산
        y_fit_chi = np.exp(polya_log(x_chi, *popt))

        # χ² 계산 (mode 근처의 데이터만 사용)
        chi2 = np.sum((y_chi - y_fit_chi) ** 2 / (y_fit_chi + 1e-10))  # 0 나누기 방지
        dof_chi = len(x_chi) - 3  # 자유도

        results[channel] = {
            'ch_id': int(channel[5:6] + ('0' if len(channel[9:]) == 1 else '') + channel[9:]),
            'ch_name': channel,
            'pmt': channelPMTNames[channel],
            'spe_mean': spe_mean,
            'spe_width': theta,
            'chi2': chi2,
            'dof': dof_chi,
            'spe_mean_err': None,
            'spe_width_err': None,
            'HV': channelHVValues[channel],
            'fit_method': 'Polya (log fit)'
        }

        a_vals = np.linspace(modeCharge - 0.5, modeCharge + 0.5, 350)
        fitted_curve = np.exp(polya_log(a_vals, *popt))
        plt.plot(a_vals, fitted_curve, color='red')

        # Mode 위치 vline 추가 (녹색 실선)
        plt.vlines(spe_mean, 0, max(fitted_curve), color='green', linestyle='solid', label=f'SPE mode ~ {spe_mean:.3f}')
        
        # χ² 값도 플롯 제목에 추가
        plt.title(f'{channel} (χ² = {chi2:.2f}, dof={dof_chi})')

    except Exception as e:
        print(f'fit didn\'t work for {channel} -> {e}')
        continue

plt.tight_layout()
plt.savefig('diagnostics/30t/temp/' + date + 'diagnosticplot_gammafit.png')

print('done with {}'.format(dateString))
exit(1)
