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

# expDecay 함수 (Fourier 필터링에 사용)
def expDecay(f, f0, d, offset):
    return offset + f0 * np.exp(-d * f)

# ROI (Region of Interest) 선택 함수
def findRoi(eventMaximumTimes, peakMask=None, roiLength=40):
    density = []
    for v in range(int(np.max(eventMaximumTimes[peakMask]))):
        ins = np.sum([v < time < v + roiLength for time in eventMaximumTimes[peakMask]])
        density.append(ins)
    roiStart = int(density.index(max(density)))
    return (roiStart, roiStart + roiLength)

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
groupSumHistograms = {}    # (필요시 사용)

# 스킵할 채널 목록과 사용 채널 선택
skip = ['adc_b1_ch9', 'adc_b4_ch13']
channels = [channel for channel in dataChannelNamesSorted30t if channel not in skip]

# 명령행 인자로부터 날짜 코드를 받아 날짜 및 디렉토리 설정
date_code = sys.argv[1]  # 예: 'YYMMDDTAABB'
date = date_code[:6]      # 'YYMMDD'
dateString = datetime.strptime(date, '%y%m%d').strftime('%d %b %Y')

# 기본 데이터 디렉토리 및 파일 검색
directory = '/media/disk_i/30t-DATA/raw_root/phase1/'
allRootFiles = os.listdir(directory)
inFiles = list(filter(lambda file: date_code in file, allRootFiles))
if not inFiles:
    directory = '/media/disk_c/WbLS-DATA/raw_root/phase3/muon/'
    allRootFiles = os.listdir(directory)
    inFiles = list(filter(lambda file: date_code in file, allRootFiles))
    if not inFiles:
        print('no root files found for date {}'.format(dateString))
        exit(0)

print('putting together {} root files for: {}'.format(len(inFiles), dateString))

# 각 루트 파일을 순차적으로 처리
for root in inFiles:
    # 파일 크기가 10MB 미만이면 건너뜁니다.
    if os.stat(directory + root).st_size < 10000000:
        print("file size small")
        continue

    groupSumHistogramsToAdd = {}
    # 파일명에서 배치 시간을 추출 (예: 210101T1234_1)
    dateTimeBatch = re.search(r'(\d{6})T\d{4}_\d{1,}', root).group(0)
    print('                                      \r', dateTimeBatch, end='\r')

    # 루트 파일 열기 및 이벤트 배열 로드
    try:
        rootFileOpen = uproot.open(directory + root)['daq']
        eventsByChannel = rootFileOpen.arrays(filter_name=channels, library='np')
        alphaEvents = rootFileOpen.arrays(filter_name='adc_b4_ch23', library='np')['adc_b4_ch23']
    except Exception as e:
        print(' ' + dateTimeBatch, ' no good: ', e)
        continue

    nChannels = len(channels)

    # 알파 이벤트 처리: 각 이벤트에 대해 중앙값 보정 후 최대값과 그 위치를 저장
    alphaEventMaxima = np.empty(0)
    alphaEventMaximumTimes = np.empty(0)
    for event in alphaEvents:
        normPosEvent = np.median(event) - event
        alphaEventMaxima = np.append(alphaEventMaxima, np.max(normPosEvent) * voltageFactor)
        alphaEventMaximumTimes = np.append(alphaEventMaximumTimes, np.argmax(normPosEvent) * 2)
    alphaMask = alphaEventMaxima > 50  # 1500 mV 이상의 알파 피크만 선택

    # 각 채널별 이벤트 처리
    for channel in channels:
        print('                                      \r', dateTimeBatch, channel, end='\n' if channel == channels[-1] else '\r')
        try:
            events = eventsByChannel[channel]
        except KeyError:
            print(f"Channel {channel} missing in file {dateTimeBatch}, skipping channel.")
            continue
        except Exception as e:
            print(f"Unexpected error with {channel} in {dateTimeBatch}: {e}")
            continue

        # 중앙값 보정 및 전압 스케일 적용 (밀리볼트 단위)
        events = (np.median(events) - events) * voltageFactor
        dataFraction = 1
        fracMask = [(not bool(event % math.floor(1 / dataFraction))) for event in range(len(events))]
        preProcessingMask = np.logical_and(alphaMask, fracMask)
        events = events[preProcessingMask]

        if len(events) == 0:
            print("No event in mask")
            continue

        numEvents = len(events)
        eventMaxima = np.empty(0)
        eventMaximumTimes = np.empty(0)
        averageSignalAmplitude = np.zeros(int(len(events[0]) / 2))
        addedSignalEvents = 0
        averageNoiseAmplitude = np.zeros(int(len(events[0]) / 2))
        addedNoiseEvents = 0

        try:
            # 각 이벤트에 대해 신호와 노이즈를 분류하여 FFT 진폭 누적
            for event in events:
                if (np.min(event) < -2) and (np.max(event) < 5):  # very pure noise
                    addedNoiseEvents += 1
                    ft_amp = np.abs(np.fft.fft(event))
                    positiveAmplitude = ft_amp[:int(len(ft_amp) / 2)]
                    averageNoiseAmplitude += positiveAmplitude
                if (np.min(event) > -5) and (np.max(event) > 10):  # very pure signal
                    addedSignalEvents += 1
                    ft_amp = np.abs(np.fft.fft(event))
                    positiveAmplitude = ft_amp[:int(len(ft_amp) / 2)]
                    averageSignalAmplitude += positiveAmplitude

            noiseBad = addedNoiseEvents > 0.01 * len(events)
            if noiseBad:
                averageNoiseAmplitude /= addedNoiseEvents
                averageSignalAmplitude /= addedSignalEvents
                freqs = np.fft.fftfreq(len(events[0]), d=2e-9)
                posFreq = freqs[:int(len(freqs) / 2)]
                popt, pcov = curve_fit(expDecay, posFreq[:int(len(posFreq) / 2)], 
                                       averageSignalAmplitude[:int(len(posFreq) / 2)], p0=[200, 4e-8, 10])
                bigPhi = np.power(expDecay(posFreq, *popt), 2) / (np.power(expDecay(posFreq, *popt), 2) + np.power(averageNoiseAmplitude, 2))
                bigPhiMask = np.concatenate([bigPhi[::-1], bigPhi])
                # 각 이벤트에 대해 FFT 필터링 적용
                for n, event in enumerate(events):
                    ft = np.fft.fftshift(np.fft.fft(event))
                    ft *= bigPhiMask
                    ft = np.fft.ifftshift(ft)
                    event_filtered = np.abs(np.fft.ifft(ft))
                    events[n] = event_filtered - np.median(event_filtered)
                    eventMaxima = np.append(eventMaxima, np.max(events[n]))
                    eventMaximumTimes = np.append(eventMaximumTimes, np.argmax(events[n]) * 2)
            else:
                for n, event in enumerate(events):
                    eventMaxima = np.append(eventMaxima, np.max(event))
                    eventMaximumTimes = np.append(eventMaximumTimes, np.argmax(event) * 2)

            # 피크 값 기준 마스크
            peakMask = eventMaxima > 2

            # ROI (Region of Interest) 관련 변수 초기화
            regionOfInterestMaxima = np.empty(0)
            regionOfInterestMinima = np.empty(0)
            regionOfInterestCharges = np.empty(0)

            # 각 이벤트에 대해 고정 ROI 선택 (피크 위치 기준 좌우 10, 20 포인트)
            for event in events:
                max_idx = np.argmax(event)
                left_idx = max(0, max_idx - 10)
                right_idx = min(len(event), max_idx + 20)
                roi = event[left_idx:right_idx]
                regionOfInterestMaxima = np.append(regionOfInterestMaxima, np.max(roi))
                regionOfInterestMinima = np.append(regionOfInterestMinima, np.min(roi))
                regionOfInterestCharges = np.append(regionOfInterestCharges, np.sum(roi) * 2 * (1/50))
        except Exception as e:
            print(' ' + dateTimeBatch, ' no good:', e)
            continue

        # 후처리 마스크: pedestal 조건과 ROI 내 최소값 조건 적용
        pedestalMask = regionOfInterestMaxima > 2.5
        eventMinMask = regionOfInterestMinima > -5
        postProcessingMask = np.logical_and(pedestalMask, eventMinMask)

        # used 히스토그램: 후처리 마스크 적용한 ROI charge 데이터
        histData = regionOfInterestCharges[postProcessingMask]
        h, edges = np.histogram(histData, bins=np.linspace(-0.25, 6.5, 201))
        # unused 히스토그램: 모든 ROI charge 데이터 (마스크 없이)
        unused, _ = np.histogram(regionOfInterestCharges, bins=np.linspace(-0.25, 6.5, 201))

        # 채널별 히스토그램 누적
        if channel in speHistograms:
            speHistograms[channel] += h
        else:
            speHistograms[channel] = h
        if channel in speHistogramsUnused:
            speHistogramsUnused[channel] += unused
        else:
            speHistogramsUnused[channel] = unused

        # (필요시 groupSumHistogramsToAdd도 업데이트 가능)
        
# 진단 플롯 생성: 각 채널에 대해 used (파란색)와 unused (노란색) 히스토그램 그리고 피팅 결과 (빨간 선)
nChannels = len(channels)
plt.figure(figsize=[20, nChannels])
results = {}
# 채널별 초기 p0: used 히스토그램의 최대값 기반
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
    
        # χ² 계산을 위한 mode 주변의 x값 선택 (mode ±1.5 범위)
        chi_mask = (xdata > (spe_mean - 1.5)) & (xdata < (spe_mean + 1.5))
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

        # modeCharge vline 추가 (회색 점선)
        #plt.vlines(modeCharge, 0, max(fitted_curve), color='grey', linestyle='dashed', label=f'mode ~ {modeCharge:.3f}')
        # spe_mean (감마 분포의 mode, x좌표) vline 추가 (녹색 실선)
        plt.vlines(spe_mean, 0, max(fitted_curve), color='green', linestyle='solid', label=f'SPE mode ~ {spe_mean:.3f}')

    except Exception as e:
        print(f'fit didn\'t work for {channel} -> {e}')
        continue

plt.tight_layout()
plt.savefig('diagnostics/30t/' + date + 'diagnosticplot.png')

#print(results)  # 필요시 결과 출력 혹은 CSV 저장
pd.DataFrame(results).transpose().to_csv('/media/disk_l/30t-DATA/csv/phase1/non-validated/bnl30t_spe_fit_results_' + date + '.csv', index=False)

print('done with {}'.format(dateString))
exit(1)
