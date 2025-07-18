#!/usr/bin/env python3
# check_spectra.py  –  сравнение PSD профилей с интегралом Найяка

import os, glob
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- путёвые константы -------------------------
SURF_FILE   = "data/surface.npy"
PROF_DIR    = "data/profiles"
OUT_DIR     = "data/compare_psd"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------- 1.  читаем поверхность --------------------
surface = np.load(SURF_FILE)
Ny, Nx  = surface.shape
dx      = 1.0                              # шаг сетки в условных единицах

# 2‑D PSD поверхности
fft2   = np.fft.fft2(surface - surface.mean())
psd2   = np.abs(fft2)**2 / (Nx*Ny)
kx_arr = np.fft.fftfreq(Nx, d=dx) * 2*np.pi
ky_arr = np.fft.fftfreq(Ny, d=dx) * 2*np.pi
kx_arr = np.fft.fftshift(kx_arr)
ky_arr = np.fft.fftshift(ky_arr)
psd2   = np.fft.fftshift(psd2)

# ------------------------- вспом. функция ----------------------------
def psd_profile_fft(profile, dx=1.0):
    N   = len(profile)
    f   = np.fft.fft(profile - profile.mean())
    kx  = np.fft.fftfreq(N, d=dx) * 2*np.pi
    psd = np.abs(f)**2 / N
    mask = kx >= 0
    return kx[mask], psd[mask]

# ------------------------- 2.  обход всех профилей -------------------
report = []
for prof_path in sorted(glob.glob(os.path.join(PROF_DIR, "profile_*.csv"))):
    idx = os.path.basename(prof_path).split("_")[1].split(".")[0]
    profile = np.loadtxt(prof_path, delimiter=",")

    # --- 2.1 PSD по FFT ---
    k_fft, psd_fft = psd_profile_fft(profile, dx)

    # --- 2.2 PSD по формуле Найяка ---
    # берём ряд ky, интеграл вдоль ky при фиксированном kx
    psd_nayak = []
    for k in k_fft:
        col_idx = np.argmin(np.abs(kx_arr - k))
        # интеграл \int_{-\infty}^{\infty} Φ(kx, ky) d ky
        psd_line = psd2[:, col_idx]          # все ky при данном kx
        psd_nayak.append(np.trapezoid(psd_line, ky_arr))
    psd_nayak = np.array(psd_nayak)

    # --- 2.3 относительная RMS‑ошибка ---
    rel_err = np.sqrt(np.mean((psd_fft - psd_nayak)**2)) / psd_fft.mean()
    report.append((idx, rel_err))

    # --- 2.4 график сравнения ---
    plt.figure(figsize=(6,4))
    plt.loglog(k_fft, psd_fft,  label="FFT‑PSD (профиль)")
    plt.loglog(k_fft, psd_nayak, '--', label="Интеграл Найяка")
    plt.xlabel("k")
    plt.ylabel("PSD")
    plt.title(f"Profile {idx}  –  rel.RMS err = {rel_err:.3%}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/psd_compare_{idx}.png", dpi=300)
    plt.close()

# ------------------------- 3.  сводка по ошибке -----------------------
with open(os.path.join(OUT_DIR, "psd_error_report.csv"), "w") as f:
    f.write("profile_idx,rel_rms_error\n")
    for idx, err in report:
        f.write(f"{idx},{err:.6f}\n")

print("✅ Сравнение PSD завершено. Сводка → compare_psd/psd_error_report.csv")
