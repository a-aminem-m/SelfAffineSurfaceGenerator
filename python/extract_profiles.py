import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import RandomField as rf
from numba import jit

# --- Параметры генерации ---
N = 128  # размер сетки
H = 0.8  # показатель Херста — определяет «гладкость» поверхности
k1 = 4 / N # нижняя граница для отсечения спектра (определяет диапазон пространственных частот)
k2 = 32 / N # верхняя граница для отсечения спектра (определяет диапазон пространственных частот)

# --- Папка для результатов ---
os.makedirs("data/profiles", exist_ok=True)
os.makedirs("data/psd", exist_ok=True)

# --- Генерация изотропной поверхности ---
np.random.seed(42)
surface = rf.periodic_gaussian_random_field(dim=2, N=N, Hurst=H, k_low=k1, k_high=k2)
np.save("data/surface.npy", surface)

# --- Функция: извлечение ортогонального профиля ---
def extract_cartesian_profile(z, x, orient="horizontal"):
    if orient == "horizontal":
        return z[x, :]   # профиль вдоль x при фиксированном y = x
    else:
        return z[:, x]   # профиль вдоль y при фиксированном x = x

# --- Подсчёт PSD и моментов ---
@jit
def compute_psd_and_moments(profile, dx=1.0):
    N = len(profile)
    profile = profile - np.mean(profile)
    fft_vals = np.fft.fft(profile)
    freqs = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    psd = (np.abs(fft_vals) ** 2) / N
    mask = freqs >= 0
    freqs = freqs[mask]
    psd = psd[mask]

    m0 = np.trapezoid(psd, freqs)
    m2 = np.trapezoid(psd * freqs**2, freqs)
    m4 = np.trapezoid(psd * freqs**4, freqs)
    return freqs, psd, m0, m2, m4

# --- Индексы, по которым берём профили (равномерно по ширине) ---
indices = np.linspace(0, N-1, N, dtype=int)

# --- CSV с моментами ---
with open("data/moments.csv", "w") as f:
    f.write("index,m0,m2,m4\n")

# --- Сохраняем моменты для дальнейшего усреднения ---
m0_array = np.zeros(N)
m2_array = np.zeros(N)
m4_array = np.zeros(N)

# --- Обработка каждого профиля ---
for i, idx in enumerate(indices):
    profile = extract_cartesian_profile(surface, idx, orient="horizontal")
    print(i)
    freqs, psd, m0, m2, m4 = compute_psd_and_moments(profile)
    m0_array[i] = m0
    m2_array[i] = m2
    m4_array[i] = m4

    # Сохраняем профиль
    np.savetxt(f"data/profiles/profile_{idx}.csv", profile, delimiter=",")

    # Сохраняем PSD
    np.savetxt(f"data/psd/psd_{idx}.csv", np.column_stack((freqs, psd)), delimiter=",", header="k,PSD", comments='')

    # Добавляем строки в moments.csv
    with open("data/moments.csv", "a") as f:
        f.write(f"{idx},{m0},{m2},{m4}\n")

    # Сохраняем график PSD
    plt.figure()
    plt.loglog(freqs, psd)
    plt.title(f"PSD of profile {idx}")
    plt.xlabel("Wavevector k")
    plt.ylabel("Power Spectral Density")
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(f"data/psd/psd_{idx}.png", dpi=300)
    plt.close()

# --- Подсчёт и сохранение средних значений ---
mean_m0 = np.mean(m0_array)
mean_m2 = np.mean(m2_array)
mean_m4 = np.mean(m4_array)
rms_m0 = np.std(m0_array)
rms_m2 = np.std(m2_array)
rms_m4 = np.std(m4_array)


with open("data/mean_moments.txt", "w") as f:
    f.write(f"Mean m0: {mean_m0:.10f} +- {rms_m0:.5f}\n")
    f.write(f"Mean m2: {mean_m2:.10f} +- {rms_m2:.5f}\n")
    f.write(f"Mean m4: {mean_m4:.10f} +- {rms_m0:.5f}\n")

print("✅ Все профили и их спектры сохранены.")