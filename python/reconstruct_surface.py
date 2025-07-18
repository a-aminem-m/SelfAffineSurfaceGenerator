# reconstruct_surface_opt.py
"""
Восстановление изотропной 2‑D поверхности по набору 1‑D профилей
=================================================================

• Читает профили из data/profiles/profile_<idx>.csv  
• Задаёт ограниченный по радиусу спектр (k <= k_s)  
• Оптимизирует амплитуды спектра (фазы случайные) так, чтобы минимизировать
  J = w1*J1 + w2*J2 + w3*J3, где
    J1 – совпадение профилей,
    J2 – изотропность спектра,
    J3 – совпадение PSD профилей.

Результат:
  results/reconstructed_surface.npy   – восстановленная поверхность
  results/opt_history.csv             – значение функционала по итерациям
  results/profile_comparison_<idx>.png – графики совпадения профилей
"""

import os
import glob
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ------------------- параметры -------------------
N       = 2048           # размер сетки поверхности
L       = 1.0            # физическая длина
h       = L / N          # шаг по пространству
k_s     = 64 * 2 * np.pi / L   # радиус отсечения спектра

w1, w2, w3 = 1.0, 1.0, 1.0      # веса функционалов
max_iter   = 50                # макс. итераций оптимизатора

PROFILE_DIR = "data/profiles"
OUT_DIR     = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------- загружаем профили -------------------
profiles   = []   # список 1‑D массивов длиной N
positions  = []   # соответствующие индексы y
for f in sorted(glob.glob(os.path.join(PROFILE_DIR, "profile_*.csv"))):
    profiles.append(np.loadtxt(f, delimiter=","))
    positions.append(int(f.split("_")[1].split(".")[0]))
profiles  = np.array(profiles)
positions = np.array(positions)
assert profiles.shape[1] == N, "Все профили должны иметь длину N"

# ------------------- сетка частот -------------------
kx = fftfreq(N, d=h) * 2 * np.pi
ky = kx.copy()
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K_mag  = np.sqrt(KX**2 + KY**2)
mask   = K_mag <= k_s        # разрешённые коэффициенты

# индексы разрешённых точек в спектре
y_idx, x_idx = np.where(mask)
M = len(x_idx)               # число оптимизируемых амплитуд
print(f"Optimised coefficients: {M} (≈ {M/N**2*100:.2f}% от общего спектра)")

# ------------------- вспомогательные функции -------------------
np.random.seed(0)
phases_full = np.exp(1j * 2 * np.pi * np.random.rand(N, N))  # фиксируем фазы


def build_spectrum(amps_flat):
    """Собираем полный 2‑D спектр из оптимизируемых амплитуд."""
    Z = np.zeros((N, N), dtype=complex)
    Z[y_idx, x_idx] = amps_flat * phases_full[y_idx, x_idx]
    # зеркальное отражение для отрицательных частот
    Z[-y_idx, -x_idx] = np.conj(Z[y_idx, x_idx])
    return Z

iter_counter = [0]

def objective(amps_flat):
    """Функционал J = w1*J1 + w2*J2 + w3*J3."""
    iter_counter[0] += 1
    if iter_counter[0] % 5 == 0:       # печать каждые 5 вызовов
        print(f"  eval {iter_counter[0]}")

    Z_hat = build_spectrum(amps_flat)
    z     = np.real(ifft2(ifftshift(Z_hat)))

    # J1: совпадение профилей
    J1 = 0.0
    for prof, pos in zip(profiles, positions):
        diff = z[pos, :] - prof
        J1 += np.mean(diff**2)

    # J2: изотропность спектра (разброс в кольцах)
    #   считаем стандартное отклонение лог(PSD) по углам
    psd2d = np.abs(Z_hat)**2
    radial_bins = np.linspace(0, k_s, 50)
    J2 = 0.0
    for i in range(len(radial_bins)-1):
        msk = (K_mag >= radial_bins[i]) & (K_mag < radial_bins[i+1])
        if np.any(msk):
            vals = np.log(psd2d[msk] + 1e-20)
            J2 += np.var(vals)
    J2 /= (len(radial_bins)-1)

    # J3: совпадение спектров профилей
    J3 = 0.0
    for prof in profiles:
        prof_fft = np.fft.fft(prof - prof.mean())
        psd_prof = np.abs(prof_fft)**2 / N
        psd_prof = psd_prof[:N//2]  # k >= 0

        # спектр из 2D согласно Найяку
        # берем столбец k_y = 0
        psd_from_surface = psd2d[N//2, N//2:]
        min_len = min(len(psd_prof), len(psd_from_surface))
        diff = np.log(psd_prof[:min_len] + 1e-20) - np.log(psd_from_surface[:min_len] + 1e-20)
        J3 += np.mean(diff**2)

    total = w1*J1 + w2*J2 + w3*J3
    return total

# ------------------- начальное приближение амплитуд -------------------
amp0 = np.random.rand(M) * 0.1

def cb(xk):
    # xk — текущий вектор амплитуд
    k = cb.counter
    if k % 1 == 0:      # печатать каждую итерацию
        cur_val = objective(xk)
        print(f"Iter {k:4d}   J = {cur_val: .3e}")
    cb.counter += 1
cb.counter = 0

# ------------------- оптимизация -------------------
print("Запускаю оптимизацию …")
res = minimize(objective, amp0, method="L-BFGS-B", callback=cb, options={"maxiter": max_iter, "disp": True})
print("Оптимизация завершена, статус:", res.message)

# ------------------- финальная поверхность -------------------
Z_opt = build_spectrum(res.x)
recon_surface = np.real(ifft2(ifftshift(Z_opt)))
np.save(os.path.join(OUT_DIR, "reconstructed_surface.npy"), recon_surface)

# ------------------- визуализация профилей -------------------
for prof, pos in zip(profiles, positions):
    plt.figure()
    plt.plot(prof, label="Target")
    plt.plot(recon_surface[pos, :], label="Reconstructed", linestyle="--")
    plt.title(f"Profile at y={pos}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"compare_profile_{pos}.png"), dpi=300)
    plt.close()

print("✅ Восстановленная поверхность сохранена в results/")
