import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# === 流函数生成（复现 decay_ic） ===
def generate_streamfunction_initial_state(n, dx, k0=10.0, epsilon=1e-6, seed=None):
    if seed is not None:
        np.random.seed(seed)
    kx = fftfreq(n, d=dx) * 2 * np.pi
    ky = fftfreq(n, d=dx) * 2 * np.pi
    kx[0] = epsilon
    ky[0] = epsilon
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    kk = np.sqrt(kx**2 + ky**2)

    c = 4.0 / (3.0 * np.sqrt(np.pi) * k0**5)
    es = c * (kk**4) * np.exp(-(kk/k0)**2)

    phase = np.exp(1j * 2 * np.pi * np.random.rand(n, n))
    wf = np.sqrt(kk * es / np.pi) * phase * (n * n)

    kk2 = kk**2
    kk2[kk2 == 0] = epsilon
    psi_hat = wf / kk2
    psi = np.real(ifft2(psi_hat))
    return psi

# === ψ → u, v ===
def compute_velocity_from_streamfunction(psi, dx):
    u = np.gradient(psi, axis=1) / dx
    v = -np.gradient(psi, axis=0) / dx
    return u, v

# === 构造扩散核 A ===
def compute_decay_matrix(dt, nu, n, dx):
    k = fftfreq(n, d=dx) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    return np.exp(-nu * k2 * dt)

# === 单步傅里叶扩散器 ===
def heat_solver(u, A):
    U_hat = fft2(u)
    U_new = A * U_hat
    return np.real(ifft2(U_new))

# === 扩散主函数（单变量扩散） ===
def simulate_diffusion(u0, A, dt, T, g=0.0, add_noise=False):
    num_steps = int(T / dt)
    u = u0.copy()
    traj = [u0.copy()]
    for _ in range(num_steps):
        u = heat_solver(u, A)
        if add_noise:
            noise = np.random.randn(*u.shape)
            u += np.sqrt(dt) * g * noise
        traj.append(u.copy())
    return np.array(traj)

# === 参数设置 ===
n = 128
Lx = Ly = 2 * np.pi
dx = Lx / n
dt = 0.01
T = 2.0         # 加长时间，与 MATLAB 中 Setups.T 对齐
nu = 1 / 100     # 增大扩散系数，与 MATLAB 中 Setups.nu 对齐
seed = 0

# === 流函数与速度场初始化 ===
psi = generate_streamfunction_initial_state(n, dx, k0=10.0, seed=seed)
u0, v0 = compute_velocity_from_streamfunction(psi, dx)

# === 扩散核 ===
A = compute_decay_matrix(dt, nu, n, dx)

# === 模拟扩散行为（仅对 u 分量）===
traj = simulate_diffusion(u0, A, dt, T)

# === 可视化 ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Initial u")
plt.imshow(traj[0], cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Middle u (T/2)")
plt.imshow(traj[len(traj)//2], cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Final u (T=2.0)")
plt.imshow(traj[-1], cmap='jet', origin='lower')
plt.colorbar()

plt.tight_layout()
plt.savefig("u_diffusion_simulation_fixed.png")

# inverse

def inverse_diffusion(u_final, T, nu, dx, epsilon=1e-8):
    n = u_final.shape[0]
    k = fftfreq(n, d=dx) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    decay = np.exp(-nu * k2 * T)
    decay[decay < epsilon] = epsilon  # 避免数值爆炸
    u_final_hat = fft2(u_final)
    u0_hat = u_final_hat / decay
    u0_reconstructed = np.real(ifft2(u0_hat))
    return u0_reconstructed

u_final = traj[-1]
u0_reconstructed = inverse_diffusion(u_final, T=T, nu=nu, dx=dx)

# 可视化对比原始与重建
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original u0")
plt.imshow(traj[0], cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Reconstructed u0 (from uT)")
plt.imshow(u0_reconstructed, cmap='jet', origin='lower')
plt.colorbar()

plt.tight_layout()
plt.savefig("reconstructed_u0.png")
