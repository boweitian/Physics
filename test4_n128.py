# test4_n128_with_vorticity.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# ==== 1. 生成湍流初态 ====
def generate_streamfunction_initial_state(n, dx, k0=10.0, epsilon=1e-6, seed=None):
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(n, d=dx) * 2 * np.pi
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

def compute_velocity_from_streamfunction(psi, dx):
    """
    从流函数 psi 计算速度分量：
      u =  ∂ψ/∂y
      v = -∂ψ/∂x
    """
    u = np.gradient(psi, axis=1) / dx
    v = -np.gradient(psi, axis=0) / dx
    return u, v

def compute_vorticity(u, v, dx):
    """
    计算涡量 ζ = ∂v/∂x - ∂u/∂y
    """
    dvdx = np.gradient(v, axis=1) / dx
    dudy = np.gradient(u, axis=0) / dx
    return dvdx - dudy

# ==== 2. 扩散过程 ====
def compute_decay_matrix(dt, alpha, n):
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    return np.exp(-alpha * k2 * dt)

def heat_solver(u, A):
    U = fft2(u)
    U_new = A * U
    return np.real(ifft2(U_new))

def simulate_heat_trajectory(u0, dt, t_end, g, A, add_noise=True):
    num_steps = int(t_end / dt)
    traj = [u0.copy()]
    u = u0.copy()
    for _ in range(num_steps):
        u_det = heat_solver(u, A)
        if add_noise:
            noise = np.random.randn(*u.shape)
            u = u_det + np.sqrt(dt) * g * noise
        else:
            u = u_det
        traj.append(u.copy())
    return np.array(traj)

# ==== 3. 深层 U-Net 结构 ====
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        t_embed = t.view(-1,1,1,1).expand(-1,1,x.shape[2],x.shape[3])
        xt = torch.cat([x, t_embed], dim=1)
        h  = self.encoder(xt)
        return self.decoder(h)

# ==== 4. DDIM 逆向采样 ====
def ddim_reverse_step(x, model, t, t_prev, A):
    with torch.no_grad():
        score  = model(x, t)
        x_np   = x.cpu().numpy().squeeze()
        heat   = -heat_solver(x_np, A).astype(np.float32)
        heat_t = torch.from_numpy(heat).unsqueeze(0).unsqueeze(0).to(x.device)
        return x + (t - t_prev) * (score + heat_t)

def inverse_simulation_ddim(model, x_final, dt, A, steps, device):
    x    = torch.from_numpy(x_final.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    traj = [x.squeeze().cpu().numpy()]
    model.eval()
    for m in reversed(range(steps)):
        t_cur  = torch.tensor([[(m+1)*dt]], dtype=torch.float32, device=device)
        t_prev = torch.tensor([[m*dt]],     dtype=torch.float32, device=device)
        x = ddim_reverse_step(x, model, t_cur, t_prev, A)
        traj.append(x.squeeze().cpu().numpy())
    return list(reversed(traj))

# ==== 5. 训练函数（保留 dtype 修正） ====
def train_model(model, A, dt, dx, steps, iters, lr, g, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for i in range(iters):
        psi = generate_streamfunction_initial_state(128, dx, seed=i)
        u0  = compute_velocity_from_streamfunction(psi, dx)[0]
        traj = simulate_heat_trajectory(u0, dt, steps*dt, g, A, add_noise=True)

        losses = []
        for t_idx in range(1, steps):
            xt   = torch.from_numpy(traj[t_idx].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            xtm1 = torch.from_numpy(traj[t_idx-1].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            t_sq = torch.tensor([[t_idx*dt]], dtype=torch.float32, device=device)

            heat_np      = -heat_solver(traj[t_idx], A).astype(np.float32)
            target_score = ((xtm1 - xt)/dt) - torch.from_numpy(heat_np).unsqueeze(0).unsqueeze(0).to(device)
            target_score = target_score.to(torch.float32)

            pred = model(xt, t_sq)
            loss = criterion(pred, target_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if i % 50 == 0:
            print(f"[Train] Iter {i}/{iters}, Loss = {np.mean(losses):.6f}")

    torch.save(model.state_dict(), "trained_model.pt")
    print("[Info] Model saved to trained_model.pt")

# ==== 6. 能量谱 & 可视化 ====
def compute_2d_energy_spectrum(u):
    n     = u.shape[0]
    U_hat = fft2(u)
    E2d   = 0.5 * np.abs(U_hat)**2
    kx    = np.fft.fftfreq(n)*n
    ky    = np.fft.fftfreq(n)*n
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.round(np.sqrt(kx**2+ky**2)).astype(int)
    k_max = n//2
    E_k   = np.zeros(k_max)
    for k in range(k_max):
        E_k[k] = E2d[k_mag==k].sum()
    return np.arange(k_max), E_k

def visualize_spectrum_triple(u0, recon, uT):
    k0, E0 = compute_2d_energy_spectrum(u0)
    kr, Er = compute_2d_energy_spectrum(recon)
    kf, Ef = compute_2d_energy_spectrum(uT)
    plt.figure(figsize=(8,5))
    plt.loglog(k0, E0, label='GT u₀')
    plt.loglog(kr, Er, label='Reco û₀')
    plt.loglog(kf, Ef, label='Final u_T')
    plt.xlabel("Wave number k")
    plt.ylabel("E(k)")
    plt.legend(); plt.grid(True, which="both", ls="--")
    plt.tight_layout(); plt.savefig("energy_spectra.png"); plt.show()

# ==== 7. 主流程 ====
def main():
    n    = 128
    dx   = 2 * np.pi / n
    dt   = 0.01
    T    = 1.0
    nu   = 1e-2
    g    = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成初态 & 计算速度、涡量
    psi    = generate_streamfunction_initial_state(n, dx, seed=42)
    u0, v0 = compute_velocity_from_streamfunction(psi, dx)
    vort_gt = compute_vorticity(u0, v0, dx)

    # 正向扩散
    A          = compute_decay_matrix(dt, nu, n)
    forward    = simulate_heat_trajectory(u0, dt, T, g=0.0, A=A, add_noise=False)
    x_final    = forward[-1]

    # 加载或训练模型
    model = UNet().to(device)
    if os.path.exists("trained_model.pt"):
        model.load_state_dict(torch.load("trained_model.pt", map_location=device))
        print("[Info] Loaded model.")
    else:
        print("[Info] Training model...")
        train_model(model, A, dt, dx, steps=int(T/dt), iters=500, lr=1e-3, g=g, device=device)

    # 逆向重建
    recon_traj = inverse_simulation_ddim(model, x_final, dt, A, steps=int(T/dt), device=device)
    u_recon    = recon_traj[0]
    v_recon    = np.zeros_like(u_recon)    # 无 v 分量时置零
    vort_recon = compute_vorticity(u_recon, v_recon, dx)

    # 末态涡量
    vort_final = compute_vorticity(x_final, np.zeros_like(x_final), dx)

    # —— 涡量图并排展示 ——
    vmin = min(vort_gt.min(), vort_recon.min(), vort_final.min())
    vmax = max(vort_gt.max(), vort_recon.max(), vort_final.max())

    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    for ax, data, title in zip(
        axs,
        [vort_gt, vort_recon, vort_final],
        ["Vorticity: Ground Truth", "Vorticity: Reconstructed", "Vorticity: Final"]
    ):
        im = ax.imshow(data, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig("comparison_vorticity.png"); plt.show()

    # —— 能量谱对比 ——
    visualize_spectrum_triple(u0, u_recon, x_final)

if __name__ == "__main__":
    main()
