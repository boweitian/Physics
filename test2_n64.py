# smdp_scorenet_turbulence_ddim.py
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
    u = np.gradient(psi, axis=1) / dx
    return u

# ==== 2. 基础扩散过程 ====
def compute_decay_matrix(dt, alpha, n):
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    A = np.exp(-alpha * k2 * dt)
    return A

def heat_solver(u, A):
    U = fft2(u)
    U_new = A * U
    return np.real(ifft2(U_new))

def simulate_heat_trajectory(u0, dt, t_end, g, A, add_noise=True):
    num_steps = int(t_end / dt)
    trajectory = [u0.copy()]
    u = u0.copy()
    for _ in range(num_steps):
        u_det = heat_solver(u, A)
        if add_noise:
            noise = np.random.randn(*u.shape)
            u = u_det + np.sqrt(dt) * g * noise
        else:
            u = u_det
        trajectory.append(u.copy())
    return np.array(trajectory)

# ==== 3. U-Net 结构 ====
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        t_embed = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        xt = torch.cat([x, t_embed], dim=1)
        x_enc = self.encoder(xt)
        x_out = self.decoder(x_enc)
        return x_out

# ==== 4. DDIM采样器 ====
def ddim_reverse_step(x, model, t, t_prev, A, eta=0.0):
    with torch.no_grad():
        score = model(x, t)
        x_np = x.detach().cpu().numpy().squeeze(0).squeeze(0)
        heat_term = -heat_solver(x_np, A)
        heat_term = torch.tensor(heat_term, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)
        x_prev = x + (t - t_prev) * (score + heat_term)
    return x_prev

def inverse_simulation_ddim(model, x_final, dt, A, steps, device):
    x = torch.tensor(x_final, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    trajectory = [x.squeeze().cpu().numpy()]
    model.eval()
    for m in reversed(range(steps)):
        t_cur = torch.tensor([[(m+1)*dt]], dtype=torch.float32).to(device)
        t_prev = torch.tensor([[m*dt]], dtype=torch.float32).to(device)
        x = ddim_reverse_step(x, model, t_cur, t_prev, A)
        trajectory.append(x.squeeze().cpu().numpy())
    trajectory.reverse()
    return trajectory

# ==== 5. 模型训练 ====
def train_model(model, A, dt, dx, steps=100, iters=500, lr=1e-3, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for i in range(iters):
        psi = generate_streamfunction_initial_state(64, dx)
        u0 = compute_velocity_from_streamfunction(psi, dx)
        traj = simulate_heat_trajectory(u0, dt, steps*dt, 0.0, A, add_noise=False)

        losses = []
        for t_idx in range(1, steps):
            xt = torch.tensor(traj[t_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            xtm1 = torch.tensor(traj[t_idx - 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            t = torch.tensor([[t_idx * dt]], dtype=torch.float32).to(device)

            # 目标分数 = (xtm1 - xt) / dt - heat_term
            heat_term = -heat_solver(xt.squeeze().cpu().numpy(), A)
            target_score = ((xtm1 - xt) / dt).squeeze() - torch.tensor(heat_term, dtype=torch.float32).to(device)
            target_score = target_score.unsqueeze(0).unsqueeze(0)

            pred = model(xt, t)
            loss = criterion(pred, target_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if i % 50 == 0:
            print(f"Iter {i}, Loss = {np.mean(losses):.6f}")

    torch.save(model.state_dict(), "trained_model.pt")
    print("[Info] Model trained and saved to trained_model.pt")

def compute_2d_energy_spectrum(u):
    n = u.shape[0]
    assert u.shape[0] == u.shape[1], "Must be square input"
    U_hat = fft2(u)
    E_2d = 0.5 * np.abs(U_hat)**2

    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2)
    k_mag = np.round(k_mag).astype(int)

    k_max = n // 2
    E_k = np.zeros(k_max)
    k_vals = np.arange(k_max)

    for k in k_vals:
        mask = k_mag == k
        E_k[k] = np.sum(E_2d[mask])

    return k_vals, E_k

def visualize_spectrum_triple(u0, recon, uT):
    k0, E0 = compute_2d_energy_spectrum(u0)
    kr, Er = compute_2d_energy_spectrum(recon)
    kf, Ef = compute_2d_energy_spectrum(uT)

    plt.figure(figsize=(8, 5))
    plt.plot(k0, E0, label='Ground Truth u0')
    plt.plot(kr, Er, label='Reconstructed u0')
    plt.plot(kf, Ef, label='Final uT')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Wave number $k$")
    plt.ylabel("Energy Spectrum $E(k)$")
    plt.title("2D Isotropic Energy Spectra")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_spectra_comparison.png")
    plt.show()
# ==== 6. 主流程 ====
def main():
    n = 128
    dx = 2 * np.pi / n
    dt = 0.01
    T = 1.0
    nu = 1e-2  # Viscosity
    g = 0.1    # Noise magnitude for training
    steps = int(T/dt)
    model_filename = f"trained_model_vector_n{n}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 生成初态 (u0, v0) 和对应的涡量
    psi = generate_streamfunction_initial_state(n, dx, seed=42)
    u0, v0 = compute_velocity_from_streamfunction(psi, dx)
    uv0 = np.stack([u0, v0])
    vort_gt = compute_vorticity(u0, v0, dx)

    # 2. 正向扩散
    A = compute_decay_matrix(dt, nu, n)
    forward_traj = simulate_heat_trajectory(uv0, dt, T, g=0.0, A=A, add_noise=False)
    x_final_uv = forward_traj[-1] 
    u_final, v_final = x_final_uv[0], x_final_uv[1]
    vort_final = compute_vorticity(u_final, v_final, dx)

    # 3. 加载或训练模型
    model = UNet().to(device)
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename, map_location=device))
        print(f"[Info] Loaded trained model from {model_filename}.")
    else:
        print("[Info] Model not found. Training from scratch...")
        train_model(model, A, dt, dx, steps=steps, iters=500, lr=1e-4, g=g, n=n, device=device)

    # 4. 逆向重建
    recon_traj = inverse_simulation_ddim(x_final_uv, model, dt, A, steps=steps, device=device)
    uv_recon = recon_traj[0]
    u_recon, v_recon = uv_recon[0], uv_recon[1]
    vort_recon = compute_vorticity(u_recon, v_recon, dx)

    # ---- 5. 可视化涡量图 (范围统一) ----
    print("[Info] Generating visualization with UNIFIED colorbar for VORTICITY...")
    vmin_vort = min(vort_gt.min(), vort_recon.min())
    vmax_vort = max(vort_gt.max(), vort_recon.max())
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.canvas.manager.set_window_title('Vorticity Comparison (Unified Colorbar)')
    fig.suptitle('Vorticity Comparison (Unified Colorbar for Fair Comparison)', fontsize=16)
    titles_vort = ["Vorticity: Ground Truth ($t_0$)", "Vorticity: Reconstructed ($t_0$)", "Vorticity: Final ($t_T$)"]
    datas_vort = [vort_gt, vort_recon, vort_final]
    for ax, data, title in zip(axs, datas_vort, titles_vort):
        im = ax.imshow(data, origin='lower', cmap='RdBu_r', vmin=vmin_vort, vmax=vmax_vort)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("comparison_vorticity_unified_colorbar.png")
    plt.show()

    # ---- 6. 可视化涡量图 (范围不统一) ----
    print("[Info] Generating visualization with INDEPENDENT colorbars for VORTICITY...")
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.canvas.manager.set_window_title('Vorticity Comparison (Independent Colorbars)')
    fig2.suptitle('Vorticity Comparison (Independent Colorbars to Show Structure)', fontsize=16)
    for ax, data, title in zip(axs2, datas_vort, titles_vort):
        im = ax.imshow(data, origin='lower', cmap='RdBu_r') 
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig2.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("comparison_vorticity_independent_colorbar.png")
    plt.show()

    # ---- 7. ## NEW ## 可视化 u 分量对比图 (模仿旧代码) ----
    print("[Info] Generating visualization for U-VELOCITY (Original DDIM Comparison)...")
    vmin_u = min(u0.min(), u_recon.min())
    vmax_u = max(u0.max(), u_recon.max())
    
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 4))
    fig3.canvas.manager.set_window_title('U-Velocity Comparison (Original Style)')
    fig3.suptitle('U-Velocity Comparison (Original Style)', fontsize=16)
    
    # 准备数据和标题
    titles_u = ["Ground Truth u0", "Reconstructed u0", "Final uT"]
    datas_u = [u0, u_recon, u_final]

    # 绘制前两张图（范围统一）
    for i in range(2):
        ax = axs3[i]
        im = ax.imshow(datas_u[i], cmap='jet', origin='lower', vmin=vmin_u, vmax=vmax_u)
        ax.set_title(titles_u[i])
        ax.set_xticks([]); ax.set_yticks([])
        fig3.colorbar(im, ax=ax, shrink=0.8)

    # 单独绘制第三张图（范围不统一）
    ax = axs3[2]
    im = ax.imshow(datas_u[2], cmap='jet', origin='lower')
    ax.set_title(titles_u[2])
    ax.set_xticks([]); ax.set_yticks([])
    fig3.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("comparison_u_velocity.png")
    plt.show()

    # ---- 8. 可视化能量谱 ----
    visualize_spectrum_triple(u0, v0, u_recon, v_recon, u_final, v_final)
    
    print("[Info] Done.")

if __name__ == "__main__":
    main()