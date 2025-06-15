import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# ==== 1. 生成湍流初态 (无变化) ====
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
    v = -np.gradient(psi, axis=0) / dx
    return u, v

def compute_vorticity(u, v, dx):
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

def heat_solver(field, A):
    """## MODIFIED: 现在处理单个场，将在向量场上循环调用"""
    F = fft2(field)
    F_new = A * F
    return np.real(ifft2(F_new))

def simulate_heat_trajectory(uv0, dt, t_end, g, A, add_noise=True):
    """## MODIFIED: 处理 (2, H, W) 的向量场 uv0"""
    num_steps = int(t_end / dt)
    traj = [uv0.copy()]
    uv = uv0.copy()
    for _ in range(num_steps):
        # 对 u 和 v 分量分别应用 heat_solver
        u_det = heat_solver(uv[0], A)
        v_det = heat_solver(uv[1], A)
        uv_det = np.stack([u_det, v_det])
        
        if add_noise:
            noise = np.random.randn(*uv.shape)
            uv = uv_det + np.sqrt(dt) * g * noise
        else:
            uv = uv_det
        traj.append(uv.copy())
    return np.array(traj)

# ==== 3. 深层 U-Net 结构 ====
class UNet(nn.Module):
    """## MODIFIED: U-Net 现在处理2通道输入和2通道输出"""
    def __init__(self):
        super().__init__()
        # 输入通道为3 (u, v, t)，输出通道为2 (score_u, score_v)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1)  # 输出2个通道
        )

    def forward(self, x_uv, t): # x_uv 的 shape: [B, 2, H, W]
        # 将时间t嵌入并与(u,v)场拼接
        t_embed = t.view(-1, 1, 1, 1).expand(-1, 1, x_uv.shape[2], x_uv.shape[3])
        xt = torch.cat([x_uv, t_embed], dim=1) # 拼接后 shape: [B, 3, H, W]
        h = self.encoder(xt)
        return self.decoder(h)

# ==== 4. DDIM 逆向采样 ====
def ddim_reverse_step(x_uv, model, t, t_prev, A):
    """## MODIFIED: 处理2通道的 x_uv"""
    with torch.no_grad():
        # 模型预测 (u,v) 两个分量的分数
        score_uv = model(x_uv, t)
        
        x_uv_np = x_uv.cpu().numpy().squeeze(0) # Shape: (2, H, W)
        
        # 对两个通道分别计算热方程项
        heat_u = -heat_solver(x_uv_np[0], A).astype(np.float32)
        heat_v = -heat_solver(x_uv_np[1], A).astype(np.float32)
        heat_t = torch.from_numpy(np.stack([heat_u, heat_v])).unsqueeze(0).to(x_uv.device)
        
        return x_uv + (t - t_prev) * (score_uv + heat_t)

def inverse_simulation_ddim(x_final_uv, model, dt, A, steps, device):
    """## MODIFIED: 处理2通道的 x_final_uv"""
    x_uv = torch.from_numpy(x_final_uv.astype(np.float32)).unsqueeze(0).to(device)
    traj = [x_uv.squeeze(0).cpu().numpy()]
    model.eval()
    for m in reversed(range(steps)):
        t_cur  = torch.tensor([[(m+1)*dt]], dtype=torch.float32, device=device)
        t_prev = torch.tensor([[m*dt]],     dtype=torch.float32, device=device)
        x_uv = ddim_reverse_step(x_uv, model, t_cur, t_prev, A)
        traj.append(x_uv.squeeze(0).cpu().numpy())
    return list(reversed(traj))

# ==== 5. 训练函数 ====
def train_model(model, A, dt, dx, steps, iters, lr, g, n, device="cpu"):
    """## MODIFIED: 训练整个向量场"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    print(f"[Info] Starting training for {iters} iterations on {device}...")
    for i in range(iters):
        # 1. 生成 (u,v) 向量场
        psi = generate_streamfunction_initial_state(n, dx, seed=i)
        u0, v0 = compute_velocity_from_streamfunction(psi, dx)
        uv0 = np.stack([u0, v0]) # Shape: (2, H, W)

        # 2. 对向量场进行正向扩散
        traj = simulate_heat_trajectory(uv0, dt, steps*dt, g, A, add_noise=True)

        losses = []
        for t_idx in range(1, steps):
            # 3. 准备2通道的 PyTorch 张量
            xt = torch.from_numpy(traj[t_idx].astype(np.float32)).unsqueeze(0).to(device)
            xtm1 = torch.from_numpy(traj[t_idx-1].astype(np.float32)).unsqueeze(0).to(device)
            t_sq = torch.tensor([[t_idx*dt]], dtype=torch.float32, device=device)

            # 4. 计算2通道的目标分数
            xt_np = traj[t_idx] # Shape: (2, H, W)
            heat_u_np = -heat_solver(xt_np[0], A).astype(np.float32)
            heat_v_np = -heat_solver(xt_np[1], A).astype(np.float32)
            heat_term = torch.from_numpy(np.stack([heat_u_np, heat_v_np])).unsqueeze(0).to(device)
            
            target_score = ((xtm1 - xt)/dt) - heat_term

            # 5. 模型预测与反向传播
            pred_score = model(xt, t_sq)
            loss = criterion(pred_score, target_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (i+1) % 50 == 0:
            print(f"[Train] Iter {i+1}/{iters}, Avg Loss = {np.mean(losses):.6f}")

    torch.save(model.state_dict(), f"trained_model_vector_n{n}.pt")
    print(f"[Info] Model saved to trained_model_vector_n{n}.pt")

# ==== 6. 能量谱 & 可视化 ====
def compute_2d_energy_spectrum(u, v):
    """## MODIFIED: 基于 (u,v) 场计算总动能谱"""
    n = u.shape[0]
    U_hat = fft2(u)
    V_hat = fft2(v)
    # 总动能 E_2d = 0.5 * (|û|^2 + |v̂|^2)
    E2d = 0.5 * (np.abs(U_hat)**2 + np.abs(V_hat)**2)
    
    kx = np.fft.fftfreq(n, d=1/n) # d=1/n gives k in integer units
    ky = np.fft.fftfreq(n, d=1/n)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.round(np.sqrt(kx**2 + ky**2)).astype(int)
    
    k_max = n // 2
    E_k = np.zeros(k_max)
    for k in range(1, k_max): # k=0 is mean, skip it
        mask = k_mag == k
        if np.any(mask):
            E_k[k] = E2d[mask].sum()
            
    return np.arange(k_max), E_k

def visualize_spectrum_triple(u0, v0, u_recon, v_recon, uT, vT):
    """## MODIFIED: 接收 (u,v) 对"""
    k_vals, E0 = compute_2d_energy_spectrum(u0, v0)
    _, Er = compute_2d_energy_spectrum(u_recon, v_recon)
    _, Ef = compute_2d_energy_spectrum(uT, vT)
    
    plt.figure(figsize=(8, 5))
    plt.loglog(k_vals[1:], E0[1:], label='Ground Truth $u_0$')
    plt.loglog(k_vals[1:], Er[1:], label='Reconstructed $u_0$', linestyle='--')
    plt.loglog(k_vals[1:], Ef[1:], label='Final $u_T$', alpha=0.7)
    plt.xlabel("Wavenumber $k$")
    plt.ylabel("Energy Spectrum $E(k)$")
    plt.title("2D Kinetic Energy Spectra")
    plt.legend(); plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("energy_spectra_comparison.png")
    plt.show()

# ==== 7. 主流程 ====
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