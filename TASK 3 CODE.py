import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

def task_3_spectral_aoa():
    # Load radar data
    mat = loadmat(r'C:\Users\st240865\Desktop\Information and electrical\RAWDATA.mat', squeeze_me=True, struct_as_record=False)
    raw = mat['data']
    ranges = mat['ranges']
    t = mat['datenums']
    wl = mat['wl']
    antpos0 = mat['antpos']

    # Antenna geometry
    x = np.real(antpos0)
    y = np.imag(antpos0)
    z = np.zeros_like(x)
    antpos = np.column_stack([x, y, z])
    B = antpos[1:, :2]  # Rx2–Rx5 vs Rx1 (X,Y only)
    K = 2 * np.pi / wl

    # Receiver phase correction
    phase_offsets_deg = np.array([0, 4.4003, 6.5031, 5.9530, 7.8495])
    phase_offsets_rad = np.deg2rad(phase_offsets_deg)
    raw_corr = raw * np.exp(-1j * phase_offsets_rad[None, None, :])

    # Use single strong range bin
    rb = raw_corr.shape[0] // 2
    signal = raw_corr[rb, :, :]  # shape: (time, receivers)

    # FFT and frequencies
    spec = np.fft.fft(signal, axis=0)
    freqs = np.fft.fftfreq(signal.shape[0], d=(t[1] - t[0]))

    # Cross-spectra: Rx2–Rx5 vs Rx1
    cross = np.zeros((len(freqs), 4), dtype=complex)
    for i in range(4):
        cross[:, i] = spec[:, i+1] * np.conj(spec[:, 0])

    # Select strongest spectral bins
    power = np.abs(cross).mean(axis=1)
    threshold = np.percentile(power, 75)
    valid_bins = np.where(power >= threshold)[0]

    # Solve AoA per valid spectral bin
    dcos_list = []
    for idx in valid_bins:
        phi = np.angle(cross[idx, :])
        d, *_ = np.linalg.lstsq(K * B, phi, rcond=None)
        if np.linalg.norm(d) <= 1:
            dcos_list.append(d)

    dcos = np.array(dcos_list)
    dz = np.sqrt(np.clip(1 - np.sum(dcos**2, axis=1), 0, None))
    xyz = np.column_stack([dcos, dz])

    # Plot 3a: Directional Cosines
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    sc1 = ax1.scatter(dcos[:, 0], dcos[:, 1], c=valid_bins[:len(dcos)], cmap='plasma', s=30, edgecolor='k', linewidth=0.3)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax1.add_artist(circle)
    ax1.set_xlabel("Directional Cosine X")
    ax1.set_ylabel("Directional Cosine Y")
    ax1.set_title("3a) Spectral-Domain AOA in Directional Cosines")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axis("equal")
    ax1.set_xlim(-0.3, 0.3)
    ax1.set_ylim(-0.3, 0.3)
    fig1.colorbar(sc1, label="Spectral Bin Index")
    fig1.tight_layout()
    fig1.savefig("Task_3a_DirectionalCosines.png", dpi=300)

    # 3b: Convert to 3D Cartesian using slant range (85 km)
    slant_range = 85_000  # meters
    x_real = dcos[:, 0] * slant_range
    y_real = dcos[:, 1] * slant_range
    z_real = dz * slant_range

    fig2 = plt.figure(figsize=(9, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    sc2 = ax2.scatter(x_real, y_real, z_real, c=valid_bins[:len(dcos)], cmap='plasma', s=30)
    ax2.set_title("3b) Spectral-Domain AOA in 3D Cartesian Coordinates")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_xlim(-30_000, 30_000)
    ax2.set_ylim(-30_000, 30_000)
    ax2.set_zlim(60_000, 90_000)
    ax2.scatter(0, 0, 0, color='black', s=80, marker='x', label='Radar Origin')
    ax2.legend()
    fig2.colorbar(sc2, label="Spectral Bin Index")
    fig2.tight_layout()
    fig2.savefig("Task_3b_CartesianAOA.png", dpi=300)

if __name__ == "__main__":
    task_3_spectral_aoa()