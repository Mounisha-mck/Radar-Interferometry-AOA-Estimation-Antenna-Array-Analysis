
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D

def task_2():
    # Load radar data
    mat = scipy.io.loadmat(r'C:\Users\st240865\Desktop\Information and electrical\RAWDATA.mat')
    data = mat['data']
    ranges = mat['ranges'].squeeze()
    wl = float(mat['wl'].squeeze())
    antpos_complex = mat['antpos'].squeeze()

    # Antenna positions
    x_pos = np.real(antpos_complex)
    y_pos = np.imag(antpos_complex)
    z_pos = np.zeros_like(x_pos)
    antpos = np.vstack((x_pos, y_pos, z_pos))

    # Use Rx2–Rx5
    rx_idx = [1, 2, 3, 4]
    rx_phases_deg = np.array([0, 4.4003, 6.5031, 5.9530, 7.8495])
    rx_phases_rad = np.deg2rad(rx_phases_deg[rx_idx])
    phase_corr = np.exp(-1j * rx_phases_rad)[None, None, :]
    corrected_data = data[:, :, rx_idx] * phase_corr

    # Select top 20 strongest range bins based on average SNR
    snr_total = np.abs(corrected_data).mean(axis=(1, 2))
    top_indices = np.argsort(snr_total)[-20:]
    top_indices = np.sort(top_indices)

    selected_data = corrected_data[top_indices]
    selected_ranges = ranges[top_indices]

    # Coherent integration
    avg_data = np.mean(selected_data, axis=1)
    ref_signal = avg_data[:, 0]
    phase_diffs = np.angle(avg_data / ref_signal[:, None])

    # Relative positions and A matrix
    rel_pos = antpos[:, rx_idx] - antpos[:, rx_idx[0]].reshape(3, 1)
    A = (-2 * np.pi / wl) * rel_pos[:, 1:].T

    # Solve for directional cosines
    dcos_list = []
    for i in range(phase_diffs.shape[0]):
        dphi = phase_diffs[i, 1:]
        dcos, _, _, _ = lstsq(A, dphi, rcond=None)
        dcos_list.append(dcos)
    dcos_list = np.array(dcos_list)

    # Convert to Cartesian coordinates
    x = dcos_list[:, 0] * selected_ranges
    y = dcos_list[:, 1] * selected_ranges
    z = np.sqrt(np.clip(1 - dcos_list[:, 0]**2 - dcos_list[:, 1]**2, 0, None)) * selected_ranges

    # Plot directional cosines
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    sc1 = ax1.scatter(dcos_list[:, 0], dcos_list[:, 1], c=selected_ranges, cmap='turbo', s=30)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax1.add_artist(circle)
    ax1.set_xlabel("dcosx = sin(θ)·cos(φ)")
    ax1.set_ylabel("dcosy = sin(θ)·sin(φ)")
    ax1.set_title("Task 2a: AOA in Directional Cosines")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.set_xlim(-0.3, 0.3)
    ax1.set_ylim(-0.3, 0.3)
    fig1.colorbar(sc1, label="Range (m)")
    fig1.tight_layout()
    fig1.savefig("Task_2a_DirectionalCosines_SelectedStrong.png", dpi=300)

    # Plot 3D Cartesian AOA
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    sc2 = ax2.scatter(x, y, z, c=z, cmap='turbo', s=30)
    ax2.set_title("Task 2b: AOA in 3D Cartesian Coordinates")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_xlim(-25, 25)
    ax2.set_ylim(-25, 25)
    ax2.set_zlim(60, 120)
    ax2.scatter(0, 0, 0, color='black', s=80, marker='x', label='Radar Origin')
    ax2.legend()
    fig2.colorbar(sc2, label="Altitude (m)")
    fig2.tight_layout()
    fig2.savefig("Task_2b_CartesianAOA_SelectedStrong.png", dpi=300)

if __name__ == "__main__":
    task_2()
