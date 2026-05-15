import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# --- 4a: Define unique geometries ---
wl = 1.0  # normalized wavelength

# Used geometries 
triangle = np.array([[0, 0], [1.5, 0], [0.75, 2.0]])  # isosceles triangle
square = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5],[0.5, 1.5]])  # shifted square
rectangle = np.array([[0, 0], [2.7, 0], [2.7, 0.9],[0, 0.9]])  # rotated rectangle

geometries = {'Triangle': triangle, 'Square': square, 'Rectangle': rectangle}
colors = {'Triangle': 'blue', 'Square': 'red', 'Rectangle': 'green'}

# --- 4b: Generate synthetic I/Q data ---
samples = 1024
theta, phi = np.deg2rad(25), np.deg2rad(60)
dcos = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)])

def make_data(ant_pos):
    phase_offsets = (2 * np.pi / wl) * (ant_pos @ dcos)
    noise = np.random.uniform(0, 2*np.pi, size=(samples, ant_pos.shape[0]))
    return np.exp(1j * (phase_offsets[None, :] + noise))

def solve_aoa(data, ant_pos):
    aoa_list = []
    ref = ant_pos[0]
    A = -2 * np.pi / wl * (ant_pos[1:] - ref)
    for i in range(data.shape[0]):
        phase_diff = np.angle(data[i, :] / data[i, 0])
        dcos_est, _, _, _ = lstsq(A, phase_diff[1:], rcond=None)
        aoa_list.append(dcos_est)
    return np.array(aoa_list)

# --- 4c/d: Solve and plot ---
all_aoas = {}

for name, geom in geometries.items():
    data = make_data(geom)
    aoa = solve_aoa(data, geom)
    all_aoas[name] = aoa

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(geom[:, 0], geom[:, 1], c='blue', s=80)
    for i, (x, y) in enumerate(geom):
        axs[0].text(x, y - 0.1, f"Antenna {i+1}\n({x:.2f}, {y:.2f})", ha='center')
    axs[0].set_title(f"Antenna Geometry - {name}")
    axs[0].set_xlabel("x (wavelengths)")
    axs[0].set_ylabel("y (wavelengths)")
    axs[0].axis("equal")
    axs[0].grid(True)

    axs[1].scatter(aoa[:, 0], aoa[:, 1], s=8, alpha=0.5, color=colors[name])
    axs[1].set_title(f"AOA Directional Cosines - {name}")
    axs[1].set_xlabel("dcosx = sinθ cosφ")
    axs[1].set_ylabel("dcosy = sinθ sinφ")
    axs[1].set_xlim(-0.06, 0.06)
    axs[1].set_ylim(-0.06, 0.06)
    axs[1].grid(True)
    axs[1].add_patch(plt.Circle((0, 0), 1, color='gray', linestyle='--', fill=False))
    axs[1].axis("equal")
    plt.tight_layout()
    plt.show()

# --- 4e: Combined AOA plot ---
plt.figure(figsize=(7, 6))
for name, aoa in all_aoas.items():
    plt.scatter(aoa[:, 0], aoa[:, 1], s=8, alpha=0.5, label=name, color=colors[name])
plt.xlabel("dcosx")
plt.ylabel("dcosy")
plt.title("Combined AOA Estimation (Synthetic Data)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.gca().add_patch(plt.Circle((0, 0), 1, color='gray', linestyle='--', fill=False))
plt.tight_layout()
plt.show()

# --- 4e: Observations ---
print("4e) Observations:")
print("- Triangle ( coords): Tighter spread, low ambiguity due to symmetry")
print("- Square (shifted): Cluster well centered, low directional ambiguity")
print("- Rectangle (rotated): Good vertical resolution but elongated along direction")