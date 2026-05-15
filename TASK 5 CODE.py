import numpy as np
import matplotlib.pyplot as plt

print("Part 5a: 7-element hexagonal array radiation pattern")

# --- Parameters ---
c = 299792458  # Speed of light (m/s)
wl = 94.57175331  # Wavelength (m)
spacing = 0.7 * wl  # Element spacing (between 0.5 and 1 wavelength)

# --- 5a: Define 7-element hexagonal array ---
ant_pos_7 = np.array([
    0 + 0j,
    spacing + 0j,
    spacing * 0.5 + spacing * np.sqrt(3)/2 * 1j,
    -spacing * 0.5 + spacing * np.sqrt(3)/2 * 1j,
    -spacing + 0j,
    -spacing * 0.5 - spacing * np.sqrt(3)/2 * 1j,
    spacing * 0.5 - spacing * np.sqrt(3)/2 * 1j
])

plt.figure(figsize=(6,6))
plt.plot(np.real(ant_pos_7), np.imag(ant_pos_7), 'o', markersize=10)
for i, pos in enumerate(ant_pos_7):
    plt.text(np.real(pos), np.imag(pos), f'A{i+1}', fontsize=10, ha='right')
plt.title('5a: 7-Antenna Hexagonal Array Geometry')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# --- Radiation pattern function ---
def ant_pattern(ant_pos, phase, wl, amplitude=None):
    max_theta = 90
    step = 1
    if amplitude is None:
        amplitude = np.ones(len(ant_pos))
    
    dummy = np.linspace(-max_theta, max_theta, int((2*max_theta+1)/step))
    nx = len(dummy)
    ny = nx
    maxdcos = np.sin(np.radians(max_theta))
    dcosx = (np.linspace(1, nx, nx)/(nx-1) - 0.5) * 2 * maxdcos
    dcosy = (np.linspace(1, ny, ny)/(ny-1) - 0.5) * 2 * maxdcos
    
    e = np.zeros((ny, nx), dtype=complex)
    for ix in range(nx):
        for iy in range(ny):
            phase_shift = (2 * np.pi / wl) * (
                np.real(ant_pos) * dcosx[ix] + np.imag(ant_pos) * dcosy[iy])
            e[iy, ix] = np.sum(amplitude * np.exp(1j * (phase_shift + np.radians(phase))))
    
    power = np.abs(e)**2
    amp = 10 * np.log10(power / np.nanmax(power) + 1e-10)  # avoid log(0)
    mask = np.sqrt(dcosx[np.newaxis, :]**2 + dcosy[:, np.newaxis]**2) > 1
    amp[mask] = np.nan
    return dcosx, dcosy, amp

# --- 5a: Radiation pattern ---
phase_7 = np.zeros(len(ant_pos_7))
dcosx_7, dcosy_7, amp_7 = ant_pattern(ant_pos_7, phase_7, wl)

plt.figure(figsize=(8,6))
plt.pcolor(dcosx_7, dcosy_7, amp_7, shading='auto', cmap='jet')
plt.colorbar(label='Normalized Power (dB)')
plt.title('5a: Radiation Pattern (7-Antenna, Equal Phase)')
plt.xlabel('dcosx')
plt.ylabel('dcosy')
plt.axis('equal')
plt.clim(-30, 0)
plt.tight_layout()
plt.show()

print("Part 5b: 7x7 hexagonal array with beam steering")

# --- 5b: 7x7 hexagonal array ---
rows, cols = 7, 7
x_coords, y_coords = [], []
for row in range(rows):
    for col in range(cols):
        x = col * spacing
        y = row * spacing * np.sqrt(3)/2
        if row % 2 == 1:
            x += spacing / 2
        x_coords.append(x)
        y_coords.append(y)

ant_pos_49 = np.array(x_coords) + 1j * np.array(y_coords)
ant_pos_49 -= np.mean(ant_pos_49)

plt.figure(figsize=(8,8))
plt.plot(np.real(ant_pos_49), np.imag(ant_pos_49), 'o')
plt.title('5b: 7x7 Hexagonal Antenna Array')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Equal phase
phase_49_equal = np.zeros(len(ant_pos_49))
dcosx_49, dcosy_49, amp_49_equal = ant_pattern(ant_pos_49, phase_49_equal, wl)

# Beam steering
phi_steer = 45
theta_steer = 15
phase_49_steered = -(2 * np.pi / wl) * (
    np.real(ant_pos_49) * np.sin(np.radians(theta_steer)) * np.cos(np.radians(phi_steer)) +
    np.imag(ant_pos_49) * np.sin(np.radians(theta_steer)) * np.sin(np.radians(phi_steer))
)

_, _, amp_49_steered = ant_pattern(ant_pos_49, phase_49_steered, wl)

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.pcolor(dcosx_49, dcosy_49, amp_49_equal, shading='auto', cmap='jet')
plt.colorbar(label='Normalized Power (dB)')
plt.title('5b: Equal Phase')
plt.xlabel('dcosx')
plt.ylabel('dcosy')
plt.axis('equal')
plt.clim(-30, 0)

plt.subplot(1,2,2)
plt.pcolor(dcosx_49, dcosy_49, amp_49_steered, shading='auto', cmap='jet')
plt.colorbar(label='Normalized Power (dB)')
plt.title('5b: Beam Steered to φ=45°, θ=15°')
plt.xlabel('dcosx')
plt.ylabel('dcosy')
plt.axis('equal')
plt.clim(-30, 0)

plt.tight_layout()
plt.show()

print("Part 5c: Amplitude tapering comparison")

# --- 5c: Tapering ---
ant_distances = np.abs(ant_pos_49)
max_dist = np.max(ant_distances)
amplitude_taper = 0.5 * (1 + np.cos(np.pi * ant_distances / max_dist))
amplitude_taper /= np.max(amplitude_taper)

_, _, amp_49_tapered = ant_pattern(ant_pos_49, phase_49_equal, wl, amplitude=amplitude_taper)

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.pcolor(dcosx_49, dcosy_49, amp_49_equal, shading='auto', cmap='jet')
plt.colorbar(label='Normalized Power (dB)')
plt.title('5c: Equal Amplitude')
plt.xlabel('dcosx')
plt.ylabel('dcosy')
plt.axis('equal')
plt.clim(-30, 0)

plt.subplot(1,2,2)
plt.pcolor(dcosx_49, dcosy_49, amp_49_tapered, shading='auto', cmap='jet')
plt.colorbar(label='Normalized Power (dB)')
plt.title('5c: Raised Cosine Tapering')
plt.xlabel('dcosx')
plt.ylabel('dcosy')
plt.axis('equal')
plt.clim(-30, 0)

plt.tight_layout()
plt.show()

print("Part 5d: Including element pattern effects")

# --- 5d: Dipole element pattern ---
theta_grid = np.sqrt(dcosx_49[np.newaxis, :]**2 + dcosy_49[:, np.newaxis]**2)
theta_grid_clipped = np.clip(theta_grid, 0, 1)  # avoid domain error

theta_rad = np.arcsin(theta_grid_clipped)
element_pattern = (3/2) * (np.sin(theta_rad)**2)
element_pattern /= np.nanmax(element_pattern)

amp_49_total_linear = 10**(amp_49_tapered / 10) * element_pattern
amp_49_total_db = 10 * np.log10(amp_49_total_linear / np.nanmax(amp_49_total_linear) + 1e-10)
amp_49_total_db[np.isnan(amp_49_tapered)] = np.nan

plt.figure(figsize=(8,6))
plt.pcolor(dcosx_49, dcosy_49, amp_49_total_db, shading='auto', cmap='jet')
plt.colorbar(label='Normalized Power (dB)')
plt.title('5d: Total Pattern (Array × Element)')
plt.xlabel('dcosx')
plt.ylabel('dcosy')
plt.axis('equal')
plt.clim(-30, 0)
plt.tight_layout()
plt.show()
