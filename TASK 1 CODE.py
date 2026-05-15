import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import get_window
from scipy.io import loadmat
from itertools import combinations

# Load .mat file
mat = loadmat(r'C:\Users\st240865\Desktop\Information and electrical\RAWDATA.mat')
data = mat["data"]
ranges = mat["ranges"].squeeze()
t_samp = mat["datenums"].squeeze()
antpos = mat["antpos"].squeeze()
wl, f = float(mat["wl"]), float(mat["f"])

no_range, no_time, no_rx = data.shape
t = np.arange(no_time)
dt = t_samp[1] - t_samp[0]
freq = np.fft.fftfreq(no_time, d=dt)
freq_shifted = fftshift(freq)

# --- 1a: Time series power ---
def plot_time_series_power(channel_index, title):
    power = 10 * np.log10(np.abs(data[:, :, channel_index])**2 + 1e-12)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(np.arange(no_time), ranges, power, shading='auto', cmap='viridis')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time Index')
    plt.ylabel('Range (km)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_time_series_power(0, "1a: Time Series Power - Rx1")
plot_time_series_power(1, "1a: Time Series Power - Rx2")

# --- 1b: Power Profiles ---
plt.figure(figsize=(8, 5))
for ch in range(no_rx):
    avg_power = 10 * np.log10(np.mean(np.abs(data[:, :, ch])**2, axis=1) + 1e-12)
    plt.plot(ranges, avg_power, label=f'Rx{ch+1}')
plt.xlabel('Range (km)')
plt.ylabel('Average Power (dB)')
plt.title('1b: Power Profiles for All Receivers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 1c: IQ Scatter Before/After Coherent Integration ---
n_segments = no_time // 2
data_trimmed = data[:, :n_segments * 2, :]
data_ci = data_trimmed.reshape(no_range, n_segments, 2, no_rx).mean(axis=2)

range_bin = np.argmin(np.abs(ranges - 89))
for rx in range(no_rx):
    plt.figure(figsize=(12, 5))
    
    # Before integration
    plt.subplot(1, 2, 1)
    plt.scatter(data[range_bin, :, rx].real, 
                data[range_bin, :, rx].imag, 
                c='blue', alpha=0.5, label='Before CI')
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.title(f"Rx{rx+1} IQ Before Coherent Integration")
    plt.axis("equal")
    plt.legend()
    
    # After integration
    plt.subplot(1, 2, 2)
    plt.scatter(data_ci[range_bin, :, rx].real, 
                data_ci[range_bin, :, rx].imag, 
                c='red', alpha=0.5, label='After CI')
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.title(f"Rx{rx+1} IQ After 2-Point Coherent Integration")
    plt.axis("equal")
    plt.legend()
    
    plt.suptitle(f'1c: IQ Scatter Plots - Rx{rx+1}')
    plt.tight_layout()
    plt.show()

# --- 1d: Amplitude & Phase Spectrum for Rx1 and Rx4 ---
def plot_spectrum(signal, title_prefix):
    fft_data = fftshift(fft(signal, axis=1), axes=1)
    amplitude = 20 * np.log10(np.abs(fft_data) + 1e-12)
    phase = np.angle(fft_data)
    freqs = np.linspace(-0.5, 0.5, fft_data.shape[1])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(amplitude, aspect="auto", extent=[freqs[0], freqs[-1], ranges[-1], ranges[0]], cmap="viridis")
    plt.colorbar(label='Amplitude (dB)')
    plt.title(f'{title_prefix} Amplitude Spectrum')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Range (km)')
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.imshow(phase, aspect="auto", extent=[freqs[0], freqs[-1], ranges[-1], ranges[0]], cmap="twilight")
    plt.colorbar(label='Phase (rad)')
    plt.title(f'{title_prefix} Phase Spectrum')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Range (km)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

plot_spectrum(data[:, :, 0], "1d: Rx1")
plot_spectrum(data[:, :, 3], "1d: Rx4")

# --- 1e: Enhanced Cross-Spectral Analysis ---
def compute_cross_spectra_improved(x, y, decim=4, nfft=None):
    """
    Improved cross-spectral calculation with proper decimation
    """
    if nfft is None:
        nfft = x.shape[1] // decim
        
    # Apply windowing before FFT
    window = np.hanning(nfft)
    window /= np.linalg.norm(window)
    
    # Initialize output arrays
    cross_amp = np.zeros((x.shape[0], nfft))
    cross_phase = np.zeros((x.shape[0], nfft))
    
    for r in range(x.shape[0]):
        # Segment the data with overlap
        for i in range(0, x.shape[1] - nfft + 1, nfft // 2):
            x_seg = x[r, i:i+nfft] * window
            y_seg = y[r, i:i+nfft] * window
            
            # Compute FFTs
            X = fft(x_seg, n=nfft)
            Y = fft(y_seg, n=nfft)
            
            # Accumulate cross-spectrum
            cross = X * np.conj(Y)
            cross_amp[r] += np.abs(cross)
            cross_phase[r] += np.angle(cross)
    
    # Average the results
    num_segments = (x.shape[1] - nfft) // (nfft // 2) + 1
    cross_amp /= num_segments
    cross_phase /= num_segments
    
    # Shift zero frequency to center
    cross_amp = fftshift(cross_amp, axes=1)
    cross_phase = fftshift(cross_phase, axes=1)
    
    return cross_amp, cross_phase

# Parameters
decim_factor = 4
nfft = no_time // decim_factor
freqs = fftshift(fftfreq(nfft, d=dt))

# Compute and plot cross-spectra for all combinations of Rx2-Rx5
rx_pairs = list(combinations(range(1, 5), 2))  # All combinations of Rx2-Rx5

for i, (a, b) in enumerate(rx_pairs):
    # Compute cross-spectra
    amp, phase = compute_cross_spectra_improved(data[:, :, a], data[:, :, b], 
                                              decim=decim_factor, nfft=nfft)
    
    # Convert amplitude to dB scale
    amp_db = 20 * np.log10(amp + 1e-12)
   
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Amplitude plot
    plt.subplot(1, 2, 1)
    plt.pcolormesh(freqs, ranges, amp_db, shading='auto', cmap='viridis')
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Range (km)')
    plt.title(f'Rx{a+1}-Rx{b+1} Cross-Spectrum Amplitude')
    plt.ylim([ranges.min(), ranges.max()])
    plt.clim([-50, 50])  # Adjust color limits as needed
    
    # Phase plot
    plt.subplot(1, 2, 2)
    plt.pcolormesh(freqs, ranges, phase, shading='auto', cmap='twilight')
    plt.colorbar(label='Phase (rad)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Range (km)')
    plt.title(f'Rx{a+1}-Rx{b+1} Cross-Spectrum Phase')
    plt.ylim([ranges.min(), ranges.max()])
    
    plt.tight_layout()
    plt.show()


# --- 1f: Enhanced Antenna Layout ---
x_coords = np.real(antpos)
y_coords = np.imag(antpos)

plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, c='red', s=150, edgecolors='black')

# Add labels
for i in range(len(x_coords)):
    plt.text(x_coords[i] + 10, y_coords[i] + 10,
             f'Rx{i+1}\n({x_coords[i]:.2f}, {y_coords[i]:.2f})m',
             fontsize=10, ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.title('1f: Antenna Positions (X-Y Plane)', fontsize=14)
plt.xlabel('East-West Distance (m)', fontsize=12)
plt.ylabel('North-South Distance (m)', fontsize=12)
plt.xlim([-500, 500])
plt.ylim([-500, 500])
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_aspect('equal', adjustable='box')

# Add reference lines and compass
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.text(450, -450, 'X: East\nY: North',
         ha='right', va='bottom', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
