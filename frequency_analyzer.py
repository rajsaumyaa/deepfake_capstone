"""
Frequency-Domain Analyzer Module for Deepfake Forensic Prototype.
Computes 2D Fast Fourier Transform (FFT), centered log-magnitude spectra,
radial frequency distribution profiles, and high-frequency energy metrics.
"""

from typing import Dict, Tuple, Any
import numpy as np


def compute_2d_fft_spectrum(gray_frame: np.ndarray) -> Dict[str, Any]:
    """
    Computes the 2D Discrete Fourier Transform of a grayscale frame via NumPy FFT.
    
    Mathematical Formulation:
        F(u, v) = sum_{x=0}^{M-1} sum_{y=0}^{N-1} f(x, y) * exp(-j * 2 * pi * (u*x/M + v*y/N))
        F_shifted(u, v) = fftshift(F(u, v))  # Shifts DC component (0,0) to center
        Magnitude Spectrum |F_shifted(u, v)| = sqrt(Re^2 + Im^2)
        Log-Magnitude Spectrum S(u, v) = log(1 + |F_shifted(u, v)|)  # Stable log scaling
    
    Returns:
        Dict containing:
            - 'fft_shifted': Complex 2D array of centered Fourier coefficients
            - 'magnitude': 2D array of absolute magnitude
            - 'log_magnitude': 2D array of numerically stable log magnitude
            - 'normalized_log_magnitude': 2D array scaled to [0.0, 1.0] for plotting
    """
    if gray_frame.ndim != 2:
        raise ValueError("Input frame must be a 2D grayscale image.")
    
    # Cast to float64 for numerical precision
    f_float = gray_frame.astype(np.float64)
    
    # Compute 2D Fast Fourier Transform
    fft_raw = np.fft.fft2(f_float)
    
    # Shift zero-frequency (DC) component to spatial center (M//2, N//2)
    fft_shifted = np.fft.fftshift(fft_raw)
    
    # Magnitude |F(u,v)|
    magnitude = np.abs(fft_shifted)
    
    # Numerically stable log magnitude: log(1 + |F|) guarantees no log(0)
    log_magnitude = np.log1p(magnitude)
    
    # Normalize to [0.0, 1.0] for clean rendering
    min_val = float(np.min(log_magnitude))
    max_val = float(np.max(log_magnitude))
    
    if max_val > min_val:
        normalized_log_magnitude = (log_magnitude - min_val) / (max_val - min_val)
    else:
        normalized_log_magnitude = np.zeros_like(log_magnitude)
    
    return {
        "fft_shifted": fft_shifted,
        "magnitude": magnitude,
        "log_magnitude": log_magnitude,
        "normalized_log_magnitude": normalized_log_magnitude,
    }


def compute_high_frequency_energy_ratio(
    magnitude_spectrum: np.ndarray,
    cutoff_ratio: float = 0.4
) -> Tuple[float, float, float]:
    """
    Quantifies the proportion of spectral power residing in higher spatial frequencies.
    
    Mathematical Formulation:
        Center coordinates: (u_0, v_0) = (M/2, N/2)
        Radial distance: r(u, v) = sqrt((u - u_0)^2 + (v - v_0)^2)
        Maximum radius: R_max = sqrt(u_0^2 + v_0^2)
        Cutoff radius: R_c = cutoff_ratio * R_max, where cutoff_ratio in (0, 1)
        
        Power: P(u, v) = |F_shifted(u, v)|^2
        Total Spectral Energy = sum_{u, v} P(u, v)
        High-Frequency Energy = sum_{r(u, v) > R_c} P(u, v)
        
        High-Frequency Energy Ratio (HFER):
            HFER = High-Frequency Energy / (Total Spectral Energy + epsilon)
            
    Limitations:
        HFER is sensitive to video compression (H.264 quantization attenuates high frequencies),
        lighting sharpness, lens blur, and camera sensor noise. It is an exploratory signal,
        NOT a calibrated classification metric.
    """
    rows, cols = magnitude_spectrum.shape
    u0, v0 = rows / 2.0, cols / 2.0
    
    # Meshgrid of distance from DC center
    y_idx, x_idx = np.indices((rows, cols))
    r = np.sqrt((x_idx - v0)**2 + (y_idx - u0)**2)
    r_max = np.sqrt(u0**2 + v0**2)
    r_cutoff = cutoff_ratio * r_max
    
    power = magnitude_spectrum ** 2
    total_energy = float(np.sum(power))
    
    high_freq_mask = (r > r_cutoff)
    high_freq_energy = float(np.sum(power[high_freq_mask]))
    
    eps = 1e-12
    hfer = high_freq_energy / (total_energy + eps)
    
    return float(hfer), high_freq_energy, total_energy


def compute_radial_profile(magnitude_spectrum: np.ndarray, num_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes azimuthal / radial integration: average spectral magnitude as a function
    of radial distance from the DC center. Often studied in forensic literature
    (e.g. Durall et al., Frank et al.) to identify synthetic upsampling grid artifacts.
    """
    rows, cols = magnitude_spectrum.shape
    u0, v0 = rows / 2.0, cols / 2.0
    
    y_idx, x_idx = np.indices((rows, cols))
    r = np.sqrt((x_idx - v0)**2 + (y_idx - u0)**2)
    r_max = np.sqrt(u0**2 + v0**2)
    
    normalized_r = (r / r_max).flatten()
    mags = magnitude_spectrum.flatten()
    
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    radial_mean = np.zeros(num_bins, dtype=np.float64)
    
    for i in range(num_bins):
        mask = (normalized_r >= bin_edges[i]) & (normalized_r < bin_edges[i+1])
        if np.any(mask):
            radial_mean[i] = np.mean(mags[mask])
        else:
            radial_mean[i] = 0.0
            
    return bin_centers, radial_mean
