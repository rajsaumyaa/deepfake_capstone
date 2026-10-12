"""
Cross-Domain Spatial-Frequency Interaction and Temporal Consistency Network
B.Tech Capstone Project-I: Exploratory Video Forensics Prototype
Streamlit Web Interface (app.py)
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
import matplotlib.pyplot as plt

from video_processor import (
    validate_video_file,
    get_video_metadata,
    extract_sampled_frames,
    create_synthetic_test_video,
)
from spatial_analyzer import SpatialAnalyzer
from frequency_analyzer import (
    compute_2d_fft_spectrum,
    compute_high_frequency_energy_ratio,
    compute_radial_profile,
)
from temporal_analyzer import (
    compute_temporal_differences,
    aggregate_temporal_summary,
)

# Page configuration
st.set_page_config(
    page_title="Deepfake Forensic Analysis Prototype",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI styling for a modern, sleek academic forensic dashboard
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .disclaimer-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(217, 119, 6, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 22px;
    }
    .disclaimer-title {
        color: #fbbf24;
        font-weight: 700;
        font-size: 0.98rem;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .disclaimer-body {
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .metric-badge {
        background: #1e293b;
        border-radius: 8px;
        padding: 10px 14px;
        border: 1px solid #334155;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 600;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-title">Cross-Domain Spatial-Frequency Interaction & Temporal Consistency Network</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">B.Tech Capstone Project-I &bull; Research-Grade Exploratory Video Forensic Prototype</div>', unsafe_allow_html=True)

# Mandatory Academic and Forensic Disclaimer Banner
st.markdown("""
<div class="disclaimer-card">
    <div class="disclaimer-title">⚠️ EXPERIMENTAL FORENSIC ANALYSIS &mdash; AUTHENTICITY NOT DETERMINED</div>
    <div class="disclaimer-body">
        This prototype extracts exploratory spatial features, 2D Fast Fourier Transform (FFT) frequency signatures, 
        and inter-frame temporal continuity metrics for forensic research. 
        <strong>It does not contain a trained deepfake classification neural network</strong> and 
        <strong>does NOT classify video content as authentic or synthetic</strong>. 
        High or low metrics reflect video dynamics, compression, or lighting and cannot be used as proof of manipulation.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.caption("Configurable parameters for frame sampling and spectral metrics.")
    
    num_sampled_frames = st.slider("Sampled Frame Count", min_value=4, max_value=24, value=8, step=2,
                                   help="Evenly spaced frames extracted across the video timeline.")
    
    res_choice = st.select_slider(
        "Processing Resolution",
        options=["128x128", "256x256", "384x384"],
        value="256x256",
        help="Downsamples frames during extraction to preserve memory and accelerate 2D FFT computation."
    )
    res_map = {"128x128": (128, 128), "256x256": (256, 256), "384x384": (384, 384)}
    target_res = res_map[res_choice]
    
    hfer_cutoff = st.slider("High-Frequency Cutoff Ratio (α)", min_value=0.1, max_value=0.8, value=0.4, step=0.05,
                            help="Radius percentage beyond which spectral power is classified as high-frequency.")
    
    st.divider()
    st.subheader("🧪 Synthetic Benchmark Video")
    st.caption("No video file? Generate an in-memory synthetic sample with an optional high-frequency artifact.")
    sim_artifact = st.checkbox("Simulate high-frequency artifact", value=True)
    generate_test_sample = st.button("Generate Synthetic Test Video", type="secondary")

# Persistent state management for uploaded or synthetic video
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "is_temp_file" not in st.session_state:
    st.session_state.is_temp_file = False

# Section A: Video Input & Ingestion
st.subheader("1. Video Ingestion & Stream Validation")
col_upload, col_demo = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload MP4 Video File (Max 100 MB)",
        type=["mp4"],
        help="Upload an MP4 video clip to inspect spatial, frequency, and temporal characteristics."
    )

if generate_test_sample:
    temp_test_dir = tempfile.gettempdir()
    synth_path = os.path.join(temp_test_dir, "synthetic_capstone_demo.mp4")
    create_synthetic_test_video(
        synth_path,
        num_frames=24,
        fps=12,
        width=target_res[0],
        height=target_res[1],
        simulate_artifact=sim_artifact
    )
    st.session_state.video_path = synth_path
    st.session_state.is_temp_file = True
    st.toast("Generated synthetic test video successfully!", icon="✅")

if uploaded_file is not None:
    # Save uploaded bytes to a safe temporary file
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"uploaded_{uploaded_file.name}")
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.video_path = temp_video_path
    st.session_state.is_temp_file = True

# If a video is ready, process and display
if st.session_state.video_path and os.path.exists(st.session_state.video_path):
    active_path = st.session_state.video_path
    
    # Validation step
    is_valid, err_msg = validate_video_file(active_path, max_size_mb=100.0, max_duration_sec=120.0)
    
    if not is_valid:
        st.error(f"❌ Video Validation Failed: {err_msg}")
    else:
        # Extract metadata
        try:
            metadata = get_video_metadata(active_path)
            
            # Display metadata cards
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Resolution", f"{metadata['width']}x{metadata['height']}")
            m2.metric("Frame Rate", f"{metadata['fps']} FPS")
            m3.metric("Duration", f"{metadata['duration_sec']} s")
            m4.metric("Total Frames", metadata['total_frames'])
            m5.metric("File Size", f"{metadata['file_size_mb']} MB")
            
            with st.expander("ℹ️ Container & Stream Properties", expanded=False):
                st.json(metadata)

            # Section B: Frame Extraction
            with st.spinner("Extracting evenly spaced representative frames..."):
                sampled_frames = extract_sampled_frames(
                    active_path,
                    num_frames=num_sampled_frames,
                    target_size=target_res
                )
            
            st.divider()
            st.subheader(f"2. Extracted Frames ({len(sampled_frames)} Sampled Intervals)")
            
            # Visual gallery
            gallery_cols = st.columns(min(len(sampled_frames), 4))
            for i, f_data in enumerate(sampled_frames):
                col = gallery_cols[i % len(gallery_cols)]
                with col:
                    st.image(
                        f_data["rgb_image"],
                        caption=f"Frame #{f_data['frame_idx']} ({f_data['timestamp_sec']}s)",
                        use_container_width=True
                    )

            # Section C: Spatial Analysis & Face Detection
            st.divider()
            st.subheader("3. Spatial Analysis & Face Region Localization")
            st.caption("Inspects facial landmarks and regions-of-interest (ROI) using Haar cascades.")
            
            spatial_analyzer = SpatialAnalyzer()
            selected_frame_idx = st.selectbox(
                "Select frame for detailed spatial & frequency inspection:",
                options=range(len(sampled_frames)),
                format_func=lambda i: f"Frame {sampled_frames[i]['frame_idx']} (t = {sampled_frames[i]['timestamp_sec']}s)"
            )
            
            target_frame_data = sampled_frames[selected_frame_idx]
            spatial_results = spatial_analyzer.detect_faces(target_frame_data["rgb_image"])
            
            sp_col1, sp_col2 = st.columns([1, 1])
            with sp_col1:
                st.image(
                    spatial_results["annotated_image"],
                    caption=f"Spatial Inspection: Frame {target_frame_data['frame_idx']}",
                    use_container_width=True
                )
            with sp_col2:
                st.info(f"**Detector Status:** {spatial_results['status_message']}")
                if spatial_results["faces_found"] > 0:
                    st.write(f"Detected {spatial_results['faces_found']} Region(s) of Interest.")
                    crop_cols = st.columns(min(spatial_results["faces_found"], 3))
                    for c_idx, crop in enumerate(spatial_results["face_crops"]):
                        crop_cols[c_idx % len(crop_cols)].image(
                            crop, caption=f"ROI #{c_idx+1}", use_container_width=True
                        )
                else:
                    st.warning("No frontal face detected in this specific frame.")
                    
                st.caption(f"**Methodological Note:** {spatial_results['heuristic_disclaimer']}")

            # Section D: Frequency-Domain Analysis (2D FFT)
            st.divider()
            st.subheader("4. Frequency-Domain Analysis (2D Fast Fourier Transform)")
            
            st.markdown(r"""
            **Mathematical Formulation:**
            The 2D Discrete Fourier Transform converts spatial coordinates $(x, y)$ into spatial frequencies $(u, v)$:
            $$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
            The zero-frequency component is shifted to center $(M/2, N/2)$. To compress the large dynamic range of spectral coefficients, 
            a numerically stable log-magnitude spectrum is computed:
            $$S(u, v) = \log\left(1 + |F_{\text{shifted}}(u, v)|\right)$$
            """)
            
            fft_data = compute_2d_fft_spectrum(target_frame_data["gray_image"])
            hfer_val, high_e, total_e = compute_high_frequency_energy_ratio(
                fft_data["magnitude"], cutoff_ratio=hfer_cutoff
            )
            bin_centers, radial_profile = compute_radial_profile(fft_data["magnitude"], num_bins=40)
            
            freq_col1, freq_col2 = st.columns([1, 1])
            with freq_col1:
                fig_spec, ax_spec = plt.subplots(figsize=(5, 5))
                cax = ax_spec.imshow(fft_data["normalized_log_magnitude"], cmap="inferno", origin="lower")
                ax_spec.set_title("Centered 2D FFT Log-Magnitude Spectrum", fontsize=11, color="white")
                ax_spec.axis("off")
                fig_spec.colorbar(cax, ax=ax_spec, fraction=0.046, pad=0.04)
                fig_spec.patch.set_facecolor('#0f172a')
                st.pyplot(fig_spec)
                plt.close(fig_spec)
                
            with freq_col2:
                # Radial Profile Plot
                fig_rad, ax_rad = plt.subplots(figsize=(5, 3.5))
                ax_rad.plot(bin_centers, np.log1p(radial_profile), color="#38bdf8", linewidth=2)
                ax_rad.axvline(x=hfer_cutoff, color="#f59e0b", linestyle="--", label=f"Cutoff α={hfer_cutoff}")
                ax_rad.set_title("Radial Spectral Energy Distribution", fontsize=10, color="white")
                ax_rad.set_xlabel("Normalized Radial Frequency (r / R_max)", color="#94a3b8", fontsize=9)
                ax_rad.set_ylabel("Log Magnitude", color="#94a3b8", fontsize=9)
                ax_rad.tick_params(colors="#94a3b8")
                ax_rad.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
                ax_rad.grid(True, linestyle=":", alpha=0.3)
                fig_rad.patch.set_facecolor('#0f172a')
                ax_rad.set_facecolor('#1e293b')
                st.pyplot(fig_rad)
                plt.close(fig_rad)
                
                st.metric("High-Frequency Energy Ratio (HFER)", f"{hfer_val:.4f}")
                st.caption(
                    "**Frequency Metric Disclaimer:** HFER quantifies the proportion of spectral energy "
                    "outside the central radius. High or low values are strongly influenced by camera sensor noise, "
                    "lossy compression (e.g. H.264 block quantization), and texture complexity. "
                    "There is no universal threshold separating real from synthetic media."
                )

            # Section E: Temporal Consistency Analysis
            st.divider()
            st.subheader("5. Temporal Consistency Analysis")
            st.markdown(r"""
            **Inter-Frame Difference Formulation:**
            Consecutive sampled frames $I_t$ and $I_{t-1}$ are compared across normalized intensity values $[0, 1]$:
            $$\text{Mean Absolute Difference (MAD)} = \frac{1}{H \cdot W} \sum_{i,j} |I_t(i,j) - I_{t-1}(i,j)|$$
            $$\text{Normalized Frame Difference (NFD)} = \frac{\sum |I_t - I_{t-1}|}{\sum I_t + \sum I_{t-1} + \epsilon}$$
            """)
            
            temporal_records = compute_temporal_differences(sampled_frames)
            temporal_summary = aggregate_temporal_summary(temporal_records)
            
            if temporal_records:
                temp_df = pd.DataFrame(temporal_records)
                
                fig_temp, ax_temp = plt.subplots(figsize=(10, 3.2))
                ax_temp.plot(temp_df["timestamp_sec"], temp_df["mad"], marker="o", color="#06b6d4", label="MAD (L1 Difference)")
                ax_temp.plot(temp_df["timestamp_sec"], temp_df["rmsd"], marker="s", color="#ec4899", linestyle="--", label="RMSD")
                ax_temp.set_title("Inter-Frame Difference Over Video Timeline", fontsize=11, color="white")
                ax_temp.set_xlabel("Video Timestamp (seconds)", color="#94a3b8", fontsize=9)
                ax_temp.set_ylabel("Difference Metric [0, 1]", color="#94a3b8", fontsize=9)
                ax_temp.tick_params(colors="#94a3b8")
                ax_temp.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
                ax_temp.grid(True, linestyle=":", alpha=0.3)
                fig_temp.patch.set_facecolor('#0f172a')
                ax_temp.set_facecolor('#1e293b')
                st.pyplot(fig_temp)
                plt.close(fig_temp)
                
                t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                t_col1.metric("Mean MAD", f"{temporal_summary['mean_mad']:.4f}")
                t_col2.metric("Peak MAD", f"{temporal_summary['max_mad']:.4f}")
                t_col3.metric("Std Dev MAD", f"{temporal_summary['std_mad']:.4f}")
                t_col4.metric("Mean RMSD", f"{temporal_summary['mean_rmsd']:.4f}")
                
                st.caption(
                    "**Temporal Confounder Note:** High frame differences reflect camera movement, "
                    "lighting changes, blinking, speech articulation, or compression artifacts. "
                    "A high temporal difference score is NOT proof of a deepfake."
                )

            # Section F: Metric Summary Table & Export Options
            st.divider()
            st.subheader("6. Computed Forensic Metrics & Data Export")
            
            # Aggregate per-frame metrics
            combined_rows = []
            for i, f_data in enumerate(sampled_frames):
                gray = f_data["gray_image"]
                res = compute_2d_fft_spectrum(gray)
                hf, _, _ = compute_high_frequency_energy_ratio(res["magnitude"], cutoff_ratio=hfer_cutoff)
                mean_lum = float(np.mean(gray)) / 255.0
                
                # Matching temporal record if available
                matching_t = next((t for t in temporal_records if t["to_frame_idx"] == f_data["frame_idx"]), None)
                mad_val = matching_t["mad"] if matching_t else 0.0
                rmsd_val = matching_t["rmsd"] if matching_t else 0.0
                
                combined_rows.append({
                    "frame_index": f_data["frame_idx"],
                    "timestamp_sec": f_data["timestamp_sec"],
                    "mean_luminance": round(mean_lum, 4),
                    "hfer_score": round(hf, 4),
                    "temporal_mad": round(mad_val, 4),
                    "temporal_rmsd": round(rmsd_val, 4)
                })
                
            metrics_df = pd.DataFrame(combined_rows)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Export buttons
            csv_bytes = metrics_df.to_csv(index=False).encode('utf-8')
            json_payload = {
                "metadata": metadata,
                "temporal_summary": temporal_summary,
                "frame_metrics": combined_rows,
                "forensic_status": "Experimental forensic analysis - authenticity not determined"
            }
            json_bytes = json.dumps(json_payload, indent=2).encode('utf-8')
            
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    label="📥 Export Metrics as CSV",
                    data=csv_bytes,
                    file_name=f"forensic_metrics_{metadata['filename']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with exp_col2:
                st.download_button(
                    label="📥 Export Forensic Report as JSON",
                    data=json_bytes,
                    file_name=f"forensic_report_{metadata['filename']}.json",
                    mime="application/json",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error encountered during video processing: {str(e)}")

else:
    st.info("👈 Please upload an MP4 video or click 'Generate Synthetic Test Video' in the sidebar to begin analysis.")

# Section G: Capstone Project Methodology, Architecture & Future Roadmap
st.divider()
st.subheader("7. Capstone Research Methodology & Roadmap")

with st.expander("📖 System Methodology & Architecture Details", expanded=False):
    st.markdown("""
    ### Three-Stream Forensic Hypothesis
    Current generation and manipulation algorithms leave subtle artifacts across three domains:
    1. **Spatial Domain:** Face-swapping and reenactment leave blending boundaries, landmark inconsistencies, and resolution mismatch.
    2. **Frequency Domain:** Transposed convolution (deconvolution) and upsampling in generative pipelines introduce periodic spectral artifacts (checkerboard patterns) observable in the 2D FFT log-magnitude spectrum.
    3. **Temporal Domain:** Frame-by-frame generative synthesis often lacks temporal cross-frame constraints, leading to inter-frame flickering, jitter, and non-natural transitions.

    ### Academic Road Map for Project-II (Trainable Architecture)
    - **Benchmark Datasets:** Ingestion of standardized datasets including **FaceForensics++** (Raw, C23, C40 compression tiers) and **Celeb-DF v2**.
    - **Cross-Attention Fusion:** A multi-branch neural network fusing spatial CNN/ViT features with 2D FFT spectral tokens.
    - **Temporal Modeling:** Bidirectional GRU or Video Temporal Transformer to model long-range sequential coherence.
    - **Formal Evaluation:** Rigorous benchmarking using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC** across cross-manipulation and cross-dataset splits.
    """)
