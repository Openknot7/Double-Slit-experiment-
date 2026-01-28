# app.py
"""
Double-Slit Experiment — Streamlit app
Physically-correct scalar-wave summation, optimized and deployable.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------- Page config ----------------
st.set_page_config(page_title="Double-Slit Simulation", layout="wide")
st.title("Double-Slit Experiment — Interactive Simulation")
st.caption("Scalar-wave summation using Huygens–Fresnel principle (exact path lengths)")

# ---------------- Sidebar / Controls ----------------
with st.sidebar:
    st.header("Parameters")
    wavelength_nm = st.slider("Wavelength (nm)", 350.0, 800.0, 532.0, step=1.0)
    slit_sep_mm = st.slider("Slit center-to-center separation (mm)", 0.05, 2.0, 0.5, step=0.01)
    slit_width_mm = st.slider("Single-slit width (mm)", 0.01, 1.0, 0.1, step=0.01)
    distance_m = st.slider("Distance screen from slits (m)", 0.05, 5.0, 1.0, step=0.05)

    st.markdown("---")
    st.header("Screen / Numerics")
    screen_half_width_mm = st.slider("Half-width of visible screen (mm)", 5.0, 200.0, 50.0, step=1.0)
    resolution = st.selectbox("1D screen resolution (points)", [400, 800, 1200, 2000], index=1)
    samples_per_slit = st.slider("Samples per slit (accuracy vs speed)", 4, 300, 80, step=2)

    st.markdown("---")
    st.header("Visualization")
    show_2d = st.checkbox("Show 2D intensity heatmap (slower)", value=False)
    if show_2d:
        # provide safer defaults for 2D
        two_d_nx = st.selectbox("2D X resolution", [180, 240, 300], index=0)
        two_d_ny = int(two_d_nx * 0.6)
    else:
        two_d_nx = two_d_ny = None

    colormap = st.selectbox("Colormap", ["inferno", "viridis", "plasma", "magma", "gray"], index=0)
    intensity_scale = st.selectbox("Intensity scale", ["linear", "log"], index=0)

    st.markdown("---")
    st.markdown("**Hints**: Increase `samples per slit` to better resolve diffraction envelope; large `resolution` or 2D may be slow on Streamlit Cloud.")

# ---------------- Unit conversions and derived params ----------------
wavelength = float(wavelength_nm) * 1e-9       # m
slit_sep = float(slit_sep_mm) * 1e-3           # m (center-to-center)
slit_width = float(slit_width_mm) * 1e-3       # m
screen_half = float(screen_half_width_mm) * 1e-3  # m
D = float(distance_m)                           # m
k = 2.0 * np.pi / wavelength

# Avoid silly inputs
if slit_sep == 0:
    st.warning("Slit separation is zero — pattern will be single-slit. Increase separation for two-slit interference.")

# Approx analytical fringe spacing for reference
approx_fringe_spacing = None
try:
    approx_fringe_spacing = wavelength * D / slit_sep
except Exception:
    approx_fringe_spacing = np.nan

st.markdown(f"**Approx fringe spacing (λD/d):** `{(approx_fringe_spacing*1e3):.3f} mm`")

# ---------------- Numerical safety caps ----------------
# Limit computational burden automatically
MAX_1D_RES = 3000
MAX_SAMPLES = 500
if resolution > MAX_1D_RES:
    st.info(f"1D resolution capped to {MAX_1D_RES} to avoid excessive compute.")
    resolution = MAX_1D_RES
if samples_per_slit > MAX_SAMPLES:
    st.info(f"Samples per slit capped to {MAX_SAMPLES}.")
    samples_per_slit = MAX_SAMPLES

# ---------------- Cached compute functions ----------------
# Note: caching functions accept only python scalars (and small ints) so cache hit works.
@st.cache_data(show_spinner=False)
def compute_1d_intensity(resolution, screen_half, slit_sep, slit_width, D, wavelength, samples_per_slit):
    """
    Compute 1D intensity on screen along x (centered at x=0).
    Returns x (m) and normalized intensity (unitless).
    This function is deliberately cache-friendly (only scalar arguments).
    """
    k = 2.0 * np.pi / wavelength
    x = np.linspace(-screen_half, screen_half, int(resolution))
    field = np.zeros_like(x, dtype=np.complex128)

    # slit centers (vertical positions y)
    centers = np.array([-slit_sep / 2.0, slit_sep / 2.0], dtype=float)

    # sample each slit along its vertical extent (y direction)
    for yc in centers:
        ys = np.linspace(yc - slit_width / 2.0, yc + slit_width / 2.0, int(samples_per_slit))
        # compute distances r for all samples to all x points:
        # shape: (S, Nx) -> sqrt(x^2 + ys^2 + D^2) with broadcasting
        # x[None, :] shape (1, Nx); ys[:, None] shape (S, 1)
        r = np.sqrt(x[None, :]**2 + ys[:, None]**2 + D**2)
        # avoid exact zero
        r = np.maximum(r, 1e-12)
        # sum contributions from these point sources
        field += np.sum(np.exp(1j * k * r) / r, axis=0)

    intensity = np.abs(field)**2
    # normalize safely
    maxI = intensity.max()
    if maxI <= 0 or not np.isfinite(maxI):
        norm = 1.0
    else:
        norm = maxI
    intensity = intensity / norm
    return x, intensity

@st.cache_data(show_spinner=False)
def compute_2d_intensity(nx, ny, screen_half, slit_sep, slit_width, D, wavelength, samples_per_slit):
    """
    Compute 2D intensity on a rectangular screen grid (X,Y).
    Returns X_vec (x positions), Y_vec (y positions), intensity (Ny x Nx).
    This uses a loop over sample points along slit (S iterations), each vectorized over the screen.
    """
    k = 2.0 * np.pi / wavelength
    X_vec = np.linspace(-screen_half, screen_half, int(nx))
    Y_vec = np.linspace(-screen_half * 0.6, screen_half * 0.6, int(ny))
    X, Y = np.meshgrid(X_vec, Y_vec)  # shape (Ny, Nx)
    field = np.zeros_like(X, dtype=np.complex128)

    centers = np.array([-slit_sep / 2.0, slit_sep / 2.0], dtype=float)
    S = int(max(1, samples_per_slit // 2))  # keep 2D sampling smaller for performance

    for yc in centers:
        ys = np.linspace(yc - slit_width / 2.0, yc + slit_width / 2.0, S)
        # iterate per sample to control memory usage (S x Ny x Nx will be large if vectorized fully)
        for y_s in ys:
            # distance from (y_s, 0) on slit plane to each pixel (X, Y, D)
            r = np.sqrt(X**2 + (Y - y_s)**2 + D**2)
            r = np.maximum(r, 1e-12)
            field += np.exp(1j * k * r) / r

    intensity = np.abs(field)**2
    maxI = intensity.max()
    if maxI <= 0 or not np.isfinite(maxI):
        norm = 1.0
    else:
        norm = maxI
    intensity = intensity / norm
    return X_vec, Y_vec, intensity

# ---------------- Compute 1D ----------------
with st.spinner("Computing 1D intensity..."):
    t0 = time.time()
    x, intensity_1d = compute_1d_intensity(
        resolution, screen_half, slit_sep, slit_width, D, wavelength, samples_per_slit
    )
    t1 = time.time()

st.success(f"1D computed in {t1 - t0:.2f} s")

# ---------------- Plot 1D ----------------
fig1, ax1 = plt.subplots(figsize=(9, 3.6))
if intensity_scale == "log":
    # convert to dB; add small floor
    vals = 10.0 * np.log10(intensity_1d + 1e-12)
    ax1.plot(x * 1e3, vals)
    ax1.set_ylabel("Intensity (dB, normalized)")
else:
    ax1.plot(x * 1e3, intensity_1d)
    ax1.set_ylabel("Normalized intensity")

ax1.set_xlabel("Screen position x (mm)")
ax1.set_title("Interference pattern — 1D slice (center row)")
ax1.grid(True)
st.pyplot(fig1)
plt.close(fig1)

# Display key parameters
st.markdown(
    f"**Parameters:** wavelength = {wavelength_nm:.1f} nm | slit separation = {slit_sep_mm:.3f} mm "
    f"| slit width = {slit_width_mm:.3f} mm | distance = {D:.3f} m"
)

st.markdown(
    "Model: each slit is sampled by point sources; field contributions are summed with exact "
    "path-lengths (no small-angle approximation). Intensity is |sum(amplitudes)|^2 and normalized."
)

# ---------------- Optional 2D ----------------
if show_2d:
    st.markdown("### 2D Intensity (computationally heavier)")
    # further safety checks
    max_pixels = 300 * 200
    if two_d_nx * two_d_ny > max_pixels:
        st.info(f"2D grid too large, reducing resolution for safety.")
        two_d_nx = min(two_d_nx, 300)
        two_d_ny = int(two_d_nx * 0.6)

    with st.spinner("Computing 2D intensity (this may take some time)..."):
        t0 = time.time()
        X_vec, Y_vec, intensity_2d = compute_2d_intensity(
            two_d_nx, two_d_ny, screen_half, slit_sep, slit_width, D, wavelength, samples_per_slit
        )
        t1 = time.time()
    st.success(f"2D computed in {t1 - t0:.2f} s")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    extent = [X_vec[0] * 1e3, X_vec[-1] * 1e3, Y_vec[0] * 1e3, Y_vec[-1] * 1e3]
    if intensity_scale == "log":
        im = ax2.imshow(10.0 * np.log10(intensity_2d + 1e-12), extent=extent, origin="lower", aspect="auto", cmap=colormap)
        ax2.set_title("2D Intensity (dB, normalized)")
    else:
        im = ax2.imshow(intensity_2d, extent=extent, origin="lower", aspect="auto", cmap=colormap)
        ax2.set_title("2D Intensity (normalized)")

    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    cbar = fig2.colorbar(im, ax=ax2, label="Intensity")
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")
st.markdown("### Performance notes")
st.write(
    "- `samples per slit` controls accuracy of the diffraction envelope (more = better but slower).\n"
    "- 1D calculation is vectorized and cached; 2D uses smaller sampling and loops per slit-sample to manage memory.\n"
    "- If Streamlit Cloud times out, reduce `resolution` and `samples per slit`, or disable 2D."
)

st.markdown("---")
st.markdown("If you want, I can: add Fraunhofer/Fresnel toggles, single-photon buildup animation, analytical overlays, or GPU acceleration (Numba/CuPy).")
