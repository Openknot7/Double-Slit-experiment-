import streamlit as st
import numpy as np
import time

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Double-Slit Quantum Simulation", layout="wide")
st.title("Double-Slit Experiment — Quantum-Accurate Live Simulation")

st.markdown("""
This simulation solves the **2D time-dependent Schrödinger equation**.
Each dot on the screen represents a **single detection event** sampled from
the wavefunction probability density |ψ|².
""")

# =========================================================
# GRID (dimensionless natural units: ħ = m = 1)
# =========================================================
Nx, Ny = 240, 160
Lx, Ly = 14.0, 9.0

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# =========================================================
# POTENTIAL — DOUBLE SLIT
# =========================================================
V = np.zeros((Ny, Nx))
V0 = 1e5

slit_sep = 1.2
slit_width = 0.5
barrier_x = -Lx/2 + 0.45 * Lx
ix = np.argmin(np.abs(x - barrier_x))

for j in range(Ny):
    # Infinite barrier except slits
    if not (
        abs(y[j] - slit_sep/2) < slit_width/2 or
        abs(y[j] + slit_sep/2) < slit_width/2
    ):
        V[j, ix] = V0

# =========================================================
# INITIAL WAVE PACKET
# =========================================================
x0, y0 = -4.5, 0.0
sigma_x, sigma_y = 0.8, 1.0
k0 = 6.0

# Create complex wavefunction from start
psi0 = np.empty((Ny, Nx), dtype=np.complex128)
psi0.real = np.exp(-((X - x0)**2)/(2*sigma_x**2) - ((Y - y0)**2)/(2*sigma_y**2))
psi0.imag = 0.0
psi0 *= np.exp(1j * k0 * X)

# Normalize
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx * dy)

# =========================================================
# FFT PRECOMPUTATION
# =========================================================
kx = 2*np.pi*np.fft.fftfreq(Nx, dx)
ky = 2*np.pi*np.fft.fftfreq(Ny, dy)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

dt = 0.005  # stable time step

def propagate(psi):
    """One time-step propagation (split-step FFT method)."""
    psi *= np.exp(-1j * V * dt / 2)
    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-1j * K2 * dt / 2)
    psi = np.fft.ifft2(psi_k)
    psi *= np.exp(-1j * V * dt / 2)
    return psi

# =========================================================
# SESSION STATE
# =========================================================
if "psi" not in st.session_state:
    st.session_state.psi = psi0.copy()
    st.session_state.running = False
    st.session_state.hits = []

# =========================================================
# CONTROLS
# =========================================================
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("▶ Start"):
        st.session_state.running = True
with c2:
    if st.button("⏸ Pause"):
        st.session_state.running = False
with c3:
    if st.button("🔄 Reset"):
        st.session_state.psi = psi0.copy()
        st.session_state.hits = []
        st.session_state.running = False

# =========================================================
# TIME EVOLUTION
# =========================================================
if st.session_state.running:
    for _ in range(3):
        st.session_state.psi = propagate(st.session_state.psi)

# =========================================================
# DETECTION (BORN RULE)
# =========================================================
prob = np.abs(st.session_state.psi)**2
screen_ix = int(Nx * 0.9)  # screen at right edge

p_slice = prob[:, screen_ix]
p_slice /= (p_slice.sum() + 1e-15)

# Randomly add a detection hit
if st.session_state.running and np.random.rand() < 0.35:
    iy = np.random.choice(len(y), p=p_slice)
    st.session_state.hits.append(y[iy])

# =========================================================
# VISUALIZATION
# =========================================================
left, right = st.columns([2, 1])

with left:
    img = prob / prob.max()
    st.image(
        (img * 255).astype(np.uint8),
        clamp=True,
        caption="Probability density |ψ(x,y)|²",
        use_column_width=True
    )

with right:
    st.subheader("Screen detections")
    if len(st.session_state.hits) > 5:
        hist, _ = np.histogram(st.session_state.hits, bins=50)
        hist = hist / hist.max()
        st.bar_chart(hist)

# =========================================================
# STREAMLIT ANIMATION CLOCK
# =========================================================
if st.session_state.running:
    time.sleep(0.05)
    st.experimental_rerun()
