import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Page config
st.set_page_config(page_title="Double-Slit Quantum Simulation", layout="wide")

# Constants
NX, NY = 180, 180
LX, LY = 10.0, 10.0

# Setup grid
@st.cache_data
def get_grids():
    x = np.linspace(-LX/2, LX/2, NX)
    y = np.linspace(-LY/2, LY/2, NY)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    
    kx = 2 * np.pi * np.fft.fftfreq(NX, dx)
    ky = 2 * np.pi * np.fft.fftfreq(NY, dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    
    return x, y, X, Y, K2

# Create barrier
@st.cache_data
def get_barrier():
    V = np.zeros((NY, NX))
    barrier_x = int(NX * 0.5)
    
    # Wall
    V[:, barrier_x] = 1e6
    
    # Slits
    center = NY // 2
    gap = 15
    width = 8
    
    V[center - gap - width:center - gap + width, barrier_x] = 0
    V[center + gap - width:center + gap + width, barrier_x] = 0
    
    return V

# Initial wave
def init_wave(X, Y):
    x0, y0 = -3.5, 0.0
    sigma = 0.6
    k0 = 10.0
    
    psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    psi = psi.astype(np.complex128) * np.exp(1j * k0 * X)
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    
    return psi

# Get grids and barrier
x, y, X, Y, K2 = get_grids()
V = get_barrier()

# Initialize state
if 'psi' not in st.session_state:
    st.session_state.psi = init_wave(X, Y)
    st.session_state.running = False
    st.session_state.hits = []
    st.session_state.frames = 0

# Sidebar
st.sidebar.title("Controls")

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶️"):
    st.session_state.running = True
if c2.button("⏸️"):
    st.session_state.running = False
if c3.button("🔄"):
    st.session_state.psi = init_wave(X, Y)
    st.session_state.hits = []
    st.session_state.frames = 0
    st.session_state.running = False

st.sidebar.write("---")
speed = st.sidebar.slider("Speed", 1, 15, 8)
gain = st.sidebar.slider("Brightness", 1.0, 15.0, 6.0)

st.sidebar.write("---")
st.sidebar.write(f"Frames: {st.session_state.frames}")
st.sidebar.write(f"Detections: {len(st.session_state.hits)}")

# Main display
st.title("🌊 Quantum Double-Slit Experiment")

col1, col2 = st.columns([2.5, 1])

with col1:
    st.write("**Wave Function**")
    wave_spot = st.empty()

with col2:
    st.write("**Detection Pattern**")
    hist_spot = st.empty()
    info_spot = st.empty()

# Physics step
dt = 0.004

if st.session_state.running:
    psi = st.session_state.psi
    
    for _ in range(speed):
        psi *= np.exp(-0.5j * V * dt)
        psi_k = np.fft.fft2(psi)
        psi_k *= np.exp(-0.5j * K2 * dt)
        psi = np.fft.ifft2(psi_k)
        psi *= np.exp(-0.5j * V * dt)
    
    st.session_state.psi = psi
    st.session_state.frames += 1
    
    # Detection
    screen_x = int(NX * 0.85)
    probs = np.abs(psi[:, screen_x])**2
    total = np.sum(probs)
    
    if total > 1e-5:
        probs = probs / total
        if np.random.rand() < 0.3 and np.all(np.isfinite(probs)) and abs(np.sum(probs) - 1.0) < 0.01:
            try:
                idx = np.random.choice(NY, p=probs)
                st.session_state.hits.append(y[idx])
            except:
                pass

# Render wave
intensity = np.abs(st.session_state.psi)**2
max_i = np.max(intensity) if np.max(intensity) > 0 else 1e-10
norm = np.clip(intensity * gain / max_i, 0, 1)

img = np.zeros((NY, NX, 3), dtype=np.uint8)
img[:, :, 1] = (norm * 255).astype(np.uint8)
img[:, :, 2] = (norm * 255).astype(np.uint8)

# Barrier
barrier = V > 100
img[barrier] = 255

# Convert and display
pil_image = Image.fromarray(img, 'RGB')
wave_spot.image(pil_image, use_column_width=True)

# Render histogram
if len(st.session_state.hits) > 3:
    counts, _ = np.histogram(st.session_state.hits, bins=40, range=(y.min(), y.max()))
    df = pd.DataFrame({'count': counts})
    hist_spot.bar_chart(df, height=350, color="#00FFFF")
    
    if len(st.session_state.hits) > 80:
        info_spot.success(f"✅ Pattern! ({len(st.session_state.hits)})")
    else:
        info_spot.info(f"Building... ({len(st.session_state.hits)})")
else:
    df = pd.DataFrame({'count': np.zeros(40)})
    hist_spot.bar_chart(df, height=350, color="#00FFFF")
    info_spot.warning("⏳ Waiting...")

# Loop
if st.session_state.running:
    import time
    time.sleep(0.025)
    st.rerun()
