import streamlit as st
import numpy as np
import time
from PIL import Image

# =========================================================
# 1. CONFIG & SETUP
# =========================================================
st.set_page_config(
    page_title="Quantum Double-Slit Experiment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Performance Settings ---
NX, NY = 256, 256  # Increased resolution for smoother visualization
LX, LY = 16.0, 16.0

# =========================================================
# 2. MATH ENGINE (Cached for Speed)
# =========================================================
@st.cache_data
def get_grid(nx, ny, lx, ly):
    """Generate the coordinate systems."""
    x = np.linspace(-lx/2, lx/2, nx)
    y = np.linspace(-ly/2, ly/2, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    
    # Momentum space for FFT
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    return x, y, X, Y, K2, dx, dy

@st.cache_data
def get_potential(nx, ny, x, y):
    """Generate the double slit barrier."""
    V = np.zeros((ny, nx))
    
    # Barrier Parameters
    barrier_x_idx = int(nx * 0.45)  # Barrier at 45% of width
    barrier_thickness = 2  # Make barrier thicker
    slit_width = 1.2
    slit_sep = 2.5
    
    # Draw Barrier (with thickness)
    for i in range(barrier_thickness):
        idx = barrier_x_idx + i
        if idx < nx:
            V[:, idx] = 1e5
    
    # Cut Slits
    y_arr = y.reshape(-1, 1)
    mask_slit1 = np.abs(y_arr - slit_sep/2) < slit_width/2
    mask_slit2 = np.abs(y_arr + slit_sep/2) < slit_width/2
    
    for i in range(barrier_thickness):
        idx = barrier_x_idx + i
        if idx < nx:
            V[mask_slit1.flatten(), idx] = 0
            V[mask_slit2.flatten(), idx] = 0
    
    return V, barrier_x_idx

def init_psi(X, Y):
    """Create the initial Gaussian wave packet."""
    x0, y0 = -5.5, 0.0
    sigma = 0.8
    k0 = 6.0  # Momentum to the right
    
    # Gaussian * Plane Wave
    psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    psi = psi.astype(np.complex128)
    psi *= np.exp(1j * k0 * X)
    
    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * (LX/NX) * (LY/NY))
    psi /= norm
    return psi

# Load Physics Objects
x, y, X, Y, K2, dx, dy = get_grid(NX, NY, LX, LY)
V, barrier_idx = get_potential(NX, NY, x, y)

# =========================================================
# 3. APP STATE MANAGEMENT
# =========================================================
if 'psi' not in st.session_state:
    st.session_state.psi = init_psi(X, Y)
    st.session_state.running = False
    st.session_state.hits = []
    st.session_state.frame_count = 0
    st.session_state.total_time = 0.0

# =========================================================
# 4. SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("⚛️ Quantum Controls")
st.sidebar.markdown("---")

col1, col2, col3 = st.sidebar.columns(3)
start_btn = col1.button("▶️", use_container_width=True, help="Start Simulation")
pause_btn = col2.button("⏸️", use_container_width=True, help="Pause Simulation")
reset_btn = col3.button("🔄", use_container_width=True, help="Reset Simulation")

# Logic to handle buttons
if start_btn:
    st.session_state.running = True
if pause_btn:
    st.session_state.running = False
if reset_btn:
    st.session_state.psi = init_psi(X, Y)
    st.session_state.hits = []
    st.session_state.frame_count = 0
    st.session_state.total_time = 0.0
    st.session_state.running = False

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameters")
steps_per_frame = st.sidebar.slider("Time Steps/Frame", 1, 15, 8, 
                                     help="More steps = faster wave propagation")
gain = st.sidebar.slider("Brightness", 1.0, 15.0, 5.0, 
                         help="Amplify wave visibility")
detection_rate = st.sidebar.slider("Detection Rate", 0.1, 1.0, 0.3, step=0.1,
                                   help="Probability of particle detection per frame")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistics")
stat_placeholder = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.info("""
**About This Simulation**

This demonstrates the quantum double-slit experiment using the Schrödinger equation. 

- **Cyan wave**: Probability amplitude
- **White barrier**: Double slit
- **Right panel**: Detected particles
- **Pattern**: Wave interference creates bands
""")

# =========================================================
# 5. MAIN DISPLAY
# =========================================================
st.title("🌊 Quantum Double-Slit Experiment")
st.markdown("### Real-time simulation of wave-particle duality")
st.markdown("---")

col_left, col_right = st.columns([2.5, 1])

with col_left:
    wave_container = st.container()
    with wave_container:
        st.markdown("**Wave Function Evolution**")
        plot_spot = st.empty()

with col_right:
    detection_container = st.container()
    with detection_container:
        st.markdown("**Particle Detection Pattern**")
        hist_spot = st.empty()
        info_spot = st.empty()

# =========================================================
# 6. SIMULATION LOOP
# =========================================================
dt = 0.004  # Smaller timestep for stability

def evolve_wavefunction(psi, V, K2, dt, steps):
    """Evolve the wave function using split-step Fourier method."""
    for _ in range(steps):
        # Half-step potential
        psi *= np.exp(-0.5j * V * dt)
        
        # Full-step kinetic in Fourier space
        psi_k = np.fft.fft2(psi)
        psi_k *= np.exp(-0.5j * K2 * dt)
        psi = np.fft.ifft2(psi_k)
        
        # Half-step potential
        psi *= np.exp(-0.5j * V * dt)
    
    return psi

def create_visualization(psi, V, gain, barrier_idx):
    """Create the wave visualization image."""
    intensity = np.abs(psi)**2
    
    # Normalize intensity
    max_val = np.max(intensity)
    if max_val < 1e-12:
        max_val = 1e-12
    
    # Apply gain and normalize to 0-1
    norm_intensity = np.clip(intensity * gain / max_val, 0, 1)
    
    # Create RGB image
    img = np.zeros((NY, NX, 3), dtype=np.uint8)
    
    # Cyan color for wave (gradient based on intensity)
    img[..., 1] = (norm_intensity * 255).astype(np.uint8)  # Green
    img[..., 2] = (norm_intensity * 255).astype(np.uint8)  # Blue
    
    # Add slight red tint to high-intensity areas for better visibility
    high_intensity = norm_intensity > 0.5
    img[high_intensity, 0] = ((norm_intensity[high_intensity] - 0.5) * 2 * 100).astype(np.uint8)
    
    # Draw barrier in white
    barrier_mask = V > 1000
    img[barrier_mask] = [255, 255, 255]
    
    # Add detection screen indicator (faint vertical line)
    screen_x = int(NX * 0.82)
    img[:, screen_x, :] = img[:, screen_x, :] // 2 + 80
    
    return img

if st.session_state.running:
    # Evolve wave function
    st.session_state.psi = evolve_wavefunction(
        st.session_state.psi, V, K2, dt, steps_per_frame
    )
    
    st.session_state.frame_count += 1
    st.session_state.total_time += dt * steps_per_frame
    
    # Detection at screen
    screen_x = int(NX * 0.82)
    prob_slice = np.abs(st.session_state.psi[:, screen_x])**2
    total_p = np.sum(prob_slice)
    
    # If wave has reached the screen
    if total_p > 1e-6:
        # Normalize to probability distribution
        pdf = prob_slice / total_p
        
        # Ensure PDF is valid (sums to 1, no NaNs or negatives)
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        pdf = np.abs(pdf)  # Ensure all positive
        pdf_sum = np.sum(pdf)
        
        if pdf_sum > 0:
            pdf = pdf / pdf_sum  # Renormalize
            
            # Detect particles based on probability
            if np.random.rand() < detection_rate:
                try:
                    hit_idx = np.random.choice(len(y), p=pdf)
                    st.session_state.hits.append(y[hit_idx])
                except ValueError:
                    # If still invalid, skip this detection
                    pass

# =========================================================
# 7. RENDERING
# =========================================================

# Update statistics
stat_placeholder.metric("Simulation Time", f"{st.session_state.total_time:.2f} a.u.")
stat_placeholder.metric("Frames", st.session_state.frame_count)
stat_placeholder.metric("Detections", len(st.session_state.hits))

# Render wave function
img = create_visualization(st.session_state.psi, V, gain, barrier_idx)
plot_spot.image(img, use_container_width=True, channels="RGB")

# Render detection histogram
if len(st.session_state.hits) > 0:
    # Create histogram
    counts, bin_edges = np.histogram(
        st.session_state.hits, 
        bins=40, 
        range=(-LY/2, LY/2)
    )
    
    # Display as bar chart
    hist_spot.bar_chart(counts, color="#00FFFF", height=400)
    
    # Show interference pattern info
    if len(st.session_state.hits) > 50:
        info_spot.success(f"✅ Interference pattern emerging! ({len(st.session_state.hits)} particles)")
    else:
        info_spot.info(f"Collecting data... ({len(st.session_state.hits)} particles)")
else:
    hist_spot.markdown("```\n" + "│\n" * 15 + "└" + "─" * 20 + "\n```")
    info_spot.warning("⏳ Waiting for wave to reach detector...")

# =========================================================
# 8. AUTO-RERUN LOGIC
# =========================================================
if st.session_state.running:
    time.sleep(0.03)  # Control frame rate (~30 FPS)
    st.rerun()
