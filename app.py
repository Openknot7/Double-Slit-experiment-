import streamlit as st
import numpy as np
import time

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Quantum Double-Slit Simulation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CACHED PHYSICS ENGINE
# =========================================================
# We cache these functions so Streamlit doesn't rebuild the 
# math grid on every single animation frame (huge performance boost).

@st.cache_data
def initialize_grid(nx, ny, lx, ly):
    """Generates the spatial grid and momentum (Fourier) space grid."""
    # Spatial Grid
    x = np.linspace(-lx/2, lx/2, nx)
    y = np.linspace(-ly/2, ly/2, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    
    # Momentum Grid (for FFT)
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2  # Kinetic energy operator factor (k^2)
    
    return x, y, X, Y, K2, dx, dy

@st.cache_data
def create_potential(ny, nx, x, y, barrier_pos, slit_sep, slit_width, v0):
    """Creates the potential energy map V(x,y) with double slits."""
    V = np.zeros((ny, nx))
    ix = np.argmin(np.abs(x - barrier_pos))
    
    # Define Slit Geometry
    slit_half_sep = slit_sep / 2
    slit_half_width = slit_width / 2

    for j in range(ny):
        # Check if we are inside a slit opening
        in_slit_1 = abs(y[j] - slit_half_sep) < slit_half_width
        in_slit_2 = abs(y[j] + slit_half_sep) < slit_half_width
        
        # If not in a slit, place the barrier
        if not (in_slit_1 or in_slit_2):
            V[j, ix] = v0
            # Thicken barrier to 3 pixels to prevent quantum tunneling artifacts
            if ix > 0: V[j, ix-1] = v0
            if ix < nx-1: V[j, ix+1] = v0
            
    return V

def init_wavefunction(X, Y, x0, y0, sigma, k0):
    """Initializes a Gaussian wave packet moving to the right."""
    # Gaussian envelope
    psi = np.exp(-((X - x0)**2)/(2*sigma**2) - ((Y - y0)**2)/(2*sigma**2))
    psi = psi.astype(np.complex128)
    # Momentum kick (exp(ikx))
    psi *= np.exp(1j * k0 * X)
    # Normalization
    norm = np.sqrt(np.sum(np.abs(psi)**2))
    return psi / norm

# =========================================================
# 3. SIMULATION SETUP
# =========================================================
# Simulation Constants
Nx, Ny = 200, 140  # Grid size (lowered slightly for cloud performance)
Lx, Ly = 14.0, 9.0 # Physical dimensions
V0 = 1e5           # Barrier height (essentially infinite)

# Initialize Grids (Cached)
x, y, X, Y, K2, dx, dy = initialize_grid(Nx, Ny, Lx, Ly)
# Initialize Potential (Cached)
V = create_potential(Ny, Nx, x, y, barrier_pos=-Lx/2 + 0.45*Lx, 
                     slit_sep=1.2, slit_width=0.5, v0=V0)

# =========================================================
# 4. SESSION STATE
# =========================================================
if "psi" not in st.session_state:
    st.session_state.psi = init_wavefunction(X, Y, -4.5, 0.0, 0.8, 6.0)
    st.session_state.running = False
    st.session_state.hits = [] 
    st.session_state.t = 0.0

# =========================================================
# 5. SIDEBAR UI
# =========================================================
st.sidebar.title("Quantum Controls")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶ Start", use_container_width=True):
        st.session_state.running = True
with col2:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.running = False

if st.sidebar.button("🔄 Reset Simulation", use_container_width=True):
    st.session_state.psi = init_wavefunction(X, Y, -4.5, 0.0, 0.8, 6.0)
    st.session_state.hits = []
    st.session_state.running = False
    st.session_state.t = 0.0

st.sidebar.divider()

# Physics Parameters
st.sidebar.markdown("**Simulation Settings**")
dt = st.sidebar.slider("Time Step (dt)", 0.001, 0.010, 0.005, format="%.3f")
steps_per_frame = st.sidebar.slider("Speed (Steps/Frame)", 1, 20, 8)

st.sidebar.info(
    "**Tip:** Increase 'Speed' to make the wave move faster. "
    "Decrease it to see the interference pattern form slowly."
)

# =========================================================
# 6. MAIN DISPLAY LAYOUT
# =========================================================
st.title("Double-Slit Experiment")
st.markdown("""
This is a **live solution of the Schrödinger Equation**. 
* **Left:** The probability density $|\\psi(x,y)|^2$ (cyan) interacting with the barrier (white).
* **Right:** Detections on the back screen, building up the interference pattern particle by particle.
""")

col_wave, col_hist = st.columns([3, 1])

with col_wave:
    wave_container = st.empty()

with col_hist:
    hist_container = st.empty()

# =========================================================
# 7. PHYSICS LOOP
# =========================================================
def split_step_propagate(psi, V, K2, dt):
    """
    Split-Step Fourier Method:
    psi(t+dt) = exp(-iVdt/2) * IFFT[ exp(-iK²dt/2) * FFT[ exp(-iVdt/2) * psi(t) ] ]
    """
    # 1. Half-step Potential
    psi *= np.exp(-0.5j * V * dt)
    
    # 2. Full-step Kinetic (in Fourier space)
    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-0.5j * K2 * dt) 
    psi = np.fft.ifft2(psi_k)
    
    # 3. Half-step Potential
    psi *= np.exp(-0.5j * V * dt)
    return psi

if st.session_state.running:
    # Run physics multiple times per frame for smoothness
    for _ in range(steps_per_frame):
        st.session_state.psi = split_step_propagate(st.session_state.psi, V, K2, dt)
        st.session_state.t += dt

    # --- MEASUREMENT LOGIC (BORN RULE) ---
    # Define the "screen" location at 85% of the grid width
    screen_x_idx = int(Nx * 0.85)
    
    # Extract probability density slice at the screen
    prob_slice = np.abs(st.session_state.psi[:, screen_x_idx])**2
    total_prob = np.sum(prob_slice)

    # Threshold: Only detect if the wave has actually reached the screen
    if total_prob > 1e-3:
        # Normalize to create a valid Probability Distribution Function (PDF)
        pdf = prob_slice / total_prob
        
        # Monte Carlo Sampling:
        # Randomly trigger a detection based on intensity
        if np.random.rand() < 0.35: # 35% detection efficiency per frame
            detected_y_idx = np.random.choice(len(y), p=pdf)
            st.session_state.hits.append(y[detected_y_idx])

# =========================================================
# 8. VISUALIZATION
# =========================================================

# --- A. Render Wavefunction Image ---
prob_density = np.abs(st.session_state.psi)**2

# Normalize for display brightness
# (Multiply by constant to make faint waves visible)
vis_img = prob_density / (np.max(prob_density) + 1e-16)
vis_img = np.clip(vis_img * 2.5, 0, 1) 

# Create RGB Array (Height, Width, 3)
img_rgb = np.zeros((Ny, Nx, 3), dtype=np.uint8)

# Map Wave to Cyan (Green + Blue)
img_rgb[..., 1] = (vis_img * 255).astype(np.uint8) 
img_rgb[..., 2] = (vis_img * 255).astype(np.uint8)

# Overlay Barrier (White)
barrier_mask = V > 0
img_rgb[barrier_mask, 0] = 255
img_rgb[barrier_mask, 1] = 255
img_rgb[barrier_mask, 2] = 255

wave_container.image(img_rgb, caption="Wavefunction Intensity", use_container_width=True, clamp=True)

# --- B. Render Detection Histogram ---
if len(st.session_state.hits) > 0:
    counts, bin_edges = np.histogram(st.session_state.hits, bins=40, range=(-Ly/2, Ly/2))
    hist_container.bar_chart(counts, color="#00FFFF") # Cyan color to match wave
    hist_container.markdown(f"**Total Detections:** {len(st.session_state.hits)}")
else:
    hist_container.info("Waiting for particle impacts...")

# =========================================================
# 9. ANIMATION LOOP TRIGGER
# =========================================================
if st.session_state.running:
    # Slight delay to allow UI to render and prevent CPU 100% lock
    time.sleep(0.01)
    st.rerun()
