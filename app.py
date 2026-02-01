import streamlit as st
import numpy as np
import time

# =========================================================
# 1. CONFIG & SETUP
# =========================================================
st.set_page_config(
    page_title="Quantum Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Performance Settings ---
# We use a smaller grid (128x128) to ensure it runs fast on Cloud CPUs
NX, NY = 128, 128 
LX, LY = 14.0, 14.0

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
    barrier_x_idx = int(nx * 0.45) # Barrier at 45% of width
    slit_width = 0.8
    slit_sep = 1.8
    
    # Draw Barrier
    V[:, barrier_x_idx] = 1e5 # Wall
    
    # Cut Slits
    # We use boolean masks to "drill" holes in the wall
    mask_slit1 = np.abs(y - slit_sep/2) < slit_width/2
    mask_slit2 = np.abs(y + slit_sep/2) < slit_width/2
    
    V[mask_slit1, barrier_x_idx] = 0
    V[mask_slit2, barrier_x_idx] = 0
    
    return V, barrier_x_idx

def init_psi(X, Y):
    """Create the initial Gaussian wave packet."""
    x0, y0 = -4.0, 0.0
    sigma = 0.6
    k0 = 5.0 # Momentum to the right
    
    # Gaussian * Plane Wave
    psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    psi = psi.astype(np.complex128)
    psi *= np.exp(1j * k0 * X)
    
    # Normalize
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
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

# =========================================================
# 4. SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("Controls")
c1, c2 = st.sidebar.columns(2)
start_btn = c1.button("▶ Play", use_container_width=True)
pause_btn = c2.button("⏸ Pause", use_container_width=True)
reset_btn = st.sidebar.button("🔄 Reset", use_container_width=True)

# Logic to handle buttons
if start_btn:
    st.session_state.running = True
if pause_btn:
    st.session_state.running = False
if reset_btn:
    st.session_state.psi = init_psi(X, Y)
    st.session_state.hits = []
    st.session_state.frame_count = 0
    st.session_state.running = False

st.sidebar.divider()
speed = st.sidebar.slider("Simulation Speed", 1, 10, 5, help="Higher = Faster wave, lower frame rate")
gain = st.sidebar.slider("Brightness Gain", 1.0, 10.0, 3.0, help="Boost visibility of faint waves")

# =========================================================
# 5. MAIN DISPLAY
# =========================================================
st.title("Double Slit Simulation")

col_left, col_right = st.columns([3, 1])
with col_left:
    plot_spot = st.empty() # Placeholder for the image
with col_right:
    status_spot = st.empty() # Placeholder for status text
    hist_spot = st.empty()   # Placeholder for histogram

# =========================================================
# 6. SIMULATION LOOP
# =========================================================
dt = 0.005

if st.session_state.running:
    
    # A. PHYSICS STEP (Split-Step Fourier)
    # We loop 'speed' times per frame to make the wave move visibly
    for _ in range(speed):
        psi = st.session_state.psi
        
        # 1. Half-step Potential
        psi *= np.exp(-0.5j * V * dt)
        
        # 2. Full-step Kinetic (FFT -> Phase -> IFFT)
        psi_k = np.fft.fft2(psi)
        psi_k *= np.exp(-0.5j * K2 * dt)
        psi = np.fft.ifft2(psi_k)
        
        # 3. Half-step Potential
        psi *= np.exp(-0.5j * V * dt)
        
        st.session_state.psi = psi
    
    st.session_state.frame_count += 1

    # B. DETECTION (The Screen)
    # The screen is at the right edge (index 85% of width)
    screen_x = int(NX * 0.85)
    
    # Get probability along that vertical line
    prob_slice = np.abs(st.session_state.psi[:, screen_x])**2
    total_p = np.sum(prob_slice)
    
    # If wave has hit the screen
    if total_p > 1e-4:
        # Normalize to get a probability distribution
        pdf = prob_slice / total_p
        
        # Randomly sample a hit based on intensity
        if np.random.rand() < 0.5: # 50% chance to detect per frame
            hit_idx = np.random.choice(len(y), p=pdf)
            st.session_state.hits.append(y[hit_idx])

# =========================================================
# 7. RENDERING (Runs every script execution)
# =========================================================

# --- Status Panel ---
status_msg = "🟢 **Running**" if st.session_state.running else "zzz **Paused**"
status_spot.markdown(f"""
{status_msg}  
Frame: `{st.session_state.frame_count}`  
Particles Detected: `{len(st.session_state.hits)}`
""")

# --- Visualizing the Wave ---
# Calculate intensity
intensity = np.abs(st.session_state.psi)**2

# Auto-Normalize: Find the max peak to scale colors
max_val = np.max(intensity)
if max_val < 1e-10: max_val = 1e-10 # Prevent divide by zero

# Create Image Buffer
img = np.zeros((NY, NX, 3), dtype=np.uint8)

# 1. Draw Wave (Cyan)
# Apply Gain and Clip to 0-1 range
norm_intensity = np.clip(intensity * gain / max_val, 0, 1)
img[..., 1] = (norm_intensity * 255).astype(np.uint8) # G
img[..., 2] = (norm_intensity * 255).astype(np.uint8) # B

# 2. Draw Barrier (White)
barrier_mask = V > 1000
img[barrier_mask] = 255

plot_spot.image(img, caption="Probability Density |ψ|²", use_container_width=True, clamp=True)

# --- Visualizing the Histogram ---
if st.session_state.hits:
    counts, _ = np.histogram(st.session_state.hits, bins=30, range=(-LY/2, LY/2))
    hist_spot.bar_chart(counts, color="#00FFFF")
else:
    hist_spot.info("Wave is traveling to screen...")

# --- Rerun Trigger ---
if st.session_state.running:
    time.sleep(0.01) # Small breath for CPU
    st.rerun()
