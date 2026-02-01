# Quantum Double-Slit Simulator 🌊

An interactive, research-grade simulation of the **2D Time-Dependent Schrödinger Equation (TDSE)**. This application visualizes wave-particle duality by propagating a quantum wave packet through a double-slit barrier and simulating individual particle detections on a screen.

---

## 🔬 Physics and Mathematical Framework

The simulation is governed by the non-relativistic Schrödinger Equation:

$$i\hbar \frac{\partial}{\partial t} \psi(\mathbf{r}, t) = \hat{H} \psi(\mathbf{r}, t)$$

Where the Hamiltonian operator $\hat{H}$ is defined as:

$$\hat{H} = -\frac{\hbar^2}{2m} \nabla^2 + V(\mathbf{r})$$

### 1. The Wave Packet
We initialize the system as a **Gaussian Wave Packet** with a specific momentum $k_0$ in the $x$-direction:

$$\psi(x, y, 0) = \exp \left( -\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2} \right) e^{ik_0 x}$$



### 2. Numerical Integration (Split-Step Fourier Method)
To solve the TDSE efficiently in real-time, the code uses the **Split-Step Fourier Method (SSFM)**. This method "splits" the potential and kinetic energy operators. For a small time step $\Delta t$, the evolution is:

$$\psi(t + \Delta t) \approx e^{-i \frac{V}{2} \Delta t} e^{-i \hat{T} \Delta t} e^{-i \frac{V}{2} \Delta t} \psi(t)$$

* **Potential Step:** Multiplied in the position domain.
* **Kinetic Step:** Multiplied in the frequency (momentum) domain using Fast Fourier Transforms (FFT), where $\nabla^2$ becomes $-k^2$:
    $$\hat{T} \psi \xrightarrow{\mathcal{F}} \frac{\hbar^2 k^2}{2m} \tilde{\psi}$$



### 3. The Born Rule and Measurement
The simulation bridges the gap between waves and particles using the **Born Rule**. The probability density is given by:

$$P(x, y) = |\psi(x, y)|^2$$.



In the app, we take a "slice" of this density at the screen position. We use **Monte Carlo sampling** to "detect" individual particles, showing how individual random hits eventually form a deterministic interference pattern.

---

## 📊 Technical Specifications

| Parameter | Value / Method | Description |
| :--- | :--- | :--- |
| **Equation** | TDSE | 2D Time-Dependent Schrödinger Equation |
| **Numerical Solver** | Split-Step Fourier | Unitary evolution preserving the norm of $\psi$ |
| **Grid Size** | $128 \times 128$ | Optimized for low-latency browser rendering |
| **Time Step ($\Delta t$)** | $0.005$ | Balancing stability and simulation speed |
| **Boundary** | Dirichlet | $\psi = 0$ at the edges of the simulation box |
| **Measurement** | Monte Carlo | Probabilistic sampling based on $\|\psi\|^2$ |
| **Visualization** | RGB Normalization | Real-time auto-gain for wave visibility |

---

## 🛠️ Installation & Deployment

### Local Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install streamlit numpy
