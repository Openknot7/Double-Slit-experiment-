# Double-Slit-experiment-

# Double-Slit Experiment — Interactive Simulation (Streamlit)

An interactive **Python + Streamlit** simulation of the classic **Young’s Double-Slit Experiment**, built using **physically correct scalar wave summation**.

This app visualizes **interference and diffraction** by explicitly summing wave amplitudes from each slit, rather than using pre-derived intensity formulas.

---

## 🔬 Physics Background

The double-slit experiment demonstrates the **wave nature of light**.

When coherent light passes through two narrow slits:
- Each slit acts as a secondary source
- Waves interfere on a distant screen
- Bright and dark fringes appear due to phase differences

This experiment is foundational to:
- Classical wave optics
- Quantum mechanics (wave–particle duality)

---

## 🧠 Physical Model Used

This simulation follows the **Huygens–Fresnel principle**:

- Light is treated as a monochromatic scalar wave
- Each slit is divided into many point sources
- The complex wave amplitude is summed at each screen point
- Intensity is computed as the squared magnitude of the total field

No small-angle or far-field approximation is forced.

---

## 📐 Mathematical Formulation

### 1. Wave from a point source
\[
\psi(r) = \frac{1}{r} e^{i k r}
\]
where  
- \( r \) = distance from slit point to screen point  
- \( k = \frac{2\pi}{\lambda} \)  
- \( \lambda \) = wavelength  

---

### 2. Exact path length
For a screen point \( (x, y) \) and slit sample at \( y_s \):
\[
r = \sqrt{x^2 + (y - y_s)^2 + D^2}
\]

---

### 3. Total field
\[
\Psi(x,y) = \sum_{\text{slits}} \sum_{\text{samples}} \frac{e^{i k r}}{r}
\]

---

### 4. Intensity
\[
I(x,y) = |\Psi(x,y)|^2
\]

Intensity is normalized for visualization.

---

## 📊 What You See

- **1D mode**: intensity along the screen centerline  
- **2D mode**: full intensity distribution on the screen  
- Emergent **interference fringes**
- Natural **single-slit diffraction envelope**

---

## 📏 Analytical Reference (for comparison)

Approximate fringe spacing (Fraunhofer limit):
\[
\Delta x \approx \frac{\lambda D}{d}
\]

This formula is **not used** in the simulation — only shown for reference.

---

## ⚙️ Parameters Explained

| Parameter | Meaning |
|---------|--------|
| Wavelength | Light wavelength (nm) |
| Slit separation | Center-to-center distance |
| Slit width | Width of each slit |
| Screen distance | Distance from slits to screen |
| Samples per slit | Numerical accuracy |
| Resolution | Number of screen points |

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
