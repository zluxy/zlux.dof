<div align="center">

# zluxDOF

**Photorealistic Depth of Field for Adobe After Effects**

*Vogel-spiral bokeh engine · vintage glass soul · CUDA accelerated*

![Version](https://img.shields.io/badge/version-3.1.0-blue) ![Platform](https://img.shields.io/badge/platform-Windows%20x64-lightgrey) ![Host](https://img.shields.io/badge/After%20Effects-2022%2B-9999ff) ![GPU](https://img.shields.io/badge/GPU-CUDA%20%E2%80%A2%20Maxwell%2B-76b900) ![License](https://img.shields.io/badge/license-Freeware-green)

<!-- PLACEHOLDER: hero shot — best-looking frame, heavy bokeh, point lights -->
![zluxDOF hero](docs/img/hero.png)

*"Bokeh is not blur. Bokeh is character."*

</div>

---

## GPU accelerated

As of **v3.0.0** the gather runs on the GPU. On a 1920×810 frame at 256 samples
the depth-of-field pass dropped from **889 ms to 37 ms**, and the whole frame
from **952 ms to under 90 ms** — measured on a Ryzen 9 9950X3D against an
RTX 5080.

Every optical feature runs there: astigmatism, anamorphic squeeze, custom
aperture textures, the iris modulator, chromatic aberration, spherical
aberration, cat's-eye vignetting and the full highlight engine. The banner in
the Effect Controls panel shows a **GPU** / **CPU** badge with the last blur
time, so you always know which path a frame took.

No NVIDIA card? Nothing breaks — the plugin falls back to the CPU renderer that
shipped before, which is still the reference the GPU path is verified against.

**Requires:** NVIDIA GPU, Maxwell or newer (GTX 900 series and up).

➡️ **[Download the latest release](../../releases/latest)**

---

## What is it?

zluxDOF is a depth-of-field plugin that simulates how **real glass** renders defocus — not a Gaussian blur with a mask, but a physically-inspired two-pass gather over a Vogel-spiral aperture with depth-aware occlusion, HDR highlight handling and a library of real vintage-lens behaviors: swirl, soap-bubble rims, onion rings, cat's-eye clipping, chromatic fringing, field curvature and halation.

Feed it any depth map — AI-generated (Depth Anything / MiDaS), a 3D render Z-pass (EXR linear-Z auto-ranged), or a hand-painted ramp — and it turns flat footage into cinema.

<!-- PLACEHOLDER: before/after split of the same frame -->
![Before / after](docs/img/before_after.png)

---

## Highlights

### 🔭 Real lens presets
26 one-click lenses, each modeled on the documented optical character of the actual glass — and every preset **bakes into the sliders**, so it's a starting point you can edit, not a black box:

| Preset | Character |
|---|---|
| **Helios 44-2 58mm f/2** | edge-rimmed discs, onion rings, cat's-eye, the Soviet swirl |
| **Cyclop 85mm f/1.5** | night-vision block, no iris blades — the wildest swirl |
| **Mir-1V 37mm** | warm enveloping vintage softness |
| **Industar-61 L/Z** | the famous 6-point star bokeh |
| **MTO-500 Mirror** | catadioptric donut discs |
| **Trioplan 100mm f/2.8** | the soap-bubble king |
| **Petzval 85 (Lomography)** | razor centre, tunnel swirl |
| **Canon 50mm f/0.95 Dream** | legendary glow + magenta fringe |
| **Lensbaby Sweet 50 / Burnside 35** | sweet-spot focus, radial streaks / swirl |
| **Wide Macro Rig (Psyche)** | the psychedelic franken-rig look |
| **Sigma 105mm f/1.4 Art** | the modern clinical reference |
| **LOMO 35-NAP 2x / Laowa Nanomorph 1.5x** | true anamorphic ovals & fringe |
| **Swirl-o-Tron 58 (Vortex)** | astigmatism cranked into a hypnotic full-frame vortex |
| **Starburst Zoom 35** | radial zoom-burst streaks + glowing field-curvature edges |
| **Anamorphic Comet 2x** | 2× squeeze sheared into leaning comet-streak bokeh |
| **Sony 135mm STF (Apodized)** | graded-ND element — discs with no edge at all, the smoothest bokeh ever built |
| **Cooke S4/i (Cine Warm)** | the Cooke Look: warm, undercorrected, highlights bloom instead of ringing |
| **CCTV 25mm f/1.4 (C-mount)** | hard hexagonal iris, violent swirl and fringing — the cheap-lens look |
| **Rodenstock Imagon** | soft-focus head: the disc dissolves into a halo |
| **Reflex 1000mm (Donut)** | the *bad* mirror lens — swirling, ringed donuts |
| **Angenieux 25-250 (Vintage Zoom)** | heavy onion rings from a zoom's element count, cool coating cast |
| **Tilt-Shift Miniature** | pure field-curvature edge blur, tight sweet spot — the toy-town rig |

<!-- PLACEHOLDER: 3-4 preset comparison grid on the same shot -->
![Lens presets](docs/img/presets_grid.png)

### 🕳 Iris construction
- **Bokeh Definition** — how hard a bokeh disc's edge is allowed to be. The
  gather pre-averages the source before it integrates the disc, and that
  pre-average *is* the edge gradient; this slider dials it away. 0% is the
  creamy v3.0 render, 100% is hard-edged "bokeh balls". Raise **Sample Quality**
  (now up to 2048) with it.
- Circular / polygonal (3–16 blades, curvature from star to circle) / notched irises
- **80 real aperture maps** in a clickable thumbnail picker (scratched, dusty, shaped pupils)
- **Procedural onion rings** — aspheric machining grooves inside every disc
- Custom iris from any layer + iris texture overlay (dust, coating wear)
- Anamorphic squeeze 0–4x with industry aspect presets
- Live **Bokeh Preview** right in the effect panel

<!-- PLACEHOLDER: aperture picker UI + a few resulting bokeh discs -->
![Iris & aperture maps](docs/img/iris_picker.png)

### 🌈 Lens character
- Spherical aberration (soap-bubble rims ↔ velvet glow cores)
- Astigmatism up to 2x elongation — sagittal (zoom-burst) or tangential (swirl)
- **Field curvature + sweet spot** — depth-independent edge blur (works even without a depth map)
- Optical vignetting (cat's-eye) with falloff control
- Catadioptric (mirror-lens) donut ring
- Chromatic aberration: classic, neutral, or green/magenta cinema fringe — computed **per-sample** so fringes scale with each point's true defocus

### ✨ Highlights that survive defocus
- Soft-knee threshold selection, bokeh gamma (crisp specular cores), additive highlight scatter
- **HDR clipping recovery** — reconstructs blown-highlight energy on 8/16-bit sources
- Bloom / halation post-pass
- Film grain with photometric distribution and focus-map placement

<!-- PLACEHOLDER: night shot with fairy lights, crisp discs + bloom -->
![Highlights & bloom](docs/img/highlights.png)

### 🧠 Depth intelligence
- Per-frame **auto-range** for non-normalized depth (EXR linear-Z just works)
- Depth-guided smoothing (de-staircases AI maps without eating object edges)
- Occlusion-boundary snap — clean silhouettes between blurred near/far fields
- Edge Protect — rescues thin details (wires, hair) that AI depth misclassifies
- Black/white point, gamma, invert, channel select, click-to-focus eyedropper

### 🖥 Workflow
- **Render Mode:** Fast (≈3x scrubbing) / Final / Extreme (hero-shot quality, full-res gather)
- **11 preview modes** including CoC Heat Map (the exact blur field the engine uses), **Focus Peaking** (aim focus like on a camera monitor) and Edge Diagnostics
- Fully multithreaded CPU renderer (scales to 32 threads), Smart Render, 8/16/32-bit

<!-- PLACEHOLDER: preview modes strip (heat map / peaking / diagnostics) -->
![Preview modes](docs/img/preview_modes.png)

---

## Installation

1. Download `zluxDOF.aex` from [Releases](../../releases).
2. Copy it anywhere under your MediaCore plugins folder, e.g.:
   ```
   C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\zlux\zluxDOF.aex
   ```
3. Restart After Effects. The effect appears under **Effect → zluxia → zluxDOF**.

That's it — **one file**. The banner, the aperture-map library and the picker are embedded inside the plugin.

> **Requirements:** Windows x64, an AVX2-capable CPU (any Intel/AMD from ~2014 onward), After Effects 2022 or newer (earlier versions will likely work but are untested).

## Quick start

1. Apply **zluxDOF** to your footage.
2. Set **Depth Map Layer** to your depth pass.
   - AI depth maps (Depth Anything, MiDaS): they are *disparity* (white = near) — enable **Invert Depth**.
   - EXR Z-passes: values outside 0–1 are auto-normalized per frame. Just plug in.
3. Switch **Preview Mode → Focus Peaking**, drag **Focus Distance** until the green overlay sits on your subject, switch back to **Rendered**.
4. Dial **Blur Amount**, pick a **Lens Preset**, season to taste.

**Pro tips**
- Work in **32 bpc** and raise **Highlight Clipping Recovery** — point lights bloom into rich, bright-edged discs like on real glass.
- Iris texture (rings, dust, star shapes) only imprints on lights **smaller than the blur disc** — that's physics, the same on a real lens.
- Use **Render Mode → Fast** while scrubbing, **Extreme** for the money shot.

---

## License

zluxDOF is **free for personal and commercial use**. It is *not* open source; redistribution, resale and modification are not permitted — see [LICENSE.txt](LICENSE.txt).

## Credits

Design & engineering: **zlux** · zluxia
Adobe and After Effects are trademarks of Adobe Inc.

<div align="center">

*If zluxDOF made your shot prettier, a ⭐ on this repo is the best thank-you.*

</div>
