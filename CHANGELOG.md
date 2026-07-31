# zluxDOF — Changelog

## v3.1.0 — Bokeh Definition

Bokeh discs get their edge back. This release is about one diagnosis: the gather
was pre-blurring the source far more than it needed to, and the reason it did
was not a sampling problem at all.

### The footprint floor was the bokeh edge

The gather reads colour from a MIP pyramid at a level sized so adjacent Vogel
taps have no gaps between them. On top of that sizing sat a **floor**: the
footprint could never be smaller than **35% of the blur radius**. At a 48 px
disc that is a 17 px pre-blur — and since the output for a point light is
`aperture mask ⊛ pyramid kernel`, that pre-blur *is* the disc's edge gradient.
The old code's claim that "the edge comes from the aperture mask, not the source
mip, so discs stay round" was half true: the mask sets the shape, but the mip
convolves it soft.

The floor existed to suppress a firefly speckle, and the note in the source
recorded the damning symptom — **more samples made it worse**. That is the
fingerprint of a *weight* problem, not a *sampling* one: `Bokeh Brightness
Boost` and `Bokeh Gamma` raise a tap's weight by a power of its luma, so one
blown texel takes over the weighted average; and a higher tap count drives the
spacing mip finer, which feeds the weighting sharper texels. The colour never
needed blurring.

So the two concerns are now separated. The **weight** luma is read at the old,
deliberately steady 35% LOD; the **colour** is read at the sharp one. Weights
vary smoothly across the disc while the disc keeps a hard edge.

Measured on a 1280×720 point-light frame, 1024 samples, boost 1.0:

| Bokeh Definition | disc edge (10–90%) | peak brightness | GPU gather |
|---|---:|---:|---:|
| 0% (= v3.0) | 14 px | 35.0 | 13.3 ms |
| 65% (default) | 8 px | 42.2 | 16.6 ms |
| 100% | **7 px** | **44.2** | 18.1 ms |

The halo that used to bleed outside the disc reads 0 at 100%. Speckle in a
high-frequency noise field, same frame — the weight-LOD split measured on its
own (`ZLUX_NOWLOD=1` in profile builds turns it off):

| boost | split off | split on |
|---|---:|---:|
| 1.0 | 8.83 | **3.17** |
| 3.0 | 12.36 | **3.02** |

With the split, speckle no longer depends on the boost at all — which is the
whole point.

### New control

**Aperture > Bokeh Definition** (0–100%, default 65). 0 reproduces the v3.0
render exactly. Raise **Sample Quality** alongside it: once the floor is gone
the Vogel inter-sample spacing is the only limit on how crisp a disc can get.

### Also in this release

- **Highlight Mode now defaults to Preservative.** Additive was the default only
  because it was the historical behaviour. It accumulates specular taps into an
  un-normalised bucket layered on top, so a bright point can exceed the source's
  own peak and clip to flat white — and the crisper gather in this release makes
  that land harder, because the energy is no longer smeared over a soft edge.
  Preservative folds the same specular emphasis into the gather weights and
  renormalises, so the disc keeps its colour and falloff. Existing projects are
  untouched; this changes only what a freshly applied effect starts with.
- **4×4 tent pyramid downsample** (was a 2×2 box). The gather reads nearly all
  of its defocused colour out of this pyramid, so the pyramid is the inside of
  every bokeh disc; a box filter leaks enough aliasing per level to read as
  blocky texels on a still and as crawling texture in motion. Deliberately not a
  Karis average — that fixes fireflies by discarding the highlight energy that
  makes bokeh worth having.
- **Pyramid edges clamp instead of mirror.** Mirroring answered off-frame taps
  with a flipped copy of the plate, so a bright point near the border was
  gathered twice and printed a phantom bokeh across the frame edge.
- **Sample Quality ceiling raised to 2048** (was 1024), with 1536/2048 rungs
  added to the Vogel ladder. Extreme mode's footprint-derived tap cap was
  clipping against the old ceiling on large bokeh.

CPU and CUDA paths agree to a mean of 0.002/255 (max 3/255) on the reference
frame.

### Seven new lens presets

Chosen to reach optical territory the existing nineteen did not, rather than to
lengthen the menu — each one was checked against the current lineup first.

- **Sony 135mm STF (Apodized)** — the apodization lens. A radially graded ND
  element inside the barrel fades the disc to nothing at its rim, so there is no
  bokeh *edge* at all. The softest previous entry (Mir-1V) spends its softness
  alongside heavy swirl and fringing; here it is the entire point.
- **Cooke S4/i (Cine Warm)** — "the Cooke Look": rounded 8-blade cine prime left
  deliberately undercorrected so highlights bloom into their discs instead of
  ringing. The counterpoint to the Sigma Art, which is the same cleanliness
  taken the other way.
- **CCTV 25mm f/1.4 (C-mount)** — the cheap security lens on mirrorless. Hard
  hexagonal iris, violent swirl and vignetting, uncorrected fringing. The
  Cyclop's swirl comes from a bladeless *circular* aperture and stays soft; this
  one is all hard edges.
- **Rodenstock Imagon (Soft Focus)** — the classic portrait soft-focus head.
  Undercorrected spherical aberration taken to the end of its range, where the
  disc stops reading as a disc and becomes a halo.
- **Reflex 1000mm (Donut)** — the *bad* mirror lens, against the clean MTO-500.
  Same central obstruction, plus the coarse machining texture and real
  astigmatism that give mirror bokeh its reputation.
- **Angenieux 25-250 (Vintage Zoom)** — every ground surface in a zoom prints
  another ring, so the onion structure runs far heavier than the Helios's
  machining marks, over a moderate swirl and cool vintage coating cast.
- **Tilt-Shift Miniature** — the one use of Field Curvature the existing rigs do
  not cover: they all pair it with heavy astigmatism to get streaks. Pure
  depth-independent edge blur with a tight sweet spot reads as "miniature"
  rather than "broken lens".

As with every preset since v2.9.4 these are optics-only — they write the iris
and aberration sliders and leave your highlight grade alone.

### Stability and correctness

A pass over the concurrency edges around the CUDA path and the AE contract.
None of these change a rendered pixel; all of them were reachable in ordinary
use, mostly on machines with many cores.

- **A busy moment no longer costs the session its GPU.** The device path
  latches OFF for the rest of the session when a CUDA call fails, which is
  right for a broken device and wrong for a transient one — and two transient
  conditions were being reported the same way: "all four pooled result sets are
  in flight" (routine Multi-Frame Rendering contention: a set is held for the
  whole composite, so five concurrent frames exhaust the pool) and "not enough
  free VRAM this instant" (a number owned mostly by other effects). Either one
  silently dropped every later frame to the CPU gather — roughly 10x slower,
  with only the panel badge to say so. They now report `ZLUX_GPU_BUSY`: this
  frame goes to the CPU, the next one retries the GPU.
- **The result-set pool no longer frees buffers out from under a live reader.**
  Raising the frame size reallocated *every* pinned set under the device mutex —
  but a set handed to an earlier frame is being read by that frame's compositor
  right then, outside the mutex. Adjustment layers switch resolution constantly,
  which is exactly how that got hit. Each set now carries its own capacity and
  grows only in the thread that just claimed it.
- **`in_use` is atomic.** It was claimed under the device mutex and released
  without it. The realistic failure was a release the claim loop never observed,
  which looked like permanent pool exhaustion.
- **Tearing down the GPU context is no longer permanent.** `Shutdown()` nulled
  a slot whose initialiser was a function-local static, so the context could
  never be recreated: every render after a GlobalSetdown/GlobalSetup pair (AE
  does reload effects mid-session) fell to the CPU. The context is now created
  lazily under the device mutex, and teardown clears the failure latch.
- **A failed aperture-texture upload is no longer ignored.** The return value
  was discarded, so a failure left the *previous* frame's iris texture bound and
  the gather rendered the wrong aperture — breaking the one guarantee the GPU
  path exists to keep, that having a GPU never changes output.
- **The aperture-map library cache is no longer a shared mutable global.** One
  cache slot, refilled in place at the top of `RenderCore`, meant a second
  effect instance on a different map would `clear()` the vector while another
  render thread was reading it in the gather — a use-after-free, and a reload on
  every frame for as long as both instances rendered. Loads now produce
  immutable maps handed out as `shared_ptr`, so each frame holds its own for its
  duration and two instances stop fighting over one slot.
- **`PF_OutFlag_PIX_INDEPENDENT` removed.** It asserts that an output pixel does
  not depend on its neighbours, which lets AE leave unneeded rows filled with
  garbage during field rendering — and a defocus gather reads exactly those
  rows. It only ever bought a field-rendering optimisation this effect cannot
  use.
- **Bit depth is read from AE rather than inferred.** `rowbytes >= width *
  sizeof(PF_PixelFloat)` cannot separate 8 bpc from 32 bpc on very narrow
  worlds, because row padding pushes a 4-px-wide 8 bpc row to exactly the float
  stride — thumbnail-sized renders were read as float out of a quarter-sized
  buffer. Smart render now uses `PF_SmartRenderInput::bitdepth`, which also
  makes every checked-out layer's depth exact.
- **Vogel LUT device buffers grow with the ladder** instead of being sized by
  the first frame. The ladder is fixed today, which is precisely why the
  first-frame sizing looked safe; the memcpys always used the current frame's
  count.

## v3.0.0 — CUDA gather

**9.0× faster per frame.** The depth-of-field gather now runs on the GPU.

### Measured

1920×810, 256 samples, Final quality, blur 40 — Ryzen 9 9950X3D (16C/32T) + RTX 5080,
steady state (second frame onward, so one-time CUDA context creation is excluded):

| stage | CPU | CUDA |
|---|---:|---:|
| depth preprocessing | 62.5 ms | 65.4 ms |
| **gather** | **889.2 ms** | **36.7 ms** |
| composite | — | 4.0 ms |
| **total per frame** | **952 ms** | **106 ms** |

The gather kernel itself runs in **10.7 ms** (10.5–14.6 ms across runs; it climbs
to ~50 ms when something else is using the GPU), i.e. ~83× the CPU gather it
replaces. The remaining 26 ms of the gather stage is host-side: radius precompute,
uploads, and the readback of four `float4` buffers.

Reproduce with the standalone harness:

```
cd debug && build_png.bat
ZLUX_REPEAT=6 ZLUX_SAMPLES=256 ZLUX_RMODE=2 ./dof_png.exe bench_photo.raw bench_depth.raw out.raw 1920 810 1 0.5 40
```

`ZLUX_NOGPU=1` forces the CPU gather and produces the reference image.
`ZLUX_REPEAT=N` matters: a single frame measures cold start (CUDA context
creation alone is 70–105 ms) and understates the result by roughly half.

### Preservative Highlights (new)

`Highlights / Bokeh Shaping > Highlight Mode` selects how Highlight Scatter
combines with the gather.

- **Additive** — the previous behaviour. Specular taps accumulate into an
  un-normalised bucket layered on top of the gather, so a bright point can push
  the result past the source's own peak and clip to flat white.
- **Preservative** — the specular emphasis is folded into the gather *weights*
  and renormalised. The result stays a convex combination of the sampled
  colours, so it can never exceed the brightest sample: exposure and dynamic
  range are conserved and the highlight is concentrated by redistributing energy
  rather than inventing it.

Measured on a synthetic scene of 45 bright point sources at blur 55:

| mode | mean | clipped pixels | peak |
|---|---:|---:|---:|
| scatter off | 19.34 | 0 | 232 |
| Additive | 19.88 | **398** | **255** (clipped) |
| Preservative | **21.13** | **0** | 247 |

Preservative reads *brighter* than additive while clipping nothing — the
highlights are more present, and the frame never blows out.

This closes the last feature gap against DOF PRO v2.0.

### Also in this release

- Astigmatism, the anisotropic multi-tap, the custom aperture texture and the
  iris modulator layer all run on the GPU now. Previously each of them silently
  forced the whole frame onto the CPU gather, which is 10-20x slower: astigmatism
  alone went from 1409 ms to 82 ms per frame.
- The CoC-discontinuity distance field moved to a jump-flooding transform on the
  GPU (13.7 ms -> 0.1 ms). The CPU version was two sequential chamfer sweeps that
  could not be threaded at all.
- The per-pixel gather radii are computed on the device, removing three
  full-frame uploads per frame.
- GPU buffers are grow-only and the CUDA context is released on unload. The old
  free/realloc on every resolution change churned ~270 MB per switch, including
  pinned host memory, which could make After Effects report "GPU out of memory"
  for effects that were not even ours (seen on adjustment layers).
- The CUDA runtime is delay-loaded and ships beside the plug-in. A machine with
  no CUDA runtime now gets a working CPU plug-in instead of one Windows refuses
  to map, which previously removed zluxDOF from the effect list entirely.
- The panel banner shows a **GPU** / **CPU** badge with the last blur time, so a
  fallback to the CPU path is visible rather than merely slow.

### Output parity

Diffed against the CPU reference, 8-bit output, 6 220 800 bytes:

- 7 977 bytes differ (**0.128 %**)
- 99.96 % of those differ by exactly 1/255
- maximum deviation 4/255, on 3 pixels
- nothing above 4/255

That is the expected `double`→`float` delta and is visually indistinguishable.

### What is new

- `zluxDOF/zluxDOF_Kernel.cu` — CUDA port of `GatherPass`. Far, near and the far
  bleed-over probe all run in one launch, since they share the centre texel, the
  centre CoC and the whole per-pixel setup.
- `zluxDOF/zluxDOF_Kernel.h` — plain-C boundary between MSVC and nvcc. No AE SDK,
  STL or CUDA types cross it.
- `zlux_gpu` bridge in `zluxDOF.cpp` — materialises the per-pixel gather radii
  using the same expressions the CPU path uses, flattens the pyramid and the
  Vogel ladder, and runs the launch.
- `ZLUX_REPEAT` in the `dof_png` harness, for steady-state measurement.

### Design notes

**The mip pyramid is a `cudaMipmappedArray` sampled with hardware trilinear
filtering.** On the CPU, `SampleMipTrilinear` is two bilinear taps — eight
scattered RGBA loads plus the blend; on the GPU it is a single `tex2DLod`
serviced by the texture units. This is the main win, and it is why the SDK's
`PrGPU/KernelSupport` wrapper is deliberately not used: it cannot express a
mipmapped texture object.

Pyramid level sizes match CUDA's own mip sizing for free — the CPU builds levels
by successive `max(1, w/2)`, and repeated integer halving is identical to
`floor(n / 2^k)`. `cudaAddressModeMirror` reproduces `MirrorCoordSafe`, and the
CoC textures use point filtering with clamp to match the CPU's truncate-and-clamp
indexing.

**Occupancy was tuned against the spill line, not against occupancy.** The kernel
is texture-latency-bound, so spilling the tap loop's live state to local memory
costs far more than extra resident warps buy:

| launch bounds | registers | spills |
|---|---:|---|
| `(256, 4)` | 64 | 56 B stores / 36 B loads |
| `(256, 2)` | 88 | **none** |

**MFR safety.** The plugin sets `PF_OutFlag2_SUPPORTS_THREADED_RENDERING`, so AE
renders several frames concurrently. The CUDA context owns a single set of device
buffers and costs ~70–105 ms to create, so neither a per-thread context nor a
per-frame one is viable; device access is serialised with a mutex instead. The
GPU work is ~37 ms per frame, and the other ~70 ms (preprocessing, composite)
still parallelises fully across MFR threads.

**cudart links statically.** The `.aex` has no `cudart64_*.dll` import, so there
is nothing to ship alongside it and no conflict with the CUDA runtime After
Effects loads for itself.

### Fallback

`zluxGpuCanRender` rejects anything the kernel does not implement and the frame
silently falls back to the CPU gather, so output never changes just because a GPU
is present. Currently rejected:

- custom aperture texture (`aperture_shape_mode == 4`) and the iris modulator
  layer — both need to sample an AE layer per tap, so the mask cannot be baked
- astigmatism, which drives the anisotropic multi-tap path

Set `ZLUX_NOGPU=1` to force the CPU path everywhere.

### GPU support

The kernel ships as a fat binary with native SASS for every architecture CUDA
13.x can target, plus `compute_120` PTX so future GPUs JIT instead of dropping to
the CPU:

| arch | generation | cards |
|---|---|---|
| sm_75 | Turing | RTX 20xx, GTX 16xx |
| sm_80 | Ampere | A100 |
| sm_86 | Ampere | RTX 30xx |
| sm_89 | Ada | RTX 40xx |
| sm_90 | Hopper | H100 |
| sm_120 | Blackwell | RTX 50xx |

Verified with `cuobjdump --list-elf`. The measured numbers above are from an
RTX 5080; other cards will differ.

`cudart` is linked statically, so the `.aex` stays a single self-contained file
with no `cudart64_*.dll` to ship and no conflict with the CUDA runtime After
Effects loads for itself.

### Known limits

- **Pascal and older get the CPU path.** CUDA 13.x cannot target below sm_75, so
  GTX 10xx users see no change. Covering them means moving the build back to
  CUDA 12.9 (which supports sm_50 up); the kernel source needs no changes.
- Non-NVIDIA GPUs get the CPU path.
- Depth preprocessing is now the bottleneck at 65 ms of the 106 ms frame (62 %).
  Moving those filters to the GPU is the next step.

---

## Earlier

Version history before v3.0.0 was not tracked in this file.

---

# Rejected: CPU Vogel-LUT compaction

Recorded so it is not retried. `VogelSample` was 11 doubles (88 B) and `VogelLUT`
holds a fixed `samples[1024]`, so a 1024-tap pixel streamed ~90 KB per output
pixel — well over the 48 KB L1d. Splitting the record hot/cold and narrowing to
float was implemented and benchmarked interleaved against the original:

| variant | mean gather |
|---|---:|
| original (88 B, double) | 853 ms |
| hot/cold split, 4 floats (16 B) | 868 ms (**1.7 % slower**) |
| hot/cold split, 4 doubles (32 B) | 857 ms (within noise) |

Reverted. The LUT is streamed sequentially and reused by every pixel, so it stays
resident in L2 and the prefetcher hides it entirely; and float storage adds a
`cvtss2sd` per field per tap, which a compute-bound loop cannot afford.

The useful part was the diagnosis: the gather is bound by **random access** — the
mip fetches and the scattered CoC reads — which is precisely what the CUDA port
addresses and what no CPU data-layout change could.

---

## Version-field overflow (fixed in v3.0.0)

`MINOR_VERSION` had been above 15 since v2.16, which silently broke AE's version
handshake. AE packs the version into fixed bit fields (`PF_Vers_*` in
`AE_Effect.h`), and `subvers` is only 4 bits:

| field | bits | max |
|---|---|---:|
| build | 0-8 | 511 |
| stage | 9-10 | 3 |
| bugfix | 11-14 | 15 |
| **subvers (minor)** | **15-18** | **15** |
| vers (major) | 19-21 | 7 |

The two encodings then disagreed about the overflow. `PF_VERSION()` masks each
field, so a minor of 28 became `28 & 0xF == 12`. `ZLUX_PIPL_VERSION` is plain
arithmetic, so `28 * 32768` carried into the major field instead. Result:

```
After Effects: effect "zluxDOF" has version mismatch.
Code version is 2.12 and PiPL version is 3.12. (160001)
```

The header's promise that the two "can never drift apart" only held for
`MINOR_VERSION <= 15`.

Fixed by moving to 3.0.0 (major bump, minor reset) and by adding `static_assert`s
in `zluxDOF.cpp` that check every field against its width and, most importantly,
assert `ZLUX_PIPL_VERSION == PF_VERSION(...)`. A future overflow now fails the
build instead of shipping.

Also fixed alongside it:

- The PiPL `CustomBuild` step listed only `zluxDOFPiPL.r` as an input, so bumping
  `zluxDOF_Version.h` did not regenerate the resource on an incremental build.
  `zluxDOF_Version.h` is now in `AdditionalInputs` for all four configurations.
- `build.bat` installed to `Common\Plug-ins\7.0\MediaCore\zlux` while the plugin
  was also present in `Support Files\Plug-ins\zlux`. AE scans both recursively,
  so two builds of the same effect could be registered at once. It now installs
  to one location only.
