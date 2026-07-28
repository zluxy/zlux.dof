# zluxDOF — Changelog

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
