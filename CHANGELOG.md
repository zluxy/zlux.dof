# zluxDOF — Changelog

## v2.28.0 — CUDA gather

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

Version history before v2.28.0 was not tracked in this file.

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
