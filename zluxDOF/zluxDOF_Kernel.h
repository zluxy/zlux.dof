// ────────────────────────────────────────────────────────────────────────────
// zluxDOF -- CUDA gather interface
//
// Plain-C boundary between the C++ renderer (zluxDOF.cpp, built by MSVC) and
// the CUDA implementation (zluxDOF_Kernel.cu, built by nvcc). Nothing in this
// header may reference AE SDK types, STL types, or CUDA types: it is compiled
// by both toolchains.
//
// Scope (stage 1): the GPU takes over GatherPass only -- the far and near
// gathers, which measured at 94% of frame time (853 ms of 908 ms at 1920x810 /
// 256 samples). All depth preprocessing stays on the CPU and is uploaded as
// finished buffers, and the CPU still owns the composite in RenderPixelImpl.
// That keeps the verification loop tight: the kernel's output is directly
// diffable against the CPU gather it replaces.
// ────────────────────────────────────────────────────────────────────────────
#ifndef ZLUXDOF_KERNEL_H
#define ZLUXDOF_KERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque device-side state (mip texture, CoC textures, LUT storage, output
// buffers). One per render; cheap to keep alive across frames of a sequence.
typedef struct ZluxGpuContext ZluxGpuContext;

// ── Vogel sample records, float mirror of the CPU VogelSample ──────────────
// Split hot/cold purely for upload compactness and because the hot record is
// exactly a float4, which is a single 16-byte vectorised load on the device.
// (Note: the same split measured as a slight LOSS on the CPU -- see the
// zluxDOF memory note -- because the CPU streams the LUT sequentially and the
// prefetcher already hides it. On the GPU the situation is different: every
// thread in a warp reads the SAME element, so a 16-byte uniform load broadcasts
// through the constant/L1 path and the narrower record genuinely pays.)
typedef struct {
	float kx;           // cos_a * fr / anamorphic_ratio
	float ky;           // sin_a * fr
	float fr;           // normalized radius on the unit disc
	float static_mask;  // baked polygon x catadioptric x softness x matte x apmap x onion
} ZluxVogelHot;

typedef struct {
	float cos_a, sin_a;   // astigmatism path only (unused in stage 1)
	float norm_x, norm_y; // rotated disc position, for per-pixel masks
	float soft_edge;      // baked softness falloff
	float sa_pos, sa_neg; // baked spherical-aberration radial profiles
	float _pad;           // keep the record 32 B / float4-aligned
} ZluxVogelCold;

// Descriptor for one Vogel LUT in the ladder (13 of them, 16..1024 samples).
typedef struct {
	int   count;
	int   offset;   // index of this LUT's first sample in the flat upload array
} ZluxVogelLutDesc;

// ── Per-frame gather parameters ────────────────────────────────────────────
// Mirrors exactly the fields of DOFSettings that GatherPass reads, plus the
// frame geometry. Kept as a flat POD so it can be passed by value to the
// kernel (it lands in constant/param space).
typedef struct {
	int   width, height;        // output resolution
	float inv_w, inv_h;         // 1/width, 1/height
	int   cache_w, cache_h;     // signed-CoC cache resolution (== output here)
	int   num_levels;           // mip pyramid depth

	// Feature gates, mirrored from DOFSettings so the kernel branches the same
	// way the CPU does. Anything the kernel cannot do is rejected on the host
	// (see zluxGpuCanRender) rather than silently ignored here.
	int   uniform_blur;         // s.no_depth
	int   energy_conserving;
	int   render_mode;          // 1 Fast, 2 Final, 3 Extreme
	int   sample_count;
	int   aperture_shape_mode;
	int   aperture_map_index;
	int   mask_angular;         // precomputed on host (steep angular mask -> tiny rotation)
	int   has_alpha;            // pyramid.has_alpha

	float focal_distance;
	float anamorphic_ratio;
	float highlight_boost;
	float bokeh_gamma;          // >0 enables the LUT weighting
	float highlight_scatter;
	float highlights_low, highlights_high, highlights_softness;
	float spherical_aberration_amount, spherical_aberration_scale;
	float vignetting, vignetting_scale;
	float catadioptric;
	float ca_strength, ca_rc, ca_gm, ca_by;
	float near_blur_factor;
} ZluxGatherParams;

// Per-pixel gather inputs that vary across the frame and are produced by the
// CPU precompute. Uploaded once per frame.
typedef struct {
	const float* signed_coc;    // cache_w * cache_h, signed CoC field
	const float* coc_disc_dist; // cache_w * cache_h, distance to nearest CoC
	                            // discontinuity; may be null
	const float* far_radius;    // per-pixel far gather radius
	const float* near_radius;   // per-pixel near gather radius (tile-dilated max)
	// Third gather: the far bleed-over probe run at FOCUSED pixels bordering far
	// content (RenderPixelImpl's ProbeFarReachWide path). Same Far kernel, its
	// own radius. 0 where the composite does not need it -- the kernel's
	// radius<=0.001 early-out makes those pixels free.
	const float* bleed_radius;
	const float* center_depth;  // per-pixel remapped depth (drives SA sign)
} ZluxGatherFields;

// Gather results, one element per output pixel. rgb is linear light; w carries
// the pass weight (Far: coverage, Near: alpha), matching PassOutput.
typedef struct {
	float* far_rgba;   // float4: rgb + coverage
	float* near_rgba;  // float4: rgb + alpha
	float* bleed_rgba; // float4: rgb + coverage (far bleed-over)
	float* mattes;     // float4: far matte, near matte, bleed matte, unused
} ZluxGatherOutputs;

// ── Lifecycle ──────────────────────────────────────────────────────────────

// Returns 1 if a usable CUDA device is present. Safe to call before create.
int zluxGpuAvailable(void);

// Human-readable description of the selected device, or NULL. Static storage.
const char* zluxGpuDeviceName(void);

// Reports whether this frame's settings are inside the kernel's supported
// subset. Stage 1 excludes the paths that need to sample an AE layer per tap
// (custom aperture texture, iris modulator) and the astigmatism/multi-tap
// path. Callers must fall back to the CPU gather when this returns 0, so the
// plugin never changes its output just because a GPU is present.
int zluxGpuCanRender(const ZluxGatherParams* params, int has_aperture_tex, int has_iris_mod, float astigmatism);

ZluxGpuContext* zluxGpuCreate(void);
void            zluxGpuDestroy(ZluxGpuContext* ctx);

// Uploads the source mip pyramid. `level_data[i]` is interleaved RGBA float,
// `level_w[i] * level_h[i] * 4` elements. Dimensions must follow successive
// halving (which matches CUDA's own mip level sizing).
int zluxGpuUploadPyramid(ZluxGpuContext* ctx,
                         const float* const* level_data,
                         const int* level_w, const int* level_h,
                         int num_levels);

// Uploads the flattened Vogel LUT ladder and the bokeh-gamma curve.
int zluxGpuUploadLuts(ZluxGpuContext* ctx,
                      const ZluxVogelHot* hot, const ZluxVogelCold* cold,
                      int total_samples,
                      const ZluxVogelLutDesc* descs, int num_luts,
                      const float* bokeh_gamma_lut257);

// Uploads the per-pixel CoC / radius / depth fields.
int zluxGpuUploadFields(ZluxGpuContext* ctx, const ZluxGatherParams* params,
                        const ZluxGatherFields* fields);

// Runs the far + near gathers and copies the results back into `out`.
// Returns 0 on success. `elapsed_ms` receives pure kernel time when non-null.
int zluxGpuGather(ZluxGpuContext* ctx, const ZluxGatherParams* params,
                  ZluxGatherOutputs* out, float* elapsed_ms);

// Last CUDA error string, or NULL if the last call succeeded.
const char* zluxGpuLastError(void);

#ifdef __cplusplus
}
#endif

#endif // ZLUXDOF_KERNEL_H
