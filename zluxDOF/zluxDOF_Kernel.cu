// ────────────────────────────────────────────────────────────────────────────
// zluxDOF -- CUDA gather kernel
//
// Port of GatherPass (zluxDOF.cpp) to CUDA. The CPU gather measured at 94% of
// frame time, and the profiling that led here showed it is bound by RANDOM
// ACCESS -- the mip-pyramid fetches and the scattered signed-CoC reads -- not
// by arithmetic or by sequential bandwidth. (A CPU-side attempt to shrink the
// Vogel LUT was measured and reverted; it made no difference, which is what
// pointed at the access pattern.) That diagnosis dictates the two design
// choices that matter here:
//
//   1. The mip pyramid lives in a cudaMipmappedArray sampled through a texture
//      object with hardware trilinear filtering. On the CPU, SampleMipTrilinear
//      is two bilinear taps = 8 scattered RGBA loads plus the blend arithmetic.
//      Here it is ONE tex2DLod instruction serviced by the texture units, with
//      their own cache and address hardware. This is the single largest win and
//      it is the reason not to use the SDK's PrGPU KernelWrapper, which has no
//      way to express a mipmapped texture object.
//
//   2. Latency is hidden by occupancy rather than by an out-of-order window.
//      Thousands of taps are in flight per SM, so the scattered CoC reads that
//      stall the CPU cost almost nothing here.
//
// Numerical note: the CPU gather runs in double (PF_FpLong). This kernel runs
// in float, deliberately -- consumer NVIDIA parts execute FP64 at 1/64 rate, so
// a literal double transcription would be slower than the CPU it replaces. The
// accumulators are the only place precision could bite, and they stay in float
// only because the gather is a bounded convex average of values in [0,1] with
// well-conditioned weights. The verification harness diffs against the CPU
// output so this stays a measured claim rather than an assumed one.
// ────────────────────────────────────────────────────────────────────────────

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "zluxDOF_Kernel.h"

// ── Error plumbing ─────────────────────────────────────────────────────────
static char g_last_error[512] = {0};
static bool g_has_error = false;

static void ClearErr() { g_last_error[0] = '\0'; g_has_error = false; }

static bool Fail(cudaError_t e, const char* what)
{
	if (e == cudaSuccess) return false;
	std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", what, cudaGetErrorString(e));
	g_has_error = true;
	return true;
}

#define CU_TRY(call, what) do { if (Fail((call), (what))) return 1; } while (0)

extern "C" const char* zluxGpuLastError(void)
{
	return g_has_error ? g_last_error : nullptr;
}

// ── Constants mirrored from the CPU renderer ───────────────────────────────
#define ZLUX_COC_CLAMP   1.0f
#define ZLUX_EPS         1e-6f
#define ZLUX_MAX_LEVELS  8

__constant__ float c_bokehGamma[257];
// Per-bokeh rotation buckets (cos, sin), mirroring kBokehRotLUT on the host.
__constant__ float2 c_bokehRot[128];

// ── Device state ───────────────────────────────────────────────────────────
struct ZluxGpuContext {
	// Source pyramid as a hardware-filtered mipmapped texture.
	cudaMipmappedArray_t mipArray   = nullptr;
	cudaTextureObject_t  texPyramid = 0;
	int   pyr_w = 0, pyr_h = 0, pyr_levels = 0;

	// Per-pixel fields. Point-sampled textures rather than raw pointers: the
	// access pattern is a scattered 2D gather, which is exactly what the
	// texture cache's 2D locality is built for.
	cudaTextureObject_t texCoc      = 0;   // signed CoC
	cudaTextureObject_t texDiscDist = 0;   // distance to CoC discontinuity
	// Aperture layers folded to luma. Wrap addressing + linear filtering, so the
	// device does the seamless tiling the CPU did with a manual floor().
	cudaArray_t         arrApTex[2]  = {nullptr, nullptr};
	cudaTextureObject_t texApTex[2]  = {0, 0};
	int                 apTexW[2]    = {0, 0};
	int                 apTexH[2]    = {0, 0};
	cudaArray_t         arrCoc      = nullptr;
	cudaArray_t         arrDiscDist = nullptr;
	bool                hasDiscDist = false;

	// Straight linear buffers -- read once per pixel, no reuse to exploit.
	float* d_farRadius    = nullptr;
	float* d_nearRadius   = nullptr;
	float* d_bleedRadius  = nullptr;
	float* d_centerDepth  = nullptr;

	// Vogel ladder, flattened. Uniform across a warp, so these go through
	// __ldg and broadcast out of L1.
	ZluxVogelHot*     d_hot   = nullptr;
	ZluxVogelCold*    d_cold  = nullptr;
	ZluxVogelLutDesc* d_descs = nullptr;
	int               num_luts = 0;

	// Outputs.
	float4* d_farRGBA   = nullptr;
	float4* d_nearRGBA  = nullptr;
	float4* d_bleedRGBA = nullptr;
	float4* d_mattes    = nullptr;

	// Page-locked staging for the readback. This was the single most expensive
	// item in the frame after the port: 4 x float4 x pixels is ~95 MB, and
	// copying it into pageable std::vectors cost more than the kernel itself --
	// the driver has to stage pageable memory through an internal pinned buffer,
	// and the vectors were additionally zero-filled by resize() every frame.
	// Allocated once per resolution and handed straight to the compositor.
	// Jump-flooding scratch for the CoC-discontinuity distance field.
	// Ping-pong buffers of nearest-seed coordinates, packed as short2.
	short2* d_seedA = nullptr;
	short2* d_seedB = nullptr;
	float*  d_disc  = nullptr;
	float*  d_tileMinCoc = nullptr;
	size_t  tile_cap = 0;

	float4* h_farRGBA   = nullptr;
	float4* h_nearRGBA  = nullptr;
	float4* h_bleedRGBA = nullptr;
	float4* h_mattes    = nullptr;

	// Capacity in PIXELS for the linear buffers, not a width/height pair.
	// Linear buffers are indexed y*width+x, so an allocation made for a larger
	// frame serves a smaller one unchanged -- they only ever grow. That matters
	// a lot on adjustment layers, where AE renders the same effect at several
	// resolutions and the old free/realloc-on-every-change churned ~270 MB per
	// switch, including pinned host memory (a kernel-level page-pinning
	// operation that fragments badly and made AE's own GPU allocations fail).
	// The cudaArrays below still track exact dimensions; 2D addressing requires it.
	size_t alloc_px = 0;
	int    arr_w = 0, arr_h = 0;

	cudaEvent_t evStart = nullptr, evStop = nullptr;
};

// ── Small device helpers, bit-for-bit mirrors of the CPU versions ──────────

__device__ __forceinline__ float Clamp01f(float v)
{
	return __saturatef(v);
}

__device__ __forceinline__ float Mixf(float a, float b, float t)
{
	return fmaf(b - a, t, a);
}

// Mirrors SmoothStep(): the degenerate-edge branch matters, several call sites
// pass edges that can collapse (e.g. radius-derived softness at radius 0).
__device__ __forceinline__ float SmoothStepf(float e0, float e1, float x)
{
	const float d = e1 - e0;
	if (fabsf(d) <= ZLUX_EPS) return (x < e0) ? 0.0f : 1.0f;
	const float t = Clamp01f((x - e0) / d);
	return t * t * fmaf(-2.0f, t, 3.0f);
}

__device__ __forceinline__ float Lumaf(float r, float g, float b)
{
	return Clamp01f(fmaf(r, 0.299f, fmaf(g, 0.587f, b * 0.114f)));
}

// Mirrors ComputeHighlightMask().
__device__ __forceinline__ float HighlightMask(float r, float g, float b,
                                               const ZluxGatherParams& P)
{
	const float lum  = Lumaf(r, g, b) * 255.0f;
	const float lo   = P.highlights_low;
	const float hi   = fmaxf(lo + 1.0f, P.highlights_high);
	const float soft = fmaxf(0.5f, P.highlights_softness * 255.0f);
	const float a = SmoothStepf(lo - soft, lo + soft, lum);
	const float bm = 1.0f - SmoothStepf(hi - soft, hi + soft, lum);
	return Clamp01f(a * bm);
}

// Mirrors PickVogelLUT().
__device__ __forceinline__ int PickVogelCount(float blur_norm, int base)
{
	const float fb = (float)base;
	if (blur_norm < 0.06f) return max(16,  (int)(fb * blur_norm * 4.0f));
	if (blur_norm < 0.18f) return max(48,  (int)(fb * fmaf(blur_norm, 1.6f, 0.2f)));
	if (blur_norm < 0.40f) return max(96,  (int)(fb * fmaf(blur_norm, 0.9f, 0.45f)));
	return base;
}

// Mirrors PickMipLevel() / PickMipLevelF(). 1.7725 = sqrt(pi).
__device__ __forceinline__ float MipLevelF(float eff_radius_px, int n_samples)
{
	const float n    = (float)max(1, n_samples);
	const float cell = eff_radius_px * 1.7725f * rsqrtf(n);
	if (cell <= 2.0f) return 0.0f;
	return fmaxf(0.0f, __log2f(cell * 0.5f));
}

__device__ __forceinline__ int MipLevelI(float eff_radius_px, int n_samples, int num_levels)
{
	if (num_levels <= 1) return 0;
	const float mf = MipLevelF(eff_radius_px, n_samples);
	if (mf <= 0.0f) return 0;
	return min((int)ceilf(mf - 1e-6f), num_levels - 1);
}

// Cat's-eye optical vignetting, hoisted exactly as MakeCatsEye/EvalCatsEye do.
struct CatsEye {
	bool  active;
	float bx, by, inner, outer;
};

__device__ __forceinline__ CatsEye MakeCatsEyeD(float u, float v, float strength, float scale)
{
	CatsEye c; c.active = false; c.bx = c.by = 0.0f; c.inner = c.outer = 1.0f;
	const float abs_s = fabsf(strength);
	if (abs_s < 0.001f) return c;
	const float fcx = u - 0.5f, fcy = v - 0.5f;
	const float fd  = sqrtf(fcx * fcx + fcy * fcy);
	if (fd < 0.02f) return c;
	const float field    = Clamp01f(fd / 0.707f);
	const float barrel_r = fmaxf(0.6f, fmaf(scale - 1.0f, 0.5f, 1.0f));
	const float max_shift = (1.0f + barrel_r) * 0.88f;
	float shift_mag = field * abs_s * 1.2f;
	if (shift_mag > max_shift) shift_mag = max_shift;
	const float sign = (strength >= 0.0f) ? -1.0f : 1.0f;
	c.bx = sign * (fcx / fd) * shift_mag;
	c.by = sign * (fcy / fd) * shift_mag;
	c.inner = barrel_r * 0.93f;
	c.outer = barrel_r * 1.02f;
	c.active = true;
	return c;
}

__device__ __forceinline__ float EvalCatsEyeD(const CatsEye& c, float px, float py)
{
	if (!c.active) return 1.0f;
	const float dx = px - c.bx, dy = py - c.by;
	return 1.0f - SmoothStepf(c.inner, c.outer, sqrtf(dx * dx + dy * dy));
}


// Mirrors SampleApertureTextureMask(). The per-bokeh rotation quantises the
// SOURCE uv to a coarse grid and hashes it, so every output pixel gathering the
// same bright point agrees on the rotation and the bokeh stays coherent, while
// different points each get their own orientation.
__device__ __forceinline__ float ApertureTexMask(cudaTextureObject_t tex,
                                                 float nx, float ny,
                                                 float field_u, float field_v,
                                                 float su, float sv,
                                                 const ZluxGatherParams& P)
{
	const float intensity = Clamp01f(P.aperture_texture_intensity);
	if (intensity < 0.001f) return 1.0f;

	float rx = nx, ry = ny;
	if (su >= 0.0f && sv >= 0.0f) {
		const int gx = (int)floorf(su * 256.0f);
		const int gy = (int)floorf(sv * 256.0f);
		unsigned int hsh = (unsigned int)gx * 374761393u ^ (unsigned int)gy * 668265263u;
		hsh = (hsh ^ (hsh >> 13)) * 1274126177u;
		hsh ^= hsh >> 16;
		const float2 r = c_bokehRot[hsh & 127u];
		rx = nx * r.x - ny * r.y;
		ry = nx * r.y + ny * r.x;
	}

	const float inv_scale = 1.0f / fmaxf(0.25f, P.aperture_texture_scale);
	float u = fmaf(rx * 0.5f, inv_scale, 0.5f);
	float v = fmaf(ry * 0.5f, inv_scale, 0.5f);
	if (fabsf(P.aperture_texture_offset) > 0.001f) {
		u += (field_u - 0.5f) * P.aperture_texture_offset;
		v += (field_v - 0.5f) * P.aperture_texture_offset;
	}
	// cudaAddressModeWrap does the seamless tiling; no manual floor() needed.
	float mask = tex2D<float>(tex, u, v);
	if (P.aperture_texture_invert) mask = 1.0f - mask;
	return Mixf(1.0f, Clamp01f(mask), intensity);
}

// ── The gather ─────────────────────────────────────────────────────────────
//
// PASS: 0 = Far, 1 = Near. Templated so the pass-specific gate compiles down to
// straight-line code with no per-tap branch, mirroring the CPU's
// `if constexpr (MODE == DofPass::Far)`.
template <int PASS>
__device__ void GatherPassD(
	const ZluxGatherParams& P,
	cudaTextureObject_t texPyr,
	cudaTextureObject_t texCoc,
	cudaTextureObject_t texDisc,
	bool                hasDisc,
	cudaTextureObject_t texApTex,
	cudaTextureObject_t texIrisMod,
	const ZluxVogelHot*  __restrict__ hot,
	const ZluxVogelCold* __restrict__ cold,
	const ZluxVogelLutDesc* __restrict__ descs,
	int    num_luts,
	float  u, float v,
	float  radius,
	float  center_depth,
	float  center_signed_coc,
	float4& out_rgbw,
	float&  out_matte)
{
	const float4 center = tex2DLod<float4>(texPyr, u, v, 0.0f);
	const bool   do_alpha = (P.has_alpha != 0);

	if (radius <= 0.001f) {
		if (PASS == 0) { out_rgbw = make_float4(center.x, center.y, center.z, 0.0f); out_matte = do_alpha ? center.w : 1.0f; }
		else           { out_rgbw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);             out_matte = 0.0f; }
		return;
	}

	const float min_dim         = 1.0f / fmaxf(P.inv_w, P.inv_h);
	const float effective_radius = radius * min_dim * 0.15f;
	const float blur_norm        = Clamp01f(radius / ZLUX_COC_CLAMP);

	const bool energy        = (P.energy_conserving != 0);
	const bool uniform_blur  = (P.uniform_blur != 0);
	const bool has_ca        = (P.ca_strength > 0.001f);
	const bool has_vig       = (fabsf(P.vignetting) > 0.001f);
	const bool has_highlight = (P.highlight_boost > 0.001f) && !energy;
	const bool has_spher     = (fabsf(P.spherical_aberration_amount) > 0.001f);
	const bool has_gamma     = (P.bokeh_gamma > 0.001f) && !energy;
	const bool has_scatter   = (P.highlight_scatter > 0.001f);
	// See the CPU gather: preservative folds the specular emphasis into the
	// weights so the result stays a convex combination of the samples and cannot
	// exceed the brightest one; additive layers an un-normalised bucket on top.
	const bool preservative  = has_scatter && (P.highlight_mode == 1);
	const bool additive_scat = has_scatter && !preservative;

	// Spherical-aberration per-pixel resolve (which baked curve, and strength).
	const float sa_norm    = (2.0f + P.spherical_aberration_scale * 6.0f + 1.0f) * 0.5f;
	const float sa_t       = Clamp01f(fabsf(P.spherical_aberration_amount));
	const bool  sa_use_pos = (P.spherical_aberration_amount *
	                          ((center_depth >= P.focal_distance) ? 1.0f : -1.0f)) > 0.0f;

	// ── Full-res silhouette band ───────────────────────────────────────────
	// Where a CoC discontinuity falls within the gather's reach, force every
	// colour tap to mip 0 so the per-tap depth gate rejects wrong-plane colour
	// BEFORE it is averaged (otherwise a coarse mip texel straddling the
	// silhouette is already a contaminated FG+BG blend).
	const float band_cap = (P.render_mode == 3) ? 32.0f : (P.render_mode == 1) ? 10.0f : 22.0f;
	bool fullres_band = false;
	if (!uniform_blur && effective_radius <= band_cap && hasDisc) {
		const float reach = fminf(fmaxf(effective_radius, 2.0f), 48.0f);
		const float d = tex2D<float>(texDisc, u * P.cache_w, v * P.cache_h);
		fullres_band = (d <= reach);
	}

	// Chromatic aberration field factor (per pixel; per-tap defocus applies in
	// the loop). Opponent-axis rows are frame-constant.
	float ca_field = 0.0f;
	if (has_ca) {
		const float cu = u - 0.5f, cv = v - 0.5f;
		ca_field = fmaf(sqrtf(cu * cu + cv * cv), 0.9f, 0.55f);
	}
	const float ca_mr =  P.ca_rc       - 0.5f * P.ca_gm - 0.5f * P.ca_by;
	const float ca_mg = -0.5f * P.ca_rc + P.ca_gm       - 0.5f * P.ca_by;
	const float ca_mb = -0.5f * P.ca_rc - 0.5f * P.ca_gm + P.ca_by;

	const CatsEye cats = has_vig ? MakeCatsEyeD(u, v, P.vignetting, P.vignetting_scale)
	                             : CatsEye{false, 0.0f, 0.0f, 1.0f, 1.0f};

	// ── Tap budget ─────────────────────────────────────────────────────────
	int desired_N = PickVogelCount(blur_norm, P.sample_count);

	// ── Astigmatism ────────────────────────────────────────────────────────
	// Everything except the per-tap projection is pixel-constant, so the field
	// angle, edge strength and the two axis vectors are hoisted here and the tap
	// loop pays only a 2x2 basis change. Mirrors ApplyAstigmatism exactly.
	const bool has_astig = (P.astigmatism > 0.001f);
	float ax_tx = 0.0f, ax_ty = 1.0f, ax_sx = 1.0f, ax_sy = 0.0f;
	float tang_scale = 1.0f, sag_scale = 1.0f;
	bool astig_active = false;
	if (has_astig) {
		const float cx = u - 0.5f, cy = v - 0.5f;
		const float dist = sqrtf(cx * cx + cy * cy);
		if (dist >= 0.025f) {
			const float fa = atan2f(cy, cx);
			const float ef = SmoothStepf(0.05f, 0.55f, dist * 2.0f);
			const float edge = fminf(P.astigmatism * ef, 2.5f);
			float sfa, cfa; __sincosf(fa, &sfa, &cfa);
			ax_tx = -sfa; ax_ty = cfa;      // tangential
			ax_sx =  cfa; ax_sy = sfa;      // sagittal
			const float stretch = fmaf(edge, 0.6f, 1.0f);
			const float squeeze = 1.0f / stretch;
			tang_scale = P.astigmatism_sagittal ? squeeze : stretch;
			sag_scale  = P.astigmatism_sagittal ? stretch : squeeze;
			astig_active = true;
		}
	}

	const float anam = fmaxf(0.1f, P.anamorphic_ratio);
	const float anam_stretch = fmaxf(anam, 1.0f / anam);
	// The mip picker and the tap cap must see the LONG axis, or an elongated
	// iris gets under-sampled and the Vogel lattice prints through as a mesh.
	const float astig_stretch = has_astig ? fmaf(fminf(2.0f, P.astigmatism), 0.6f, 1.0f) : 1.0f;
	const float aniso = astig_stretch * anam_stretch;
	const float eff_long = effective_radius * aniso;

	{
		const int   mip_est   = MipLevelI(eff_long, desired_N, P.num_levels);
		const float footprint = fullres_band ? 2.0f : 2.0f * exp2f((float)mip_est);
		const float n_needed  = CUDART_PI_F * effective_radius * effective_radius * aniso
		                      / fmaxf(1.0f, footprint * footprint);
		const float headroom  = (P.render_mode == 3) ? 2.5f : 1.6f;
		const int   n_cap     = (int)ceilf(n_needed * headroom) + 24;
		desired_N = min(desired_N, max(48, n_cap));
	}

	int best = num_luts - 1;
	for (int li = 0; li < num_luts; ++li) {
		if (descs[li].count >= desired_N) { best = li; break; }
	}
	const int actual_N = descs[best].count;
	const int base     = descs[best].offset;

	// ── Mip selection, including the radius-tied footprint floor ────────────
	int   mip   = max(0, MipLevelI(eff_long, actual_N, P.num_levels));
	float mip_f = MipLevelF(eff_long, actual_N);
	{
		// Floor the footprint at a fixed fraction of the blur radius so a
		// defocused region always averages a real neighbourhood rather than
		// echoing sharp source texture into the bokeh (the "fingerprint
		// speckle"). Rises with highlight boost, which amplifies bright texels.
		const float k_floor = 0.35f + (energy ? 0.0f : fminf(0.30f, P.highlight_boost * 0.08f));
		const float fp_floor = fmaxf(2.0f, effective_radius * k_floor);
		const float mipf_floor = __log2f(fp_floor * 0.5f);
		if (mipf_floor > mip_f) mip_f = mipf_floor;
		const int mip_floor_i = (int)ceilf(mipf_floor - 1e-6f);
		if (mip_floor_i > mip) mip = mip_floor_i;
		const int maxL = P.num_levels - 1;
		mip   = min(mip, maxL);
		mip_f = fminf(mip_f, (float)maxL);
	}
	if (fullres_band) { mip = 0; mip_f = 0.0f; }

	// ── Per-pixel blue-noise rotation (interleaved gradient noise) ──────────
	const float px_f = u / fmaxf(P.inv_w, 1e-9f);
	const float py_f = v / fmaxf(P.inv_h, 1e-9f);
	const float ign  = fmodf(52.9829189f * fmodf(fmaf(0.06711056f, px_f, 0.00583715f * py_f), 1.0f), 1.0f);
	const float bn   = (ign - 0.5f) * (P.mask_angular ? (CUDART_PI_F / 120.0f) : (CUDART_PI_F / 9.0f));
	float cos_bn, sin_bn;
	__sincosf(bn, &sin_bn, &cos_bn);

	// ── Anisotropic multi-tap ──────────────────────────────────────────────
	// A stretched iris forced the mip one level coarser above, which costs
	// short-axis detail. Two taps at mip-1, offset along the long axis, cover
	// the same long-axis footprint while resolving texture at the finer level.
	const bool use_multitap = (aniso > 1.25f) && (mip > 0) && !has_ca;
	float major_x = 1.0f, major_y = 0.0f, tap_off_px = 0.0f;
	if (use_multitap) {
		if (has_astig) {
			const float cx = u - 0.5f, cy = v - 0.5f;
			const float d2 = cx * cx + cy * cy;
			if (d2 > 1e-6f) {
				const float inv_d = rsqrtf(d2);
				if (P.astigmatism_sagittal) { major_x = cx * inv_d; major_y = cy * inv_d; }
				else                        { major_x = -cy * inv_d; major_y = cx * inv_d; }
			}
		} else {
			major_x = (P.anamorphic_ratio > 1.0f) ? 0.0f : 1.0f;
			major_y = (P.anamorphic_ratio > 1.0f) ? 1.0f : 0.0f;
		}
		const float cell_long = eff_long * 1.7725f * rsqrtf((float)max(1, actual_N));
		tap_off_px = cell_long * 0.25f;
	}
	const float tap_du = major_x * tap_off_px * P.inv_w;
	const float tap_dv = major_y * tap_off_px * P.inv_h;

	const bool  jitter = (aniso > 1.05f) || (P.mask_angular != 0);
	const float jit_cell  = effective_radius * 1.7725f * rsqrtf((float)max(1, actual_N));
	const float jit_amt_u = jit_cell * P.inv_w * 0.55f;
	const float jit_amt_v = jit_cell * P.inv_h * 0.55f;

	// ── Accumulators ───────────────────────────────────────────────────────
	float3 acc = make_float3(0.0f, 0.0f, 0.0f);
	float3 spec = make_float3(0.0f, 0.0f, 0.0f);
	float  w_sum = 0.0f, potential = 0.0f, cover = 0.0f, a_acc = 0.0f;
	float  near_max_reach = 0.0f;

	// ── Centre seed ────────────────────────────────────────────────────────
	// Deliberately wider thresholds than the per-tap gate: seeding with the
	// sharp centre where the CoC is tiny is what creates the bright halo around
	// sharp subjects, so the seed weight ramps off as the CoC grows.
	const float seed_gate  = 1.0f - SmoothStepf(0.015f, 0.04f, fabsf(center_signed_coc));
	const float seed_spher = has_spher ? Mixf(1.0f, (sa_use_pos ? 0.0f : 1.0f) * sa_norm, sa_t) : 1.0f;
	{
		const bool seed_ok = (PASS == 0)
			? ((uniform_blur || center_signed_coc > 0.008f) && seed_gate > 0.001f)
			: (!uniform_blur && center_signed_coc < -0.008f);
		if (seed_ok) {
			float cw_geom = (PASS == 0) ? seed_gate * seed_spher : seed_spher;
			const float cw = has_highlight
				? cw_geom * fmaf(Lumaf(center.x, center.y, center.z), P.highlight_boost * 8.0f, 1.0f)
				: cw_geom;
			acc.x = fmaf(center.x, cw, acc.x);
			acc.y = fmaf(center.y, cw, acc.y);
			acc.z = fmaf(center.z, cw, acc.z);
			if (do_alpha) a_acc = fmaf(center.w, cw, a_acc);
			w_sum     += cw;
			potential += cw_geom;
			cover     += cw_geom;
			near_max_reach = (PASS == 0) ? fmaxf(0.0f, center_signed_coc) : -center_signed_coc;
		}
	}

	const float pos_cx = effective_radius * P.inv_w;
	const float pos_cy = effective_radius * P.inv_h;

	// ── Tap loop ───────────────────────────────────────────────────────────
	for (int i = 0; i < actual_N; ++i) {
		// ZluxVogelHot is laid out exactly as a float4 (kx, ky, fr, static_mask),
		// so the whole record is one 16-byte vectorised load. Every thread in
		// the warp reads the same address, so __ldg broadcasts it out of the
		// read-only cache instead of issuing 32 separate transactions.
		const float4 vs = __ldg(reinterpret_cast<const float4*>(hot) + (base + i));
		const float vs_fr = vs.z;

		if (!P.has_aperture_tex && vs.w < 0.001f) continue;

		float off_u, off_v;
		if (astig_active) {
			// Astigmatism must act on the UN-squashed iris circle, with the
			// anamorphic squeeze applied last -- composing them the other way
			// cancels the swirl along the anamorphic axis.
			const ZluxVogelCold vc = cold[base + i];
			const float ca_ = vc.cos_a, sa_ = vc.sin_a;
			const float cos_a_r = ca_ * cos_bn - sa_ * sin_bn;
			const float sin_a_r = ca_ * sin_bn + sa_ * cos_bn;
			const float ix = cos_a_r * vs_fr * pos_cx;
			const float iy = sin_a_r * vs_fr * pos_cy;
			const float tc = ix * ax_tx + iy * ax_ty;   // tangential component
			const float sc = ix * ax_sx + iy * ax_sy;   // sagittal component
			off_u = (ax_tx * tc * tang_scale + ax_sx * sc * sag_scale) / anam;
			off_v =  ax_ty * tc * tang_scale + ax_sy * sc * sag_scale;
		} else {
			// Rotate the baked unit-disc offset by the per-pixel blue-noise angle.
			const float kx_r = vs.x * cos_bn - vs.y * sin_bn;
			const float ky_r = vs.x * sin_bn + vs.y * cos_bn;
			off_u = kx_r * pos_cx;
			off_v = ky_r * pos_cy;
		}

		float su = u + off_u;
		float sv = v + off_v;
		if (jitter) {
			const float h1 = fmodf(fmaf((float)i, 0.7548776662f, ign), 1.0f);
			const float h2 = fmodf(fmaf((float)i, 0.5698402910f, ign * 1.3247179572f), 1.0f);
			su = fmaf(h1 - 0.5f, jit_amt_u, su);
			sv = fmaf(h2 - 0.5f, jit_amt_v, sv);
		}

		float mask = vs.w;

		// Custom iris shape REPLACES the baked mask (its rotation and offset both
		// depend on the sampled source position, so it cannot be baked); the
		// modulator layer multiplies whatever shape is already there.
		if (P.has_aperture_tex) {
			const ZluxVogelCold vc = cold[base + i];
			mask = ApertureTexMask(texApTex, vc.norm_x, vc.norm_y, u, v, su, sv, P)
			     * vc.soft_edge;
			if (mask < 0.001f) continue;
		} else if (P.has_iris_mod) {
			const ZluxVogelCold vc = cold[base + i];
			mask *= ApertureTexMask(texIrisMod, vc.norm_x, vc.norm_y, u, v, su, sv, P);
			if (mask < 0.001f) continue;
		}

		if (has_spher) {
			const ZluxVogelCold vc = cold[base + i];
			mask *= Mixf(1.0f, (sa_use_pos ? vc.sa_pos : vc.sa_neg) * sa_norm, sa_t);
			if (mask < 0.001f) continue;
		}
		if (has_vig) {
			const ZluxVogelCold vc = cold[base + i];
			mask *= EvalCatsEyeD(cats, vc.norm_x, vc.norm_y);
			if (mask < 0.001f) continue;
		}

		potential += mask;

		// Sample-side CoC. Point-sampled through the texture unit; the CPU
		// truncates the same way (int cast of u*cache_w).
		float s_coc;
		if (uniform_blur) {
			s_coc = radius;
		} else {
			s_coc = tex2D<float>(texCoc, su * P.cache_w, sv * P.cache_h);
		}

		// ── Pass-specific acceptance ───────────────────────────────────────
		float gate, plane_w = 1.0f;
		const float softness = fmaxf(0.02f, radius * 0.15f);
		const float req      = vs_fr * radius;

		if (PASS == 0) {
			if (fullres_band) {
				plane_w = SmoothStepf(-0.004f, 0.010f, s_coc);
				if (plane_w < 0.01f) continue;
			} else if (s_coc < 0.006f) {
				continue;
			}
			gate = SmoothStepf(fmaxf(0.0f, req - softness), req + softness, fmaxf(0.0f, s_coc));
			// Occlusion gate: reject in proportion to how much NEARER the
			// sample sits than the centre, so a closer far surface cannot drag
			// its colour across the silhouette. The reverse direction is
			// intentionally not gated (physical background-over-foreground bleed).
			const float occl = SmoothStepf(0.10f, 0.32f,
				(center_signed_coc - s_coc) / fmaxf(0.04f, radius));
			gate *= (1.0f - occl) * plane_w;
			if (gate < 0.01f) continue;
			near_max_reach = fmaxf(near_max_reach, s_coc);
		} else {
			if (fullres_band) {
				plane_w = SmoothStepf(-0.004f, 0.010f, -s_coc);
				if (plane_w < 0.01f) continue;
			} else if (s_coc > -0.006f) {
				continue;
			}
			const float reach = -s_coc;
			gate = SmoothStepf(fmaxf(0.0f, req - softness), req + softness, reach) * plane_w;
			if (gate < 0.01f) continue;
			near_max_reach = fmaxf(near_max_reach, reach);
		}

		float w = mask * gate;
		cover += w;

		// ── Colour ─────────────────────────────────────────────────────────
		float cr, cg, cb, ca_a = 1.0f;
		if (has_ca) {
			const float plane_sign = (s_coc >= 0.0f) ? 1.0f : -1.0f;
			const float tap_def = Clamp01f(fabsf(s_coc) * (1.0f / ZLUX_COC_CLAMP));
			const float f = ca_field * fmaf(tap_def, 1.15f, 0.35f) * plane_sign;
			const float rs = fminf(fmaxf(ca_mr * f, -2.8f), 2.8f);
			const float gs = fminf(fmaxf(ca_mg * f, -2.8f), 2.8f);
			const float bs = fminf(fmaxf(ca_mb * f, -2.8f), 2.8f);
			cr = tex2DLod<float4>(texPyr, su + off_u * rs, sv + off_v * rs, mip_f).x;
			cg = tex2DLod<float4>(texPyr, su + off_u * gs, sv + off_v * gs, mip_f).y;
			cb = tex2DLod<float4>(texPyr, su + off_u * bs, sv + off_v * bs, mip_f).z;
			if (do_alpha) ca_a = tex2DLod<float4>(texPyr, su, sv, mip_f).w;
		} else if (use_multitap) {
			const float lf = (float)(mip - 1);
			const float4 c0 = tex2DLod<float4>(texPyr, su - tap_du, sv - tap_dv, lf);
			const float4 c1 = tex2DLod<float4>(texPyr, su + tap_du, sv + tap_dv, lf);
			cr = (c0.x + c1.x) * 0.5f;
			cg = (c0.y + c1.y) * 0.5f;
			cb = (c0.z + c1.z) * 0.5f;
			ca_a = (c0.w + c1.w) * 0.5f;
		} else {
			// The whole point of the port: one hardware trilinear fetch where
			// the CPU did eight scattered loads plus the blend.
			const float4 c = tex2DLod<float4>(texPyr, su, sv, mip_f);
			cr = c.x; cg = c.y; cb = c.z; ca_a = c.w;
		}

		const float lum = Lumaf(cr, cg, cb);

		if (has_highlight) w *= fmaf(lum, P.highlight_boost * 8.0f, 1.0f);
		if (has_gamma) {
			// 256-interval lerp replacing the per-tap pow(); the curve is
			// frame-constant so it is baked host-side into constant memory.
			const float lf = lum * 256.0f;
			int   li = min(255, max(0, (int)lf));
			const float lt = lf - (float)li;
			w *= fmaf(c_bokehGamma[li + 1] - c_bokehGamma[li], lt, c_bokehGamma[li]);
		}

		if (preservative) {
			const float sf = HighlightMask(cr, cg, cb, P);
			if (sf > 0.001f) w *= fmaf(P.highlight_scatter * 8.0f, sf, 1.0f);
		}

		acc.x = fmaf(cr, w, acc.x);
		acc.y = fmaf(cg, w, acc.y);
		acc.z = fmaf(cb, w, acc.z);
		w_sum += w;
		if (do_alpha) a_acc = fmaf(ca_a, w, a_acc);

		if (additive_scat) {
			const float sf = HighlightMask(cr, cg, cb, P);
			if (sf > 0.001f) {
				const float sw = mask * gate * sf;
				spec.x = fmaf(cr, sw, spec.x);
				spec.y = fmaf(cg, sw, spec.y);
				spec.z = fmaf(cb, sw, spec.z);
			}
		}
	}

	if (w_sum <= ZLUX_EPS) {
		if (PASS == 0) { out_rgbw = make_float4(center.x, center.y, center.z, 0.0f); out_matte = do_alpha ? center.w : 1.0f; }
		else           { out_rgbw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);             out_matte = 0.0f; }
		return;
	}

	const float inv_w_sum = 1.0f / w_sum;
	float3 rgb = make_float3(acc.x * inv_w_sum, acc.y * inv_w_sum, acc.z * inv_w_sum);

	// Info-density deficit fill (Far only). Where coverage is short the few
	// surviving samples read as a foggy patch; blend toward the local blurred
	// colour -- but only DOWNWARD, since the local mip at a silhouette is
	// contaminated by the (often bright) occluder and letting it brighten paints
	// the milky halo this whole gate exists to remove.
	if (PASS == 0) {
		const float cov = Clamp01f(cover / fmaxf(potential, ZLUX_EPS));
		const float deficit = Clamp01f(1.0f - cov);
		if (deficit > 0.001f) {
			const float4 fill = tex2DLod<float4>(texPyr, u, v, mip_f);
			rgb.x = Mixf(rgb.x, fminf(rgb.x, fill.x), deficit);
			rgb.y = Mixf(rgb.y, fminf(rgb.y, fill.y), deficit);
			rgb.z = Mixf(rgb.z, fminf(rgb.z, fill.z), deficit);
		}
	}

	if (additive_scat) {
		const float ss = P.highlight_scatter / fmaxf(1.0f, (float)actual_N);
		rgb.x = fmaf(spec.x, ss, rgb.x);
		rgb.y = fmaf(spec.y, ss, rgb.y);
		rgb.z = fmaf(spec.z, ss, rgb.z);
	}

	// Area-coverage alpha. `radius` is the tile-dilated maximum, which strong
	// blur anywhere nearby inflates far beyond what local content can cover, so
	// the denominator is renormalized to the disc the observed content could
	// plausibly fill. Without this every silhouette goes translucent at high
	// blur.
	const float r_loc = fminf(fmaxf(near_max_reach, 1e-6f), radius);
	const float area  = r_loc / fmaxf(radius, 1e-6f);
	const float alpha = (potential > ZLUX_EPS)
		? Clamp01f(cover / fmaxf(potential * area * area, ZLUX_EPS))
		: 0.0f;

	out_rgbw  = make_float4(rgb.x, rgb.y, rgb.z, alpha);
	out_matte = do_alpha ? Clamp01f(a_acc * inv_w_sum)
	                     : ((PASS == 0) ? (do_alpha ? center.w : 1.0f) : 0.0f);
}



// ── Per-pixel gather radii ─────────────────────────────────────────────────
//
// Ported from the CPU precompute that fed the gather. Every input it needs is
// already resident (the CoC texture) or tiny (the per-tile near-reach array),
// and computing the radii here also removes three full-frame uploads.
// The expressions are copied verbatim from RenderPixelImpl so the composite
// downstream keeps its own gating and sees exactly what it saw before.

__device__ __forceinline__ float FieldCurvatureCocD(float u, float v, float amount, float sweet)
{
	const float dx = u - 0.5f, dy = v - 0.5f;
	const float fr = sqrtf(dx * dx + dy * dy) * 2.0f;
	const float t = SmoothStepf(sweet, sweet + 0.55f, fr);
	return amount * 0.35f * t * t;
}

// Occlusion-sliver rescue: on the interpolated depth edge between a blurred
// foreground and a blurred background the CoC crosses zero, so those pixels read
// as "in focus" and would keep the sharp source as their base layer, showing up
// as a ghost stripe along silhouettes.
__device__ __forceinline__ float DetectCocSliverD(cudaTextureObject_t coc,
                                                  int w, int h, int x, int y,
                                                  float center, float* out_far_max)
{
	if (fabsf(center) >= 0.06f) { *out_far_max = 0.0f; return 0.0f; }
	float c_min = center, c_max = center;
	#pragma unroll
	for (int k = 0; k < 2; ++k) {
		const int o = (k == 0) ? 3 : 6;
		const float c1 = tex2D<float>(coc, (float)max(0, x - o),     (float)y);
		const float c2 = tex2D<float>(coc, (float)min(w - 1, x + o), (float)y);
		const float c3 = tex2D<float>(coc, (float)x, (float)max(0, y - o));
		const float c4 = tex2D<float>(coc, (float)x, (float)min(h - 1, y + o));
		c_min = fminf(c_min, fminf(fminf(c1, c2), fminf(c3, c4)));
		c_max = fmaxf(c_max, fmaxf(fmaxf(c1, c2), fmaxf(c3, c4)));
	}
	*out_far_max = fmaxf(0.0f, c_max);
	const float fields = fminf(-c_min, c_max);
	return SmoothStepf(0.012f, 0.04f, fields)
	     * (1.0f - SmoothStepf(0.03f, 0.06f, fabsf(center)));
}

__device__ __forceinline__ float ProbeFarReachWideD(cudaTextureObject_t coc,
                                                    int w, int h, int x, int y,
                                                    float px_per_coc)
{
	float c_max = 0.0f;
	#pragma unroll
	for (int k = 0; k < 5; ++k) {
		const int o = (k == 0) ? 3 : (k == 1) ? 6 : (k == 2) ? 12 : (k == 3) ? 24 : 48;
		float c =        tex2D<float>(coc, (float)max(0, x - o),     (float)y);
		c = fmaxf(c,     tex2D<float>(coc, (float)min(w - 1, x + o), (float)y));
		c = fmaxf(c,     tex2D<float>(coc, (float)x, (float)max(0, y - o)));
		c = fmaxf(c,     tex2D<float>(coc, (float)x, (float)min(h - 1, y + o)));
		if (c > c_max && fmaf(c * px_per_coc, 1.5f, 3.0f) >= (float)o) c_max = c;
	}
	return c_max;
}

__global__ void zluxRadiiKernel(cudaTextureObject_t texCoc,
                                const float* __restrict__ tileMinCoc,
                                ZluxGatherParams P,
                                int tilesX, int tileSize,
                                float px_per_coc, float uniform_base,
                                float field_curvature, float field_sweet,
                                float* __restrict__ farR,
                                float* __restrict__ nearR,
                                float* __restrict__ bleedR)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= P.width || y >= P.height) return;
	const int idx = y * P.width + x;

	if (P.uniform_blur) {
		float ur = fmaxf(0.0f, uniform_base);
		if (field_curvature > 0.001f) {
			ur += FieldCurvatureCocD(((float)x + 0.5f) * P.inv_w,
			                         ((float)y + 0.5f) * P.inv_h,
			                         field_curvature, field_sweet);
		}
		farR[idx] = ur; nearR[idx] = 0.0f; bleedR[idx] = 0.0f;
		return;
	}

	const float sc = tex2D<float>(texCoc, (float)x, (float)y);

	float sliver_far = 0.0f;
	const float sliver = DetectCocSliverD(texCoc, P.width, P.height, x, y, sc, &sliver_far);
	const float far_need = fmaxf(fmaxf(0.0f, sc), sliver * sliver_far);
	farR[idx] = (far_need > 0.001f) ? far_need : 0.0f;

	// Near uses the tile-dilated maximum so a nearby near pixel can still splash
	// into this one even when this pixel is focused itself.
	const int tx = x / tileSize, ty = y / tileSize;
	const float tile_near = fmaxf(0.0f, -tileMinCoc[ty * tilesX + tx]);
	const float nr = fmaxf(tile_near, fmaxf(0.0f, -sc));
	nearR[idx] = (nr > 0.001f) ? nr : 0.0f;

	const float nb = ProbeFarReachWideD(texCoc, P.width, P.height, x, y, px_per_coc);
	bleedR[idx] = (nb > 0.012f) ? nb : 0.0f;
}

// ── CoC-discontinuity distance field ───────────────────────────────────────
//
// The gather forces full-res colour taps in a band around every CoC silhouette,
// and reads the distance to the nearest discontinuity to decide. On the CPU that
// field is built by a two-pass chamfer transform whose raster sweeps are
// strictly sequential -- 13.7 ms that cannot be threaded.
//
// Here it is a jump-flooding transform instead: every pass is fully parallel and
// only log2(range) passes are needed. Crucially the gather only ever tests
// `dist <= reach` with reach clamped to [2, 48], so distances beyond ~64 px are
// irrelevant and the flood can start at a bounded step instead of the image
// size. That turns an O(n) serial scan into ~7 parallel passes.
//
// The result is true Euclidean distance where the CPU produced a chamfer
// approximation (which overestimates by up to ~4%), so the band boundary can
// land a pixel differently. That only shifts which taps read mip 0, never the
// structure of the image.
#define ZLUX_DISC_THRESH 0.03f
#define ZLUX_DISC_RANGE  64      // px; beyond the gather's max reach of 48

__global__ void zluxDiscSeedKernel(cudaTextureObject_t texCoc,
                                   int w, int h,
                                   short2* __restrict__ seed)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	const float c = tex2D<float>(texCoc, x, y);
	float md = 0.0f;
	if (x > 0)     md = fmaxf(md, fabsf(c - tex2D<float>(texCoc, x - 1, y)));
	if (x < w - 1) md = fmaxf(md, fabsf(c - tex2D<float>(texCoc, x + 1, y)));
	if (y > 0)     md = fmaxf(md, fabsf(c - tex2D<float>(texCoc, x, y - 1)));
	if (y < h - 1) md = fmaxf(md, fabsf(c - tex2D<float>(texCoc, x, y + 1)));

	seed[y * w + x] = (md > ZLUX_DISC_THRESH) ? make_short2((short)x, (short)y)
	                                          : make_short2((short)-1, (short)-1);
}

__global__ void zluxJfaStepKernel(const short2* __restrict__ src,
                                  short2* __restrict__ dst,
                                  int w, int h, int step)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	short2 best = src[y * w + x];
	float bestd = (best.x < 0) ? 3.4e38f
	            : (float)((x - best.x) * (x - best.x) + (y - best.y) * (y - best.y));

	#pragma unroll
	for (int dy = -1; dy <= 1; ++dy) {
		#pragma unroll
		for (int dx = -1; dx <= 1; ++dx) {
			const int nx = x + dx * step, ny = y + dy * step;
			if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
			const short2 cand = src[ny * w + nx];
			if (cand.x < 0) continue;
			const float d = (float)((x - cand.x) * (x - cand.x) + (y - cand.y) * (y - cand.y));
			if (d < bestd) { bestd = d; best = cand; }
		}
	}
	dst[y * w + x] = best;
}

__global__ void zluxDiscResolveKernel(const short2* __restrict__ seed,
                                      float* __restrict__ out,
                                      int w, int h, float big)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	const short2 s = seed[y * w + x];
	out[y * w + x] = (s.x < 0)
		? big
		: sqrtf((float)((x - s.x) * (x - s.x) + (y - s.y) * (y - s.y)));
}

// ── Entry kernel ───────────────────────────────────────────────────────────
// Far and near run in one launch: they share the centre texel, the centre CoC
// and all the per-pixel setup, so splitting them would double that work and the
// global traffic with it.
// Occupancy target. The gather is texture-latency-bound, not arithmetic-bound,
// so the goal is enough resident warps to hide the fetches WITHOUT spilling the
// tap loop's live state to local memory. (256,4) forces a 64-register budget and
// measured 56 bytes of spill stores -- local memory traffic inside the hottest
// loop, which is exactly what must not happen. (256,2) allows 128 registers.
__launch_bounds__(256, 2)
__global__ void zluxGatherKernel(
	ZluxGatherParams P,
	cudaTextureObject_t texPyr,
	cudaTextureObject_t texCoc,
	cudaTextureObject_t texDisc,
	int hasDisc,
	cudaTextureObject_t texApTex,
	cudaTextureObject_t texIrisMod,
	const ZluxVogelHot*  __restrict__ hot,
	const ZluxVogelCold* __restrict__ cold,
	const ZluxVogelLutDesc* __restrict__ descs,
	int num_luts,
	const float* __restrict__ farRadius,
	const float* __restrict__ nearRadius,
	const float* __restrict__ bleedRadius,
	const float* __restrict__ centerDepth,
	float4* __restrict__ outFar,
	float4* __restrict__ outNear,
	float4* __restrict__ outBleed,
	float4* __restrict__ outMatte)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= P.width || y >= P.height) return;

	const int idx = y * P.width + x;
	// Pixel centres, matching the CPU's (x + 0.5) / w convention.
	const float u = ((float)x + 0.5f) * P.inv_w;
	const float v = ((float)y + 0.5f) * P.inv_h;

	const float coc   = tex2D<float>(texCoc, u * P.cache_w, v * P.cache_h);
	const float depth = centerDepth[idx];

	float4 far_rgbw, near_rgbw, bleed_rgbw;
	float  far_matte, near_matte, bleed_matte;

	GatherPassD<0>(P, texPyr, texCoc, texDisc, hasDisc != 0, texApTex, texIrisMod,
	               hot, cold, descs, num_luts,
	               u, v, farRadius[idx], depth, coc, far_rgbw, far_matte);

	GatherPassD<1>(P, texPyr, texCoc, texDisc, hasDisc != 0, texApTex, texIrisMod,
	               hot, cold, descs, num_luts,
	               u, v, nearRadius[idx], depth, coc, near_rgbw, near_matte);

	// Far bleed-over probe. Same Far gather, its own (usually zero) radius, so
	// pixels the composite does not need it for exit immediately.
	GatherPassD<0>(P, texPyr, texCoc, texDisc, hasDisc != 0, texApTex, texIrisMod,
	               hot, cold, descs, num_luts,
	               u, v, bleedRadius[idx], depth, coc, bleed_rgbw, bleed_matte);

	outFar[idx]   = far_rgbw;
	outNear[idx]  = near_rgbw;
	outBleed[idx] = bleed_rgbw;
	outMatte[idx] = make_float4(far_matte, near_matte, bleed_matte, 0.0f);
}

// ────────────────────────────────────────────────────────────────────────────
// Host side
// ────────────────────────────────────────────────────────────────────────────

extern "C" int zluxGpuAvailable(void)
{
	int n = 0;
	if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
	return n > 0 ? 1 : 0;
}

extern "C" const char* zluxGpuDeviceName(void)
{
	static char name[256] = {0};
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return nullptr;
	std::snprintf(name, sizeof(name), "%s (sm_%d%d, %zu MB)",
	              prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20);
	return name;
}

extern "C" int zluxGpuCanRender(const ZluxGatherParams* P, int has_aperture_tex,
                                int has_iris_mod)
{
	if (!P) return 0;
	// Custom iris shape needs its layer actually uploaded.
	if (P->aperture_shape_mode == 4 && !has_aperture_tex) return 0;
	(void)has_iris_mod;
	if (P->num_levels <= 0 || P->num_levels > ZLUX_MAX_LEVELS) return 0;
	return 1;
}

extern "C" ZluxGpuContext* zluxGpuCreate(void)
{
	ClearErr();
	if (!zluxGpuAvailable()) {
		std::snprintf(g_last_error, sizeof(g_last_error), "no CUDA device");
		g_has_error = true;
		return nullptr;
	}
	ZluxGpuContext* ctx = new ZluxGpuContext();
	if (cudaEventCreate(&ctx->evStart) != cudaSuccess ||
	    cudaEventCreate(&ctx->evStop)  != cudaSuccess) {
		delete ctx;
		std::snprintf(g_last_error, sizeof(g_last_error), "cudaEventCreate failed");
		g_has_error = true;
		return nullptr;
	}
	return ctx;
}

static void DestroyPyramid(ZluxGpuContext* ctx)
{
	if (ctx->texPyramid) { cudaDestroyTextureObject(ctx->texPyramid); ctx->texPyramid = 0; }
	if (ctx->mipArray)   { cudaFreeMipmappedArray(ctx->mipArray);     ctx->mipArray = nullptr; }
}

static void DestroyFields(ZluxGpuContext* ctx)
{
	if (ctx->texCoc)      { cudaDestroyTextureObject(ctx->texCoc);      ctx->texCoc = 0; }
	if (ctx->texDiscDist) { cudaDestroyTextureObject(ctx->texDiscDist); ctx->texDiscDist = 0; }
	if (ctx->arrCoc)      { cudaFreeArray(ctx->arrCoc);      ctx->arrCoc = nullptr; }
	if (ctx->arrDiscDist) { cudaFreeArray(ctx->arrDiscDist); ctx->arrDiscDist = nullptr; }
}

extern "C" void zluxGpuDestroy(ZluxGpuContext* ctx)
{
	if (!ctx) return;
	DestroyPyramid(ctx);
	DestroyFields(ctx);
	cudaFree(ctx->d_farRadius);
	cudaFree(ctx->d_nearRadius);
	cudaFree(ctx->d_bleedRadius);
	cudaFree(ctx->d_centerDepth);
	cudaFree(ctx->d_hot);
	cudaFree(ctx->d_cold);
	cudaFree(ctx->d_descs);
	cudaFree(ctx->d_farRGBA);
	cudaFree(ctx->d_nearRGBA);
	cudaFree(ctx->d_bleedRGBA);
	cudaFree(ctx->d_mattes);
	cudaFree(ctx->d_seedA);
	cudaFree(ctx->d_seedB);
	cudaFree(ctx->d_disc);
	cudaFree(ctx->d_tileMinCoc);
	for (int i = 0; i < 2; ++i) {
		if (ctx->texApTex[i]) cudaDestroyTextureObject(ctx->texApTex[i]);
		if (ctx->arrApTex[i]) cudaFreeArray(ctx->arrApTex[i]);
	}
	cudaFreeHost(ctx->h_farRGBA);
	cudaFreeHost(ctx->h_nearRGBA);
	cudaFreeHost(ctx->h_bleedRGBA);
	cudaFreeHost(ctx->h_mattes);
	if (ctx->evStart) cudaEventDestroy(ctx->evStart);
	if (ctx->evStop)  cudaEventDestroy(ctx->evStop);
	delete ctx;
}

extern "C" int zluxGpuUploadPyramid(ZluxGpuContext* ctx,
                                    const float* const* level_data,
                                    const int* level_w, const int* level_h,
                                    int num_levels)
{
	ClearErr();
	if (!ctx || num_levels <= 0 || num_levels > ZLUX_MAX_LEVELS) return 1;

	// Reallocate only when the shape changes -- a frame sequence at fixed
	// resolution reuses the same array and just refills it.
	if (ctx->mipArray && (ctx->pyr_w != level_w[0] || ctx->pyr_h != level_h[0] ||
	                      ctx->pyr_levels != num_levels)) {
		DestroyPyramid(ctx);
	}
	if (!ctx->mipArray) {
		cudaChannelFormatDesc ch = cudaCreateChannelDesc<float4>();
		cudaExtent ext = make_cudaExtent(level_w[0], level_h[0], 0);
		CU_TRY(cudaMallocMipmappedArray(&ctx->mipArray, &ch, ext, num_levels),
		       "cudaMallocMipmappedArray");
		ctx->pyr_w = level_w[0]; ctx->pyr_h = level_h[0]; ctx->pyr_levels = num_levels;

		cudaResourceDesc rd; std::memset(&rd, 0, sizeof(rd));
		rd.resType = cudaResourceTypeMipmappedArray;
		rd.res.mipmap.mipmap = ctx->mipArray;

		cudaTextureDesc td; std::memset(&td, 0, sizeof(td));
		// Mirror addressing matches the CPU's MirrorCoordSafe edge handling.
		td.addressMode[0] = cudaAddressModeMirror;
		td.addressMode[1] = cudaAddressModeMirror;
		td.filterMode        = cudaFilterModeLinear;
		td.mipmapFilterMode  = cudaFilterModeLinear;   // hardware trilinear
		td.normalizedCoords  = 1;
		td.maxMipmapLevelClamp = (float)(num_levels - 1);
		td.readMode = cudaReadModeElementType;

		cudaResourceViewDesc rvd; std::memset(&rvd, 0, sizeof(rvd));
		rvd.format      = cudaResViewFormatFloat4;
		rvd.width       = level_w[0];
		rvd.height      = level_h[0];
		rvd.firstMipmapLevel = 0;
		rvd.lastMipmapLevel  = num_levels - 1;

		CU_TRY(cudaCreateTextureObject(&ctx->texPyramid, &rd, &td, &rvd),
		       "cudaCreateTextureObject(pyramid)");
	}

	for (int L = 0; L < num_levels; ++L) {
		cudaArray_t levelArr = nullptr;
		CU_TRY(cudaGetMipmappedArrayLevel(&levelArr, ctx->mipArray, L),
		       "cudaGetMipmappedArrayLevel");
		const size_t pitch = (size_t)level_w[L] * 4 * sizeof(float);
		CU_TRY(cudaMemcpy2DToArray(levelArr, 0, 0, level_data[L], pitch, pitch,
		                           level_h[L], cudaMemcpyHostToDevice),
		       "cudaMemcpy2DToArray(pyramid level)");
	}
	return 0;
}

extern "C" int zluxGpuUploadLuts(ZluxGpuContext* ctx,
                                 const ZluxVogelHot* hot, const ZluxVogelCold* cold,
                                 int total_samples,
                                 const ZluxVogelLutDesc* descs, int num_luts,
                                 const float* bokeh_gamma_lut257)
{
	ClearErr();
	if (!ctx || total_samples <= 0 || num_luts <= 0) return 1;

	if (!ctx->d_hot) {
		CU_TRY(cudaMalloc(&ctx->d_hot,   (size_t)total_samples * sizeof(ZluxVogelHot)),  "cudaMalloc(hot)");
		CU_TRY(cudaMalloc(&ctx->d_cold,  (size_t)total_samples * sizeof(ZluxVogelCold)), "cudaMalloc(cold)");
		CU_TRY(cudaMalloc(&ctx->d_descs, (size_t)num_luts * sizeof(ZluxVogelLutDesc)),   "cudaMalloc(descs)");
		ctx->num_luts = num_luts;
	}
	CU_TRY(cudaMemcpy(ctx->d_hot,  hot,  (size_t)total_samples * sizeof(ZluxVogelHot),  cudaMemcpyHostToDevice), "memcpy(hot)");
	CU_TRY(cudaMemcpy(ctx->d_cold, cold, (size_t)total_samples * sizeof(ZluxVogelCold), cudaMemcpyHostToDevice), "memcpy(cold)");
	CU_TRY(cudaMemcpy(ctx->d_descs, descs, (size_t)num_luts * sizeof(ZluxVogelLutDesc), cudaMemcpyHostToDevice), "memcpy(descs)");
	if (bokeh_gamma_lut257) {
		CU_TRY(cudaMemcpyToSymbol(c_bokehGamma, bokeh_gamma_lut257, 257 * sizeof(float)),
		       "memcpyToSymbol(bokehGamma)");
	}
	return 0;
}

// Helper: (re)create a single-channel float texture backed by a cudaArray.
static int MakeFloatTex(cudaArray_t* arr, cudaTextureObject_t* tex,
                        const float* src, int w, int h, const char* what)
{
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
	if (!*arr) {
		CU_TRY(cudaMallocArray(arr, &ch, w, h), what);
		cudaResourceDesc rd; std::memset(&rd, 0, sizeof(rd));
		rd.resType = cudaResourceTypeArray;
		rd.res.array.array = *arr;
		cudaTextureDesc td; std::memset(&td, 0, sizeof(td));
		// Point sampling + clamp: the CPU truncates the coordinate and clamps
		// to the edge, so this reproduces it exactly.
		td.addressMode[0] = cudaAddressModeClamp;
		td.addressMode[1] = cudaAddressModeClamp;
		td.filterMode     = cudaFilterModePoint;
		td.normalizedCoords = 0;
		td.readMode = cudaReadModeElementType;
		CU_TRY(cudaCreateTextureObject(tex, &rd, &td, nullptr), what);
	}
	const size_t pitch = (size_t)w * sizeof(float);
	CU_TRY(cudaMemcpy2DToArray(*arr, 0, 0, src, pitch, pitch, h, cudaMemcpyHostToDevice), what);
	return 0;
}


extern "C" int zluxGpuUploadApertureTex(ZluxGpuContext* ctx, int slot,
                                        const float* luma, int w, int h,
                                        const float* rot_cos_sin_128)
{
	ClearErr();
	if (!ctx || slot < 0 || slot > 1) return 1;
	if (rot_cos_sin_128) {
		CU_TRY(cudaMemcpyToSymbol(c_bokehRot, rot_cos_sin_128, 128 * sizeof(float2)),
		       "memcpyToSymbol(bokehRot)");
	}
	if (!luma || w <= 0 || h <= 0) return 0;   // nothing to upload for this slot

	if (ctx->arrApTex[slot] && (ctx->apTexW[slot] != w || ctx->apTexH[slot] != h)) {
		cudaDestroyTextureObject(ctx->texApTex[slot]); ctx->texApTex[slot] = 0;
		cudaFreeArray(ctx->arrApTex[slot]);            ctx->arrApTex[slot] = nullptr;
	}
	if (!ctx->arrApTex[slot]) {
		cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
		CU_TRY(cudaMallocArray(&ctx->arrApTex[slot], &ch, w, h), "cudaMallocArray(apTex)");
		cudaResourceDesc rd; std::memset(&rd, 0, sizeof(rd));
		rd.resType = cudaResourceTypeArray;
		rd.res.array.array = ctx->arrApTex[slot];
		cudaTextureDesc td; std::memset(&td, 0, sizeof(td));
		// Wrap + linear + normalized: reproduces the CPU's manual u -= floor(u)
		// seamless tiling and its bilinear world sample in one fetch.
		td.addressMode[0] = cudaAddressModeWrap;
		td.addressMode[1] = cudaAddressModeWrap;
		td.filterMode     = cudaFilterModeLinear;
		td.normalizedCoords = 1;
		td.readMode = cudaReadModeElementType;
		CU_TRY(cudaCreateTextureObject(&ctx->texApTex[slot], &rd, &td, nullptr),
		       "cudaCreateTextureObject(apTex)");
		ctx->apTexW[slot] = w; ctx->apTexH[slot] = h;
	}
	const size_t pitch = (size_t)w * sizeof(float);
	CU_TRY(cudaMemcpy2DToArray(ctx->arrApTex[slot], 0, 0, luma, pitch, pitch, h,
	                           cudaMemcpyHostToDevice), "memcpy(apTex)");
	return 0;
}

extern "C" int zluxGpuUploadFields(ZluxGpuContext* ctx, const ZluxGatherParams* P,
                                   const ZluxGatherFields* F)
{
	ClearErr();
	if (!ctx || !P || !F) return 1;
	const int w = P->cache_w, h = P->cache_h;
	const size_t n = (size_t)P->width * P->height;

	// Refuse frames that would take an unreasonable share of the card. Without
	// this, a large comp silently starves After Effects' own GPU pipeline, which
	// then reports "GPU out of memory" for effects that are not even ours. The
	// caller treats a failure here as "use the CPU gather", which is correct.
	{
		size_t freeB = 0, totalB = 0;
		if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
			const size_t need = n * (4 * sizeof(float4) + 4 * sizeof(float)
			                         + 2 * sizeof(short2) + sizeof(float))
			                  + n * 4 * sizeof(float) * 2;   // arrays + pyramid headroom
			if (need > freeB / 2) {
				std::snprintf(g_last_error, sizeof(g_last_error),
				              "frame needs %zu MB but only %zu MB free; leaving it to the CPU",
				              need >> 20, freeB >> 20);
				g_has_error = true;
				return 1;
			}
		}
	}

	if (ctx->arr_w != P->cache_w || ctx->arr_h != P->cache_h) {
		DestroyFields(ctx);
		ctx->arr_w = P->cache_w; ctx->arr_h = P->cache_h;
	}
	if (ctx->alloc_px < n) {
		cudaFree(ctx->d_farRadius);   ctx->d_farRadius = nullptr;
		cudaFree(ctx->d_nearRadius);  ctx->d_nearRadius = nullptr;
		cudaFree(ctx->d_bleedRadius); ctx->d_bleedRadius = nullptr;
		cudaFree(ctx->d_centerDepth); ctx->d_centerDepth = nullptr;
		cudaFree(ctx->d_farRGBA);     ctx->d_farRGBA = nullptr;
		cudaFree(ctx->d_nearRGBA);    ctx->d_nearRGBA = nullptr;
		cudaFree(ctx->d_bleedRGBA);   ctx->d_bleedRGBA = nullptr;
		cudaFree(ctx->d_mattes);      ctx->d_mattes = nullptr;
		cudaFree(ctx->d_seedA);       ctx->d_seedA = nullptr;
		cudaFree(ctx->d_seedB);       ctx->d_seedB = nullptr;
		cudaFree(ctx->d_disc);        ctx->d_disc = nullptr;
		CU_TRY(cudaMalloc(&ctx->d_farRadius,   n * sizeof(float)),  "cudaMalloc(farRadius)");
		CU_TRY(cudaMalloc(&ctx->d_nearRadius,  n * sizeof(float)),  "cudaMalloc(nearRadius)");
		CU_TRY(cudaMalloc(&ctx->d_bleedRadius, n * sizeof(float)),  "cudaMalloc(bleedRadius)");
		CU_TRY(cudaMalloc(&ctx->d_centerDepth, n * sizeof(float)),  "cudaMalloc(centerDepth)");
		CU_TRY(cudaMalloc(&ctx->d_farRGBA,     n * sizeof(float4)), "cudaMalloc(farRGBA)");
		CU_TRY(cudaMalloc(&ctx->d_nearRGBA,    n * sizeof(float4)), "cudaMalloc(nearRGBA)");
		CU_TRY(cudaMalloc(&ctx->d_bleedRGBA,   n * sizeof(float4)), "cudaMalloc(bleedRGBA)");
		CU_TRY(cudaMalloc(&ctx->d_mattes,      n * sizeof(float4)), "cudaMalloc(mattes)");
		cudaFreeHost(ctx->h_farRGBA);   ctx->h_farRGBA = nullptr;
		cudaFreeHost(ctx->h_nearRGBA);  ctx->h_nearRGBA = nullptr;
		cudaFreeHost(ctx->h_bleedRGBA); ctx->h_bleedRGBA = nullptr;
		cudaFreeHost(ctx->h_mattes);    ctx->h_mattes = nullptr;
		CU_TRY(cudaHostAlloc(&ctx->h_farRGBA,   n * sizeof(float4), cudaHostAllocDefault), "cudaHostAlloc(far)");
		CU_TRY(cudaHostAlloc(&ctx->h_nearRGBA,  n * sizeof(float4), cudaHostAllocDefault), "cudaHostAlloc(near)");
		CU_TRY(cudaHostAlloc(&ctx->h_bleedRGBA, n * sizeof(float4), cudaHostAllocDefault), "cudaHostAlloc(bleed)");
		CU_TRY(cudaHostAlloc(&ctx->h_mattes,    n * sizeof(float4), cudaHostAllocDefault), "cudaHostAlloc(mattes)");
		ctx->alloc_px = n;
	}

	if (MakeFloatTex(&ctx->arrCoc, &ctx->texCoc, F->signed_coc, w, h, "coc texture")) return 1;
	ctx->hasDiscDist = (F->coc_disc_dist != nullptr);
	if (ctx->hasDiscDist) {
		if (MakeFloatTex(&ctx->arrDiscDist, &ctx->texDiscDist, F->coc_disc_dist, w, h,
		                 "disc-dist texture")) return 1;
	}

	// Null when zluxGpuBuildRadii will produce them on the device instead.
	if (F->far_radius)   CU_TRY(cudaMemcpy(ctx->d_farRadius,   F->far_radius,   n * sizeof(float), cudaMemcpyHostToDevice), "memcpy(farRadius)");
	if (F->near_radius)  CU_TRY(cudaMemcpy(ctx->d_nearRadius,  F->near_radius,  n * sizeof(float), cudaMemcpyHostToDevice), "memcpy(nearRadius)");
	if (F->bleed_radius) CU_TRY(cudaMemcpy(ctx->d_bleedRadius, F->bleed_radius, n * sizeof(float), cudaMemcpyHostToDevice), "memcpy(bleedRadius)");
	CU_TRY(cudaMemcpy(ctx->d_centerDepth, F->center_depth, n * sizeof(float), cudaMemcpyHostToDevice), "memcpy(centerDepth)");
	return 0;
}



extern "C" int zluxGpuBuildRadii(ZluxGpuContext* ctx, const ZluxGatherParams* P,
                                 const float* tile_min_coc, int tiles_x, int tiles_y,
                                 int tile_size, float px_per_coc, float uniform_base,
                                 float field_curvature, float field_sweet)
{
	ClearErr();
	if (!ctx || !P || !ctx->texCoc) return 1;
	const size_t ntiles = (size_t)tiles_x * tiles_y;
	if (!ctx->d_tileMinCoc || ctx->tile_cap < ntiles) {
		cudaFree(ctx->d_tileMinCoc);
		CU_TRY(cudaMalloc(&ctx->d_tileMinCoc, ntiles * sizeof(float)), "cudaMalloc(tileMinCoc)");
		ctx->tile_cap = ntiles;
	}
	CU_TRY(cudaMemcpy(ctx->d_tileMinCoc, tile_min_coc, ntiles * sizeof(float),
	                  cudaMemcpyHostToDevice), "memcpy(tileMinCoc)");

	const dim3 block(16, 16, 1);
	const dim3 grid((P->width + block.x - 1) / block.x,
	                (P->height + block.y - 1) / block.y, 1);
	zluxRadiiKernel<<<grid, block>>>(ctx->texCoc, ctx->d_tileMinCoc, *P,
	                                 tiles_x, tile_size, px_per_coc, uniform_base,
	                                 field_curvature, field_sweet,
	                                 ctx->d_farRadius, ctx->d_nearRadius, ctx->d_bleedRadius);
	CU_TRY(cudaGetLastError(), "radii kernel");
	return 0;
}

extern "C" int zluxGpuBuildDiscDist(ZluxGpuContext* ctx, const ZluxGatherParams* P)
{
	ClearErr();
	if (!ctx || !P || !ctx->texCoc) return 1;
	const int w = P->cache_w, h = P->cache_h;
	const size_t n = (size_t)w * h;

	if (!ctx->d_seedA) {
		CU_TRY(cudaMalloc(&ctx->d_seedA, ctx->alloc_px * sizeof(short2)), "cudaMalloc(seedA)");
		CU_TRY(cudaMalloc(&ctx->d_seedB, ctx->alloc_px * sizeof(short2)), "cudaMalloc(seedB)");
		CU_TRY(cudaMalloc(&ctx->d_disc,  ctx->alloc_px * sizeof(float)),  "cudaMalloc(disc)");
	}

	const dim3 block(16, 16, 1);
	const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

	zluxDiscSeedKernel<<<grid, block>>>(ctx->texCoc, w, h, ctx->d_seedA);
	CU_TRY(cudaGetLastError(), "disc seed kernel");

	// Flood from a bounded step: the gather's reach never exceeds 48 px, so
	// seeding the ladder at 64 covers everything that can matter and skips the
	// passes a full-image JFA would waste on unreachable distances.
	short2* src = ctx->d_seedA;
	short2* dst = ctx->d_seedB;
	for (int step = ZLUX_DISC_RANGE; step >= 1; step >>= 1) {
		zluxJfaStepKernel<<<grid, block>>>(src, dst, w, h, step);
		short2* t = src; src = dst; dst = t;
	}
	CU_TRY(cudaGetLastError(), "jfa step kernel");

	const float big = (float)(w + h);
	zluxDiscResolveKernel<<<grid, block>>>(src, ctx->d_disc, w, h, big);
	CU_TRY(cudaGetLastError(), "disc resolve kernel");

	// Publish into the texture the gather already samples, so nothing
	// downstream changes.
	if (!ctx->arrDiscDist) {
		cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
		CU_TRY(cudaMallocArray(&ctx->arrDiscDist, &ch, w, h), "cudaMallocArray(discDist)");
		cudaResourceDesc rd; std::memset(&rd, 0, sizeof(rd));
		rd.resType = cudaResourceTypeArray;
		rd.res.array.array = ctx->arrDiscDist;
		cudaTextureDesc td; std::memset(&td, 0, sizeof(td));
		td.addressMode[0] = cudaAddressModeClamp;
		td.addressMode[1] = cudaAddressModeClamp;
		td.filterMode     = cudaFilterModePoint;
		td.normalizedCoords = 0;
		td.readMode = cudaReadModeElementType;
		CU_TRY(cudaCreateTextureObject(&ctx->texDiscDist, &rd, &td, nullptr),
		       "cudaCreateTextureObject(discDist)");
	}
	CU_TRY(cudaMemcpy2DToArray(ctx->arrDiscDist, 0, 0, ctx->d_disc,
	                           (size_t)w * sizeof(float), (size_t)w * sizeof(float), h,
	                           cudaMemcpyDeviceToDevice),
	       "memcpy(disc -> array)");
	ctx->hasDiscDist = true;
	return 0;
}

extern "C" int zluxGpuGather(ZluxGpuContext* ctx, const ZluxGatherParams* P,
                             ZluxGatherOutputs* out, float* elapsed_ms)
{
	ClearErr();
	if (!ctx || !P || !out) return 1;

	const dim3 block(16, 16, 1);
	const dim3 grid((P->width + block.x - 1) / block.x,
	                (P->height + block.y - 1) / block.y, 1);

	if (elapsed_ms) cudaEventRecord(ctx->evStart);

	zluxGatherKernel<<<grid, block>>>(
		*P, ctx->texPyramid, ctx->texCoc, ctx->texDiscDist, ctx->hasDiscDist ? 1 : 0,
		ctx->texApTex[0], ctx->texApTex[1],
		ctx->d_hot, ctx->d_cold, ctx->d_descs, ctx->num_luts,
		ctx->d_farRadius, ctx->d_nearRadius, ctx->d_bleedRadius, ctx->d_centerDepth,
		ctx->d_farRGBA, ctx->d_nearRGBA, ctx->d_bleedRGBA, ctx->d_mattes);

	if (elapsed_ms) cudaEventRecord(ctx->evStop);
	CU_TRY(cudaGetLastError(), "kernel launch");
	CU_TRY(cudaDeviceSynchronize(), "kernel execution");
	if (elapsed_ms) cudaEventElapsedTime(elapsed_ms, ctx->evStart, ctx->evStop);

	// Readback into the page-locked staging buffers, then hand the caller those
	// pointers directly. The matte buffer is skipped entirely for opaque sources,
	// which is the common case and saves a quarter of the transfer.
	const size_t n = (size_t)P->width * P->height;
	CU_TRY(cudaMemcpy(ctx->h_farRGBA,   ctx->d_farRGBA,   n * sizeof(float4), cudaMemcpyDeviceToHost), "memcpy(farRGBA back)");
	CU_TRY(cudaMemcpy(ctx->h_nearRGBA,  ctx->d_nearRGBA,  n * sizeof(float4), cudaMemcpyDeviceToHost), "memcpy(nearRGBA back)");
	CU_TRY(cudaMemcpy(ctx->h_bleedRGBA, ctx->d_bleedRGBA, n * sizeof(float4), cudaMemcpyDeviceToHost), "memcpy(bleedRGBA back)");
	if (P->has_alpha) {
		CU_TRY(cudaMemcpy(ctx->h_mattes, ctx->d_mattes, n * sizeof(float4), cudaMemcpyDeviceToHost), "memcpy(mattes back)");
	}

	out->far_rgba   = reinterpret_cast<float*>(ctx->h_farRGBA);
	out->near_rgba  = reinterpret_cast<float*>(ctx->h_nearRGBA);
	out->bleed_rgba = reinterpret_cast<float*>(ctx->h_bleedRGBA);
	out->mattes     = reinterpret_cast<float*>(ctx->h_mattes);
	return 0;
}
