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

	int alloc_w = 0, alloc_h = 0;

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

	const float anam = fmaxf(0.1f, P.anamorphic_ratio);
	const float aniso = fmaxf(anam, 1.0f / anam);      // no astigmatism in stage 1
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

		if (vs.w < 0.001f) continue;

		// Rotate the baked unit-disc offset by the per-pixel blue-noise angle.
		const float kx_r = vs.x * cos_bn - vs.y * sin_bn;
		const float ky_r = vs.x * sin_bn + vs.y * cos_bn;
		float off_u = kx_r * pos_cx;
		float off_v = ky_r * pos_cy;

		float su = u + off_u;
		float sv = v + off_v;
		if (jitter) {
			const float h1 = fmodf(fmaf((float)i, 0.7548776662f, ign), 1.0f);
			const float h2 = fmodf(fmaf((float)i, 0.5698402910f, ign * 1.3247179572f), 1.0f);
			su = fmaf(h1 - 0.5f, jit_amt_u, su);
			sv = fmaf(h2 - 0.5f, jit_amt_v, sv);
		}

		float mask = vs.w;

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

		acc.x = fmaf(cr, w, acc.x);
		acc.y = fmaf(cg, w, acc.y);
		acc.z = fmaf(cb, w, acc.z);
		w_sum += w;
		if (do_alpha) a_acc = fmaf(ca_a, w, a_acc);

		if (has_scatter) {
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

	if (has_scatter) {
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

	GatherPassD<0>(P, texPyr, texCoc, texDisc, hasDisc != 0, hot, cold, descs, num_luts,
	               u, v, farRadius[idx], depth, coc, far_rgbw, far_matte);

	GatherPassD<1>(P, texPyr, texCoc, texDisc, hasDisc != 0, hot, cold, descs, num_luts,
	               u, v, nearRadius[idx], depth, coc, near_rgbw, near_matte);

	// Far bleed-over probe. Same Far gather, its own (usually zero) radius, so
	// pixels the composite does not need it for exit immediately.
	GatherPassD<0>(P, texPyr, texCoc, texDisc, hasDisc != 0, hot, cold, descs, num_luts,
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
                                int has_iris_mod, float astigmatism)
{
	if (!P) return 0;
	// Paths that need to sample an AE layer per tap are not ported yet; the
	// aperture mask for those cannot be baked into static_mask because it
	// depends on the sampled source position.
	if (P->aperture_shape_mode == 4) return 0;   // custom aperture texture
	if (has_aperture_tex || has_iris_mod) return 0;
	// Astigmatism drives the anisotropic multi-tap path, also unported.
	if (astigmatism > 0.001f) return 0;
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

extern "C" int zluxGpuUploadFields(ZluxGpuContext* ctx, const ZluxGatherParams* P,
                                   const ZluxGatherFields* F)
{
	ClearErr();
	if (!ctx || !P || !F) return 1;
	const int w = P->cache_w, h = P->cache_h;
	const size_t n = (size_t)P->width * P->height;

	if (ctx->alloc_w != P->width || ctx->alloc_h != P->height) {
		DestroyFields(ctx);
		cudaFree(ctx->d_farRadius);   ctx->d_farRadius = nullptr;
		cudaFree(ctx->d_nearRadius);  ctx->d_nearRadius = nullptr;
		cudaFree(ctx->d_bleedRadius); ctx->d_bleedRadius = nullptr;
		cudaFree(ctx->d_centerDepth); ctx->d_centerDepth = nullptr;
		cudaFree(ctx->d_farRGBA);     ctx->d_farRGBA = nullptr;
		cudaFree(ctx->d_nearRGBA);    ctx->d_nearRGBA = nullptr;
		cudaFree(ctx->d_bleedRGBA);   ctx->d_bleedRGBA = nullptr;
		cudaFree(ctx->d_mattes);      ctx->d_mattes = nullptr;
		CU_TRY(cudaMalloc(&ctx->d_farRadius,   n * sizeof(float)),  "cudaMalloc(farRadius)");
		CU_TRY(cudaMalloc(&ctx->d_nearRadius,  n * sizeof(float)),  "cudaMalloc(nearRadius)");
		CU_TRY(cudaMalloc(&ctx->d_bleedRadius, n * sizeof(float)),  "cudaMalloc(bleedRadius)");
		CU_TRY(cudaMalloc(&ctx->d_centerDepth, n * sizeof(float)),  "cudaMalloc(centerDepth)");
		CU_TRY(cudaMalloc(&ctx->d_farRGBA,     n * sizeof(float4)), "cudaMalloc(farRGBA)");
		CU_TRY(cudaMalloc(&ctx->d_nearRGBA,    n * sizeof(float4)), "cudaMalloc(nearRGBA)");
		CU_TRY(cudaMalloc(&ctx->d_bleedRGBA,   n * sizeof(float4)), "cudaMalloc(bleedRGBA)");
		CU_TRY(cudaMalloc(&ctx->d_mattes,      n * sizeof(float4)), "cudaMalloc(mattes)");
		ctx->alloc_w = P->width; ctx->alloc_h = P->height;
	}

	if (MakeFloatTex(&ctx->arrCoc, &ctx->texCoc, F->signed_coc, w, h, "coc texture")) return 1;
	ctx->hasDiscDist = (F->coc_disc_dist != nullptr);
	if (ctx->hasDiscDist) {
		if (MakeFloatTex(&ctx->arrDiscDist, &ctx->texDiscDist, F->coc_disc_dist, w, h,
		                 "disc-dist texture")) return 1;
	}

	CU_TRY(cudaMemcpy(ctx->d_farRadius,   F->far_radius,   n * sizeof(float), cudaMemcpyHostToDevice), "memcpy(farRadius)");
	CU_TRY(cudaMemcpy(ctx->d_nearRadius,  F->near_radius,  n * sizeof(float), cudaMemcpyHostToDevice), "memcpy(nearRadius)");
	CU_TRY(cudaMemcpy(ctx->d_bleedRadius, F->bleed_radius, n * sizeof(float), cudaMemcpyHostToDevice), "memcpy(bleedRadius)");
	CU_TRY(cudaMemcpy(ctx->d_centerDepth, F->center_depth, n * sizeof(float), cudaMemcpyHostToDevice), "memcpy(centerDepth)");
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
		ctx->d_hot, ctx->d_cold, ctx->d_descs, ctx->num_luts,
		ctx->d_farRadius, ctx->d_nearRadius, ctx->d_bleedRadius, ctx->d_centerDepth,
		ctx->d_farRGBA, ctx->d_nearRGBA, ctx->d_bleedRGBA, ctx->d_mattes);

	if (elapsed_ms) cudaEventRecord(ctx->evStop);
	CU_TRY(cudaGetLastError(), "kernel launch");
	CU_TRY(cudaDeviceSynchronize(), "kernel execution");
	if (elapsed_ms) cudaEventElapsedTime(elapsed_ms, ctx->evStart, ctx->evStop);

	const size_t n = (size_t)P->width * P->height;
	if (out->far_rgba)  CU_TRY(cudaMemcpy(out->far_rgba,  ctx->d_farRGBA,  n * sizeof(float4), cudaMemcpyDeviceToHost), "memcpy(farRGBA back)");
	if (out->near_rgba) CU_TRY(cudaMemcpy(out->near_rgba, ctx->d_nearRGBA, n * sizeof(float4), cudaMemcpyDeviceToHost), "memcpy(nearRGBA back)");
	if (out->bleed_rgba) CU_TRY(cudaMemcpy(out->bleed_rgba, ctx->d_bleedRGBA, n * sizeof(float4), cudaMemcpyDeviceToHost), "memcpy(bleedRGBA back)");
	if (out->mattes)    CU_TRY(cudaMemcpy(out->mattes,    ctx->d_mattes,   n * sizeof(float4), cudaMemcpyDeviceToHost), "memcpy(mattes back)");
	return 0;
}
