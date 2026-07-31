#include "zluxDOF.h"

// ── Version-field guards ────────────────────────────────────────────────────
// AE packs the version into fixed-width bit fields, and the two places that
// encode it disagree on overflow: PF_VERSION() masks, while the PiPL's
// ZLUX_PIPL_VERSION is plain arithmetic that carries into the next field. A
// MINOR_VERSION of 28 therefore shipped as "code 2.12 / PiPL 3.12" and AE threw
// a version-mismatch dialog. Fail the build instead of shipping that again.
static_assert(MAJOR_VERSION >= 0 && MAJOR_VERSION <= 7,
              "MAJOR_VERSION must fit PF_Vers_VERS_BITS (0..7)");
static_assert(MINOR_VERSION >= 0 && MINOR_VERSION <= 15,
              "MINOR_VERSION must fit PF_Vers_SUBVERS_BITS (0..15) -- bump MAJOR and reset MINOR to 0");
static_assert(BUG_VERSION >= 0 && BUG_VERSION <= 15,
              "BUG_VERSION must fit PF_Vers_BUGFIX_BITS (0..15)");
static_assert(BUILD_VERSION >= 0 && BUILD_VERSION <= 511,
              "BUILD_VERSION must fit PF_Vers_BUILD_BITS (0..511)");
static_assert(ZLUX_STAGE_NUM >= 0 && ZLUX_STAGE_NUM <= 3,
              "ZLUX_STAGE_NUM must fit PF_Vers_STAGE_BITS (0..3)");
// The whole point: the arithmetic the PiPL uses must equal what the code
// reports, or AE compares the two and refuses to load cleanly.
static_assert(ZLUX_PIPL_VERSION ==
              PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, ZLUX_STAGE_NUM, BUILD_VERSION),
              "ZLUX_PIPL_VERSION and PF_VERSION() disagree -- a version field has overflowed");

// CUDA gather. Defined when the build links zluxDOF_Kernel.cu; without it the
// renderer compiles and runs exactly as before on the CPU path.
#ifdef ZLUX_CUDA
#include "zluxDOF_Kernel.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef AE_OS_WIN
#include <shlwapi.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "shlwapi.lib")
#endif

#include "adobesdk/DrawbotSuite.h"

// ────────────────────────────────────────────────────────────────────────────
// Banner state. A single decoded PNG buffer (BGRA, pre-multiplied) is cached
// for the lifetime of the plugin and drawn by the custom-UI event handler.
// If the PNG file can't be located, HandleEvent falls back to a stylised
// "zluxDOF · by zlux · Zluxia" brand bar drawn with DRAWBOT primitives.
// ────────────────────────────────────────────────────────────────────────────
namespace zlux_banner {

struct BannerImage {
	std::vector<uint8_t> pixels_bgra;  // row-major, rowbytes = width * 4
	int width = 0;
	int height = 0;
	bool load_attempted = false;
	bool load_ok = false;
};

static BannerImage g_banner;
#ifdef AE_OS_WIN
static ULONG_PTR g_gdiplus_token = 0;
static bool g_gdiplus_ready = false;
static HMODULE GetPluginModule()
{
	HMODULE h = nullptr;
	::GetModuleHandleExW(
		GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		reinterpret_cast<LPCWSTR>(&GetPluginModule),
		&h);
	return h;
}

static void EnsureGdiPlus()
{
	if (g_gdiplus_ready) return;
	Gdiplus::GdiplusStartupInput startup{};
	if (Gdiplus::GdiplusStartup(&g_gdiplus_token, &startup, nullptr) == Gdiplus::Ok) {
		g_gdiplus_ready = true;
	}
}

static void ShutdownGdiPlus()
{
	if (!g_gdiplus_ready) return;
	Gdiplus::GdiplusShutdown(g_gdiplus_token);
	g_gdiplus_ready = false;
	g_gdiplus_token = 0;
}

// Build a list of candidate absolute paths where the banner PNG might live.
// Order (most-specific first):
//   1. ZLUXDOF_BANNER_PATH environment variable (explicit override)
//   2. <plugin dir>/zluxDOF_banner.png        (next to the .aex)
//   3. <plugin dir>/Zlux/zluxDOF_banner.png   (MediaCore layout)
static std::vector<std::wstring> BannerCandidatePaths()
{
	std::vector<std::wstring> out;
	wchar_t env[MAX_PATH * 2] = {};
	DWORD n = ::GetEnvironmentVariableW(L"ZLUXDOF_BANNER_PATH", env, MAX_PATH * 2);
	if (n > 0 && n < MAX_PATH * 2) {
		out.emplace_back(env);
	}

	wchar_t dll_path[MAX_PATH] = {};
	HMODULE h = GetPluginModule();
	if (h && ::GetModuleFileNameW(h, dll_path, MAX_PATH) != 0) {
		::PathRemoveFileSpecW(dll_path);
		std::wstring dir(dll_path);
		out.emplace_back(dir + L"\\zluxDOF_banner.png");
		out.emplace_back(dir + L"\\Zlux\\zluxDOF_banner.png");
	}
	return out;
}

// Shared GDI+ bitmap -> premultiplied BGRA8 conversion.
static bool ConvertGdiplusBitmap(Gdiplus::Bitmap& src, BannerImage& out)
{
	if (src.GetLastStatus() != Gdiplus::Ok) return false;

	const UINT w = src.GetWidth();
	const UINT h = src.GetHeight();
	if (w == 0 || h == 0 || w > 4096 || h > 4096) return false;

	Gdiplus::Rect rect(0, 0, static_cast<INT>(w), static_cast<INT>(h));
	Gdiplus::BitmapData bmp_data{};
	if (src.LockBits(&rect, Gdiplus::ImageLockModeRead,
	                 PixelFormat32bppPARGB, &bmp_data) != Gdiplus::Ok) {
		return false;
	}

	out.width = static_cast<int>(w);
	out.height = static_cast<int>(h);
	out.pixels_bgra.assign(static_cast<size_t>(w) * h * 4, 0);
	for (UINT y = 0; y < h; ++y) {
		const uint8_t* src_row = static_cast<const uint8_t*>(bmp_data.Scan0) +
		                         static_cast<INT_PTR>(y) * bmp_data.Stride;
		uint8_t* dst_row = out.pixels_bgra.data() + static_cast<size_t>(y) * w * 4;
		std::memcpy(dst_row, src_row, static_cast<size_t>(w) * 4);
	}
	src.UnlockBits(&bmp_data);
	return true;
}

// Decodes a PNG/BMP file into a BGRA8 buffer via GDI+.
static bool DecodePngFile(const std::wstring& path, BannerImage& out)
{
	EnsureGdiPlus();
	if (!g_gdiplus_ready) return false;

	Gdiplus::Bitmap src(path.c_str(), FALSE);
	return ConvertGdiplusBitmap(src, out);
}

// Decodes an image embedded in the .aex itself as an RCDATA resource (see
// zluxDOF_Assets.rc: ZLUXBANNER / ZLUXPICKER / APMAP01..APMAP80). Since
// v2.11 every runtime asset ships INSIDE the plugin binary, so the installed
// plugin is a single self-contained file; the on-disk paths remain as
// optional user overrides.
static bool DecodeImageResource(const wchar_t* res_name, BannerImage& out)
{
	EnsureGdiPlus();
	if (!g_gdiplus_ready) return false;

	HMODULE mod = GetPluginModule();
	if (!mod) return false;
	HRSRC rsrc = ::FindResourceW(mod, res_name, MAKEINTRESOURCEW(10) /* RT_RCDATA */);
	if (!rsrc) return false;
	HGLOBAL hglob = ::LoadResource(mod, rsrc);
	const DWORD size = ::SizeofResource(mod, rsrc);
	if (!hglob || size == 0) return false;
	const void* data = ::LockResource(hglob);
	if (!data) return false;

	IStream* stream = ::SHCreateMemStream(static_cast<const BYTE*>(data), size);
	if (!stream) return false;
	bool ok = false;
	{
		// Scoped so the Bitmap releases its stream reference before Release().
		Gdiplus::Bitmap src(stream, FALSE);
		ok = ConvertGdiplusBitmap(src, out);
	}
	stream->Release();
	return ok;
}
#endif  // AE_OS_WIN

static void EnsureBannerLoaded()
{
	if (g_banner.load_attempted) return;
	g_banner.load_attempted = true;
#ifdef AE_OS_WIN
	for (const std::wstring& p : BannerCandidatePaths()) {
		if (DecodePngFile(p, g_banner)) {
			g_banner.load_ok = true;
			return;
		}
	}
	// Embedded copy inside the .aex -- the normal path for a single-file
	// install (the disk paths above act as user overrides).
	if (DecodeImageResource(L"ZLUXBANNER", g_banner)) {
		g_banner.load_ok = true;
		return;
	}
#endif
}

static void ReleaseBanner()
{
	g_banner.pixels_bgra.clear();
	g_banner.pixels_bgra.shrink_to_fit();
	g_banner.width = 0;
	g_banner.height = 0;
	g_banner.load_attempted = false;
	g_banner.load_ok = false;
#ifdef AE_OS_WIN
	ShutdownGdiPlus();
#endif
}

}  // namespace zlux_banner

// ── Aperture Map library ───────────────────────────────────────────────────
// Loads one of the 80 grayscale iris shapes shipped in aperture_lib/ (white =
// open aperture, black = blocked) and caches it as a normalized float mask.
// The selected shape is baked into every Vogel sample's aperture mask in
// FinalizeVogelLUT, giving the DOF PRO "aperture map library" feature without
// the user having to load each shape as a layer.
namespace zlux_apmap {

// Immutable once built. That immutability is the whole point: the previous
// version was ONE process-global cache with a single loaded_index, refilled in
// place at the top of RenderCore -- which meant that with
// PF_OutFlag2_SUPPORTS_THREADED_RENDERING set, a second effect instance using a
// different map would `gray.clear()` the vector while another render thread was
// midway through reading gray.data() in the gather. That is a use-after-free,
// and it reallocated on every frame for as long as both instances rendered.
//
// Now a load produces a fresh ApMap that nothing mutates afterwards, and every
// consumer holds a shared_ptr to it for the duration of its frame. Readers
// therefore cannot have the map pulled out from under them, and two instances
// on different maps each keep their own rather than fighting over one slot.
struct ApMap {
	std::vector<float> gray;   // w*h, transmission in [0,1] (level 0)
	// Box-filtered mip chain (level k stored at mips[k-1], dims mip_w/mip_h).
	// The gather bakes this map at N discrete Vogel points per LUT; sampling
	// the full-res texture there aliases its fine structure into fingerprint
	// moiré rings on every bokeh disc (and Bokeh Gamma amplifies the per-tap
	// weight contrast, making the rings pop). Each LUT samples the level
	// whose texel pitch matches its inter-sample spacing instead.
	std::vector<std::vector<float>> mips;
	std::vector<int> mip_w;
	std::vector<int> mip_h;
	int w = 0;
	int h = 0;
};

using ApMapRef = std::shared_ptr<const ApMap>;

// Builds the box-filtered mip chain down to ~8px. Called once per map load,
// before the map is published to any reader.
static void BuildApMapMips(ApMap& m)
{
	m.mips.clear();
	m.mip_w.clear();
	m.mip_h.clear();
	if (m.gray.empty()) return;
	const float* prev = m.gray.data();
	int pw = m.w;
	int ph = m.h;
	while (pw >= 16 && ph >= 16) {
		const int nw = pw / 2;
		const int nh = ph / 2;
		std::vector<float> lvl(static_cast<size_t>(nw) * nh);
		for (int y = 0; y < nh; ++y) {
			for (int x = 0; x < nw; ++x) {
				const size_t s = (static_cast<size_t>(y) * 2) * pw + static_cast<size_t>(x) * 2;
				lvl[static_cast<size_t>(y) * nw + x] =
					(prev[s] + prev[s + 1] + prev[s + pw] + prev[s + pw + 1]) * 0.25f;
			}
		}
		m.mips.emplace_back(std::move(lvl));
		m.mip_w.push_back(nw);
		m.mip_h.push_back(nh);
		prev = m.mips.back().data();
		pw = nw;
		ph = nh;
	}
}

#ifdef AE_OS_WIN
static std::vector<std::wstring> ApMapPaths(int index)
{
	std::vector<std::wstring> out;
	wchar_t name[64] = {};
	swprintf(name, 64, L"aperture%02d.bmp", index);
	wchar_t dll_path[MAX_PATH] = {};
	HMODULE h = zlux_banner::GetPluginModule();
	if (h && ::GetModuleFileNameW(h, dll_path, MAX_PATH) != 0) {
		::PathRemoveFileSpecW(dll_path);
		std::wstring dir(dll_path);
		out.emplace_back(dir + L"\\aperture_lib\\" + name);
		out.emplace_back(dir + L"\\Zlux\\aperture_lib\\" + name);
	}
#ifdef _DEBUG
	// Dev fallback: the source tree's aperture_lib. Debug builds only -- a
	// personal filesystem path must never ship inside the release binary
	// (it is trivially visible to `strings` and leaks the dev machine).
	out.emplace_back(std::wstring(L"C:\\Users\\zlux\\Desktop\\zluxDOF\\aperture_lib\\") + name);
#endif
	return out;
}
#endif

// Converts a decoded BGRA image into a grayscale transmission mask
// (shared by the file and the embedded-resource load paths).
static void AdoptApMapImage(const zlux_banner::BannerImage& tmp, ApMap& m)
{
	m.w = tmp.width;
	m.h = tmp.height;
	m.gray.assign(static_cast<size_t>(tmp.width) * tmp.height, 0.0f);
	for (size_t i = 0; i < m.gray.size(); ++i) {
		const uint8_t* px = &tmp.pixels_bgra[i * 4];   // BGRA
		m.gray[i] = (px[0] * 0.114f + px[1] * 0.587f + px[2] * 0.299f) * (1.0f / 255.0f);
	}
	BuildApMapMips(m);
}

// Decodes aperture index (1..80) from disk or the embedded resources. Returns
// null when nothing could be loaded.
static ApMapRef DecodeApMap(int index)
{
	auto m = std::make_shared<ApMap>();
#ifdef AE_OS_WIN
	for (const std::wstring& p : ApMapPaths(index)) {
		zlux_banner::BannerImage tmp;
		if (zlux_banner::DecodePngFile(p, tmp)) {  // GDI+ loads BMP too
			AdoptApMapImage(tmp, *m);
			return m;
		}
	}
	// Embedded copy inside the .aex (APMAP01..APMAP80) -- the normal path
	// for a single-file install; disk files above act as user overrides.
	{
		wchar_t res[16] = {};
		swprintf(res, 16, L"APMAP%02d", index);
		zlux_banner::BannerImage tmp;
		if (zlux_banner::DecodeImageResource(res, tmp)) {
			AdoptApMapImage(tmp, *m);
			return m;
		}
	}
#endif
	return nullptr;
}

// Returns the map for aperture index (1..80), decoding it once and caching it
// thereafter. index<=0 means "no map". The returned reference keeps the map
// alive for as long as the caller holds it, which is what makes the readers
// below safe to run concurrently with another instance loading a different map.
//
// Failures are cached as null so a missing file is not re-probed every frame.
static ApMapRef LoadApMap(int index)
{
	if (index <= 0) return nullptr;

	static std::mutex m;
	static std::map<int, ApMapRef> cache;   // null value = "tried, not available"
	{
		std::lock_guard<std::mutex> lk(m);
		auto it = cache.find(index);
		if (it != cache.end()) return it->second;
	}

	// Decode OUTSIDE the lock: it is a GDI+ image decode, and holding the lock
	// across it would stall every other render thread that wants any map.
	ApMapRef decoded = DecodeApMap(index);

	std::lock_guard<std::mutex> lk(m);
	// Another thread may have won the race; keep whichever entry landed first so
	// all callers observe the same object.
	auto ins = cache.emplace(index, decoded);
	return ins.first->second;
}

inline bool Active(const ApMap* m) { return m && m->w > 0 && !m->gray.empty(); }

// Bilinear transmission sample at a normalized disc position (nx,ny in [-1,1]).
// Outside the unit square -> blocked. The map's +y row 0 is the top, while the
// disc's +y points down (AE screen space), so v is flipped.
inline PF_FpLong Sample(const ApMap* m, PF_FpLong nx, PF_FpLong ny)
{
	if (!Active(m)) return 1.0;
	const PF_FpLong u = nx * 0.5 + 0.5;
	const PF_FpLong v = ny * 0.5 + 0.5;
	if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) return 0.0;
	const PF_FpLong fx = u * (m->w - 1);
	const PF_FpLong fy = v * (m->h - 1);
	const int x0 = static_cast<int>(fx);
	const int y0 = static_cast<int>(fy);
	const int x1 = std::min(x0 + 1, m->w - 1);
	const int y1 = std::min(y0 + 1, m->h - 1);
	const PF_FpLong tx = fx - x0;
	const PF_FpLong ty = fy - y0;
	const float* g = m->gray.data();
	const PF_FpLong a = g[static_cast<size_t>(y0) * m->w + x0];
	const PF_FpLong b = g[static_cast<size_t>(y0) * m->w + x1];
	const PF_FpLong c = g[static_cast<size_t>(y1) * m->w + x0];
	const PF_FpLong d = g[static_cast<size_t>(y1) * m->w + x1];
	return (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty;
}

// Bilinear sample at mip level (0 = full res). Used by FinalizeVogelLUT so the
// per-sample bake reads a prefiltered texel matched to the LUT's density.
inline PF_FpLong SampleLevel(const ApMap* m, int level, PF_FpLong nx, PF_FpLong ny)
{
	if (!Active(m)) return 1.0;
	if (level <= 0 || m->mips.empty()) return Sample(m, nx, ny);
	const int li = std::min(level - 1, static_cast<int>(m->mips.size()) - 1);
	const int lw = m->mip_w[static_cast<size_t>(li)];
	const int lh = m->mip_h[static_cast<size_t>(li)];
	const PF_FpLong u = nx * 0.5 + 0.5;
	const PF_FpLong v = ny * 0.5 + 0.5;
	if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) return 0.0;
	const PF_FpLong fx = u * (lw - 1);
	const PF_FpLong fy = v * (lh - 1);
	const int x0 = static_cast<int>(fx);
	const int y0 = static_cast<int>(fy);
	const int x1 = std::min(x0 + 1, lw - 1);
	const int y1 = std::min(y0 + 1, lh - 1);
	const PF_FpLong tx = fx - x0;
	const PF_FpLong ty = fy - y0;
	const float* g = m->mips[static_cast<size_t>(li)].data();
	const PF_FpLong a = g[static_cast<size_t>(y0) * lw + x0];
	const PF_FpLong b = g[static_cast<size_t>(y0) * lw + x1];
	const PF_FpLong c = g[static_cast<size_t>(y1) * lw + x0];
	const PF_FpLong d = g[static_cast<size_t>(y1) * lw + x1];
	return (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty;
}

// Mip level whose texel pitch tracks the Vogel inter-sample spacing for an
// N-tap LUT (texel ≈ half the spacing, so a bilinear tap integrates roughly
// one sample cell of the map). With ~256 taps over a 256px map the spacing is
// ~14px -- sampled at full res that aliases the map's fine texture into
// fingerprint moiré on every disc; level 3 (8px texels) integrates it away.
inline int PickLevelForSamples(const ApMap* m, A_long n)
{
	if (!Active(m) || m->mips.empty()) return 0;
	const double spacing = static_cast<double>(m->w) * 0.886
	                     / std::sqrt(static_cast<double>(n > 1 ? n : 1));
	const double lf = std::log2(spacing * 0.5 > 1.0 ? spacing * 0.5 : 1.0);
	int L = static_cast<int>(std::lround(lf));
	const int max_l = static_cast<int>(m->mips.size());
	if (L < 0) L = 0;
	if (L > max_l) L = max_l;
	return L;
}

// Clickable picker montage (all 80 shapes in an 8x10 grid) drawn in the custom
// UI control. Loaded once via the banner's GDI+ PNG decoder from
// <plugin>/aperture_lib/_picker.png.
static zlux_banner::BannerImage g_picker;
static void EnsurePickerLoaded()
{
	if (g_picker.load_attempted) return;
	g_picker.load_attempted = true;
#ifdef AE_OS_WIN
	std::vector<std::wstring> paths;
	wchar_t dll_path[MAX_PATH] = {};
	HMODULE h = zlux_banner::GetPluginModule();
	if (h && ::GetModuleFileNameW(h, dll_path, MAX_PATH) != 0) {
		::PathRemoveFileSpecW(dll_path);
		std::wstring dir(dll_path);
		paths.emplace_back(dir + L"\\aperture_lib\\_picker.png");
		paths.emplace_back(dir + L"\\Zlux\\aperture_lib\\_picker.png");
	}
#ifdef _DEBUG
	// Debug-only dev fallback (see ApMapPaths: no personal paths in release).
	paths.emplace_back(L"C:\\Users\\zlux\\Desktop\\zluxDOF\\aperture_lib\\_picker.png");
#endif
	for (const std::wstring& p : paths) {
		if (zlux_banner::DecodePngFile(p, g_picker)) { g_picker.load_ok = true; return; }
	}
	// Embedded copy inside the .aex (single-file install).
	if (zlux_banner::DecodeImageResource(L"ZLUXPICKER", g_picker)) {
		g_picker.load_ok = true;
		return;
	}
#endif
}

}  // namespace zlux_apmap

// ── Depth histogram (for the Levels custom-UI control) ──────────────────────
// Computed once per render from the NORMALIZED-but-not-remapped depth (the same
// d-axis the Black/White/Gamma controls operate on), stashed here so the custom
// UI draw -- which has no depth pixels of its own -- can plot it. Single writer
// (one render at a time per the panel that owns the control); a torn read is
// only ever a cosmetic one-frame glitch in a histogram, so no lock is needed.
namespace zlux_depthhist {
	constexpr int kBins = 256;
	static A_long g_bins[kBins] = { 0 };
	static A_long g_max = 1;
	static bool   g_valid = false;
}

static void UnionLRect(const PF_LRect* src, PF_LRect* dst)
{
	if (src->left < dst->left) dst->left = src->left;
	if (src->top < dst->top) dst->top = src->top;
	if (src->right > dst->right) dst->right = src->right;
	if (src->bottom > dst->bottom) dst->bottom = src->bottom;
}

namespace {

constexpr PF_FpLong kPi = 3.14159265358979323846;
constexpr PF_FpLong kTau = 6.28318530717958647692;
constexpr PF_FpLong kGoldenAngle = 2.39996323;
constexpr PF_FpLong kCocClamp = 1.0;
constexpr PF_FpLong kEps = 1e-6;
constexpr A_long kCocTileSize = 16;

struct Vec2 {
	PF_FpLong x;
	PF_FpLong y;
};

struct Color3 {
	PF_FpLong r;
	PF_FpLong g;
	PF_FpLong b;
};

struct DOFSettings {
	A_long display_mode;
	// DoF is always the "Aperture (Simple)" model: a normalized inverse-Z CoC
	// driven by Blur Amount + Focus Distance. The depth map is always read as
	// Normalized (linear), white = near / black = far. The v2.21 "DoF Mode" and
	// "Depth Encoding" popups (plus the thin-lens Physical Lens controls) were
	// retired in v2.22 -- the single abstract model is what users actually dial.
	A_long depth_channel;
	PF_FpLong focal_distance;
	PF_FpLong blur_strength;
	PF_FpLong focus_range;
	PF_FpLong anamorphic_ratio;
	PF_FpLong highlight_boost;
	// Chromatic aberration as three independent opponent-colour axes (v2.12):
	// Red/Cyan, Green/Magenta, Blue/Yellow, each in [-1.2 .. 1.2]
	// (slider % * 0.012). The per-tap fringe scales with the tap's own
	// defocus and flips sign across the focal plane (LoCA behaviour).
	PF_FpLong ca_rc;
	PF_FpLong ca_gm;
	PF_FpLong ca_by;
	PF_FpLong ca_strength;   // max |axis| -- cheap has-CA gate
	// Global speed/quality governor (top-of-panel popup):
	//   1 = Fast (Preview) -- ~4-6x faster scrubbing: sample budget cut to
	//       ~40% (floor 16, cap 192) AND the entire far + near blur layers
	//       (colour, coverage alpha and the far bleed-over weight) are
	//       gathered once in a half-resolution pre-pass and bilinearly
	//       upsampled. Both layers are smooth bokeh by definition, so the
	//       2x upsample reads clean; the sharp in-focus layer still
	//       composites at full resolution.
	//   2 = Final          -- the reference pipeline, untouched.
	//   3 = Extreme (Hero) -- for huge Blur Amounts that must be flawless:
	//       sample budget doubled (up to the 1024 LUT max), the footprint-
	//       aware tap cap relaxed (1.6x -> 2.5x headroom), and the half-res
	//       pre-pass disabled so both blur layers are gathered per-pixel
	//       at full resolution.
	A_long render_mode;
	// P0: Energy-conserving (physical) gather. When TRUE the gather normalizes
	// colour by pure geometric coverage (mask·gate) instead of the brightness-
	// weighted sum -- no Bokeh Brightness Boost / Bokeh Gamma weighting in the
	// average. Matches a real lens / 3D-render defocus (a linear-light average of
	// the HDR source) and removes the high-variance weighting that fireflies at
	// strong boost. Punch then comes from Highlight Recovery (HDR) + Scatter.
	PF_Boolean energy_conserving;
	PF_FpLong vignetting;
	PF_FpLong vignetting_scale;
	PF_FpLong astigmatism;
	// Field curvature (v2.10): depth-independent edge blur (sweet-spot look).
	// Added to |signed CoC| in the depth precompute; see FieldCurvatureCoc.
	PF_FpLong field_curvature;   // 0..1
	PF_FpLong field_sweet;       // 0..1 protected centre radius
	PF_FpLong bokeh_rotation_rad;
	A_long aperture_blades;
	PF_FpLong catadioptric;
	PF_FpLong softness;
	Vec2 auto_focus_point;
	PF_Boolean enable_highlight;
	PF_Boolean astigmatism_type_sagittal;
	PF_Boolean depth_invert;
	PF_FpLong depth_blackpoint;
	PF_FpLong depth_whitepoint;
	PF_FpLong depth_gamma;       // remap curve on normalized depth (1 = linear)
	PF_FpLong depth_smoothing;   // 0..1; depth-guided smoothing radius on signed CoC (de-staircase, edge-preserving)
	A_long aperture_shape_mode;
	A_long aperture_map_index;   // 0 = off; 1..80 = built-in aperture_lib BMP
	// Procedural onion rings (v2.9): concentric machining grooves of a molded
	// aspheric element, visible as rings inside every bokeh disc (the classic
	// Helios 44 / modern-aspheric signature). Works on top of any iris shape
	// or aperture map. amount 0 disables; count = grooves across the radius.
	PF_FpLong onion_amount;      // 0..1
	PF_FpLong onion_count;       // 3..40
	PF_FpLong blade_curve;
	PF_FpLong notch_angle;
	PF_FpLong notch_scale;
	PF_FpLong spherical_aberration_amount;
	PF_FpLong spherical_aberration_scale;
	PF_FpLong highlights_low;
	PF_FpLong highlights_high;
	PF_FpLong highlights_softness;
	PF_FpLong highlights_saturation;
	// Non-linear gather exponent that concentrates gathered energy on
	// bright samples. 0 = linear (physical) gather. At >0 each sample's
	// contribution is weighted by pow(luma, bokeh_gamma), so specular
	// highlights are preserved as crisp bokeh cores instead of being
	// diffused into the surrounding mid-tones by the unbiased average.
	// Produces the "DOF PRO punchy bokeh" look through the same gather
	// without a separate pre-filter pass.
	PF_FpLong bokeh_gamma;
	// Specular-sprite scatter strength. 0 = pure gather (energy-conserving
	// weighted average). >0 adds an additive *scatter* contribution for
	// samples above the highlight threshold window -- those bright taps
	// accumulate into a separate un-normalised sprite buffer that is
	// layered on top of the gather result. Gives the DOF PRO "crisp
	// specular bokeh" signature: point lights keep their peak energy
	// through defocus instead of being averaged away, producing the
	// plump cinematic bokeh shape with a bright iris edge.
	PF_FpLong highlight_scatter;
	// 0 = Additive (un-normalised sprite layered on top, can clip),
	// 1 = Preservative (specular emphasis folded into the gather weights and
	// renormalised, so exposure and dynamic range are conserved).
	A_long highlight_mode;
	// Highlight clipping recovery: at capture time, 8/16bpc sources clamp
	// saturated specular peaks to 1.0 and lose the real HDR energy that
	// would otherwise give bokeh its DOF PRO punch. When this slider is
	// non-zero, PopulatePyramidLevel0 extrapolates the clipped pixels
	// back above 1.0 before the gather reads them, so point lights
	// disperse into bright, clearly-edged bokeh instead of muddy discs.
	PF_FpLong highlight_recovery;
	Color3 highlights_tint;
	PF_FpLong near_blur_factor;
	PF_FpLong foreground_protect; // 0..1; keeps sharp near details from being erased by foreground blur
	A_long sample_count;
	// v3.1: 0..1. How crisp the bokeh is allowed to be. The gather pre-averages
	// the source over a footprint floored at a fraction of the blur radius; that
	// floor is what softens a bokeh disc's edge. 0 reproduces the v3.0 look
	// (floor = 35% of the radius), 1 removes the floor entirely and lets the
	// Vogel inter-sample spacing be the only limit -- crisp, hard-edged discs at
	// the cost of needing the tap count to keep up. See the footprint-floor block
	// in GatherPass.
	PF_FpLong bokeh_definition;
	PF_FpLong matte_top;
	PF_FpLong matte_bottom;
	PF_FpLong matte_left;
	PF_FpLong matte_right;
	PF_FpLong aperture_texture_intensity; // 0..1
	PF_FpLong aperture_texture_scale;     // 0.25..4 (UV repeat/zoom)
	PF_FpLong aperture_texture_offset;    // -1..1, field-driven UV shift
	PF_Boolean aperture_texture_invert;
	PF_FpLong noise_amount;
	PF_Boolean noise_animated;
	PF_Boolean noise_monochromatic;
	A_long noise_luma_distribution;
	A_long noise_map_distribution;
	Color3 noise_tint;
	A_long current_time;
	PF_Boolean no_depth;

	// ── Depth auto-range normalization (computed per-frame in RenderCore) ──
	// Non-normalized depth inputs -- most importantly linear-Z EXR passes whose
	// values are not in [0,1] -- are remapped into [0,1] using a robust content
	// range so the Focus / Black Point / White Point controls operate in a
	// meaningful space instead of clamping everything past 1.0 to "far". For
	// conventional [0,1] maps auto-ranging is disabled and the legacy behavior
	// is preserved exactly. Set once per frame before any depth is sampled.
	PF_Boolean depth_autorange;
	PF_FpLong  depth_range_min;
	PF_FpLong  depth_range_inv_span; // 1 / (range_max - range_min)

	// Frame-constant LUT for the Bokeh Gamma highlight weight. The gather
	// applies pow(1 + luma, bokeh_gamma) per accepted tap; gamma is constant
	// across the frame and luma is in [0,1], so we precompute the curve once
	// and replace the per-tap pow() with a cheap lerp. 257 entries = 256
	// intervals over [0,1] plus the closing endpoint.
	PF_FpLong  bokeh_gamma_lut[257];
};

struct CoCTileData {
	PF_FpLong min_coc;
	PF_FpLong max_coc;
	PF_FpLong min_depth;
	PF_FpLong max_depth;
};

struct VogelSample {
	// Raw Vogel placement (rotation baked by BuildVogelLUT).
	PF_FpLong cos_a;
	PF_FpLong sin_a;
	PF_FpLong fr;

	// Frame-constant, per-sample data baked by FinalizeVogelLUT. These get
	// recomputed once when the LUT is built and are then read O(N_pix)
	// times inside the gather loop, avoiding a bag of trig / sqrt / pow
	// evaluations that would otherwise run per-tap per-pixel.
	PF_FpLong kx;          // cos_a * fr / anamorphic_ratio  (position scale x)
	PF_FpLong ky;          // sin_a * fr                      (position scale y)
	PF_FpLong norm_x;      // rotated normalized disc position (for masks)
	PF_FpLong norm_y;
	PF_FpLong static_mask; // baked polygonal × catadioptric × softness × matte
	PF_FpLong soft_edge;   // baked softness falloff (1 if no softness)
	// Baked spherical-aberration radial profiles (frame-constant: depend only on
	// fr and the frame's sharpness exponent). The gather picks one per pixel by
	// the sign of (aberration × plane_sign) and lerps with the strength, so the
	// per-tap std::pow() in ComputeSphericalProfile is eliminated.
	//   sa_pos = pow(fr,       sharpness)   (bright-edge / overcorrected)
	//   sa_neg = pow(1 - fr,   sharpness)   (bright-core / undercorrected)
	PF_FpLong sa_pos;
	PF_FpLong sa_neg;
};

// v3.1: raised from 1024. With the footprint floor gone (Bokeh Definition) the
// colour mip follows the Vogel inter-sample spacing, so the tap count is what
// sets how crisp a disc can get -- and Extreme mode's footprint-derived cap was
// clipping against the old 1024 ceiling on large bokeh. A LUT is
// kMaxVogelSamples * sizeof(VogelSample) (~180 KB at 2048); the whole ladder is
// a few MB, built once per frame.
static constexpr A_long kMaxVogelSamples = 2048;
struct VogelLUT {
	VogelSample samples[kMaxVogelSamples];
	A_long count;
};

inline void BuildVogelLUT(VogelLUT& lut, A_long N, PF_FpLong rotation)
{
	lut.count = std::min<A_long>(N, kMaxVogelSamples);
	const PF_FpLong inv_N = 1.0 / static_cast<PF_FpLong>(lut.count);
	for (A_long i = 0; i < lut.count; ++i) {
		const PF_FpLong fi = static_cast<PF_FpLong>(i) + 0.5;
		const PF_FpLong angle = fi * kGoldenAngle + rotation;
		lut.samples[i].fr = std::sqrt(fi * inv_N);
		lut.samples[i].cos_a = std::cos(angle);
		lut.samples[i].sin_a = std::sin(angle);
		// Baked fields are neutral until FinalizeVogelLUT runs.
		lut.samples[i].kx = 0.0;
		lut.samples[i].ky = 0.0;
		lut.samples[i].norm_x = 0.0;
		lut.samples[i].norm_y = 0.0;
		lut.samples[i].static_mask = 1.0;
		lut.samples[i].soft_edge = 1.0;
		lut.samples[i].sa_pos = 1.0;
		lut.samples[i].sa_neg = 1.0;
	}
}

template <typename T>
inline T ClampValue(T v, T lo, T hi)
{
	return std::max(lo, std::min(v, hi));
}

inline PF_FpLong Clamp01(PF_FpLong v)
{
	return ClampValue<PF_FpLong>(v, 0.0, 1.0);
}

inline PF_FpLong Mix(PF_FpLong a, PF_FpLong b, PF_FpLong t)
{
	return a + (b - a) * t;
}

inline PF_FpLong SmoothStep(PF_FpLong edge0, PF_FpLong edge1, PF_FpLong x)
{
	if (std::abs(edge1 - edge0) <= kEps) {
		return x < edge0 ? 0.0 : 1.0;
	}
	const PF_FpLong t = Clamp01((x - edge0) / (edge1 - edge0));
	return t * t * (3.0 - 2.0 * t);
}

inline PF_FpLong Luma(const Color3& c)
{
	return Clamp01(c.r * 0.299 + c.g * 0.587 + c.b * 0.114);
}

// Procedural onion-ring transmission at normalized disc radius r (0..1):
// concentric grooves carved by the diamond-turning of molded aspherics show
// up as alternating dark rings inside the bokeh disc. Grooves strengthen
// toward the rim (matching the real artifact) via the fade term. Shared by
// the FinalizeVogelLUT bake and the panel bokeh-preview renderer.
inline PF_FpLong OnionRingMask(PF_FpLong r, PF_FpLong amount, PF_FpLong count)
{
	// Narrow, deep grooves (cubed cosine lobe) rather than a soft sine: real
	// machining rings are thin crisp circles, and the soft profile washed out
	// to near-invisibility once the gather's reconstruction smoothed it.
	// NOTE: rings only ever imprint on sources SMALLER than the CoC disc
	// (point lights / speculars) -- an extended uniform source integrates the
	// whole textured pupil identically at every sensor point, so it shows no
	// rings through ANY real lens either.
	PF_FpLong ring = 0.5 + 0.5 * std::cos(r * count * kTau);
	ring = ring * ring * ring;
	const PF_FpLong fade = 0.25 + 0.75 * r;
	return Clamp01(1.0 - amount * 0.85 * ring * fade);
}

// ── Field curvature (lens-field blur) ───────────────────────────────────────
// A curved focal surface throws the frame edges out of focus REGARDLESS of
// subject depth: the sharp "sweet spot" centre with smeared edges of
// Lensbaby-style glass and wide-angle-on-macro-tube rigs. Returns the extra
// CoC at normalized frame position (u, v); `sweet` (0..1) is the protected
// centre radius, the ramp width is fixed so the falloff reads consistently.
// Combined with high Astigmatism the edge blur elongates into the radial /
// swirl streaks of the reference shots.
inline PF_FpLong FieldCurvatureCoc(PF_FpLong u, PF_FpLong v,
                                   PF_FpLong amount, PF_FpLong sweet)
{
	const PF_FpLong dx = u - 0.5;
	const PF_FpLong dy = v - 0.5;
	const PF_FpLong fr = std::sqrt(dx * dx + dy * dy) * 2.0; // 1.0 at edge midpoints
	const PF_FpLong t = SmoothStep(sweet, sweet + 0.55, fr);
	return amount * 0.35 * t * t;
}

inline PF_FpLong Length(const Vec2& v)
{
	return std::sqrt(v.x * v.x + v.y * v.y);
}

inline Vec2 Rotate(const Vec2& v, PF_FpLong angle)
{
	const PF_FpLong c = std::cos(angle);
	const PF_FpLong s = std::sin(angle);
	return {v.x * c - v.y * s, v.x * s + v.y * c};
}

// Split the row range [0, total) into contiguous chunks across the CPU cores
// and run body(row_begin, row_end) on each in its own thread. Because every
// row is owned by exactly one thread, any loop whose per-row work is
// independent (or whose only shared writes are partitioned so no two threads
// touch the same destination -- e.g. tile-aligned chunking) produces output
// identical to the equivalent serial loop. `min_chunk` keeps tiny ranges
// single-threaded so we never pay thread-spawn overhead for trivial work.
template <typename Fn>
inline void ParallelRows(A_long total, A_long min_chunk, Fn&& body)
{
	if (total <= 0) return;
	const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
	A_long nthreads = static_cast<A_long>(std::min<unsigned>(hw, 32u));
	if (nthreads <= 1 || total < std::max<A_long>(1, min_chunk) * 2) {
		body(0, total);
		return;
	}
	A_long chunks = std::min<A_long>(nthreads,
		std::max<A_long>(1, total / std::max<A_long>(1, min_chunk)));
	const A_long per = (total + chunks - 1) / chunks;
	std::vector<std::thread> pool;
	pool.reserve(static_cast<size_t>(chunks));
	// Crash-safety, two failure modes both fatal if left unhandled:
	//   1. A worker that lets an exception escape its entry function calls
	//      std::terminate() -- so the body is wrapped; nothing escapes a thread.
	//   2. std::thread construction can throw (OS thread/handle exhaustion under
	//      a busy host). If it does mid-loop, the half-filled `pool` of joinable
	//      threads would std::terminate on destruction during unwinding. We
	//      instead catch it, join what we spawned, and run the rest serially.
	A_long next = 0; // first row not yet assigned to a spawned worker
	try {
		for (A_long c = 0; c < chunks; ++c) {
			const A_long s = c * per;
			const A_long e = std::min<A_long>(total, s + per);
			if (s >= e) break;
			pool.emplace_back([&body, s, e]() {
				try { body(s, e); } catch (...) { /* never escape the thread */ }
			});
			next = e;
		}
	} catch (...) {
		// Thread spawn failed -- fall through, join the survivors, finish serially.
	}
	for (auto& th : pool) th.join();
	if (next < total) body(next, total);
}

template <typename PIX>
struct PixTraits;

template <>
struct PixTraits<PF_Pixel8> {
	static constexpr PF_FpLong max_chan = 255.0;
	static constexpr PF_FpLong inv_max = 1.0 / 255.0;
};

template <>
struct PixTraits<PF_Pixel16> {
	static constexpr PF_FpLong max_chan = 32768.0;
	static constexpr PF_FpLong inv_max = 1.0 / 32768.0;
};

template <>
struct PixTraits<PF_PixelFloat> {
	static constexpr PF_FpLong max_chan = 1.0;
	static constexpr PF_FpLong inv_max = 1.0;
};

template <typename PIX>
inline const PIX* PixelPtr(const PF_EffectWorld* world, A_long x, A_long y)
{
	const char* row = reinterpret_cast<const char*>(world->data) + y * world->rowbytes;
	return reinterpret_cast<const PIX*>(row) + x;
}

template <typename PIX>
inline PIX* PixelPtr(PF_EffectWorld* world, A_long x, A_long y)
{
	char* row = reinterpret_cast<char*>(world->data) + y * world->rowbytes;
	return reinterpret_cast<PIX*>(row) + x;
}

template <typename PIX>
inline Color3 ColorFromPix(const PIX& p)
{
	return {
		static_cast<PF_FpLong>(p.red) * PixTraits<PIX>::inv_max,
		static_cast<PF_FpLong>(p.green) * PixTraits<PIX>::inv_max,
		static_cast<PF_FpLong>(p.blue) * PixTraits<PIX>::inv_max
	};
}

template <typename PIX>
inline PIX PixFromColor(const Color3& c, PF_FpLong alpha_norm)
{
	PIX p{};
	if constexpr (std::is_same_v<PIX, PF_PixelFloat>) {
		// 32bpc is expected to carry HDR / scene-linear values -- clamping
		// would destroy legitimate over-range highlights that make bokeh
		// look rich. Pass the colour through untouched; let AE deal with
		// whatever display clamp it wants downstream.
		p.red   = static_cast<float>(c.r);
		p.green = static_cast<float>(c.g);
		p.blue  = static_cast<float>(c.b);
		p.alpha = static_cast<float>(alpha_norm);
	} else {
		p.red   = static_cast<decltype(p.red)>(Clamp01(c.r) * PixTraits<PIX>::max_chan);
		p.green = static_cast<decltype(p.green)>(Clamp01(c.g) * PixTraits<PIX>::max_chan);
		p.blue  = static_cast<decltype(p.blue)>(Clamp01(c.b) * PixTraits<PIX>::max_chan);
		p.alpha = static_cast<decltype(p.alpha)>(Clamp01(alpha_norm) * PixTraits<PIX>::max_chan);
	}
	return p;
}

inline A_long MirrorCoord(A_long c, A_long size)
{
	if (c < 0) c = -c - 1;
	const A_long period = size * 2;
	c = c % period;
	return (c >= size) ? (period - 1 - c) : c;
}

template <typename PIX>
inline Color3 SampleColorUV(const PF_EffectWorld* world, PF_FpLong u, PF_FpLong v)
{
	const A_long iw = world->width;
	const A_long ih = world->height;
	if (iw <= 1 || ih <= 1) {
		return ColorFromPix(*PixelPtr<PIX>(world, 0, 0));
	}
	const PF_FpLong px = u * static_cast<PF_FpLong>(iw) - 0.5;
	const PF_FpLong py = v * static_cast<PF_FpLong>(ih) - 0.5;
	const A_long fx0 = static_cast<A_long>(std::floor(px));
	const A_long fy0 = static_cast<A_long>(std::floor(py));
	const A_long x0 = MirrorCoord(fx0, iw);
	const A_long y0 = MirrorCoord(fy0, ih);
	const A_long x1 = MirrorCoord(fx0 + 1, iw);
	const A_long y1 = MirrorCoord(fy0 + 1, ih);
	const PF_FpLong fx = px - std::floor(px);
	const PF_FpLong fy = py - std::floor(py);
	const Color3 c00 = ColorFromPix(*PixelPtr<PIX>(world, x0, y0));
	const Color3 c10 = ColorFromPix(*PixelPtr<PIX>(world, x1, y0));
	const Color3 c01 = ColorFromPix(*PixelPtr<PIX>(world, x0, y1));
	const Color3 c11 = ColorFromPix(*PixelPtr<PIX>(world, x1, y1));
	const PF_FpLong w00 = (1.0 - fx) * (1.0 - fy);
	const PF_FpLong w10 = fx * (1.0 - fy);
	const PF_FpLong w01 = (1.0 - fx) * fy;
	const PF_FpLong w11 = fx * fy;
	return {
		c00.r * w00 + c10.r * w10 + c01.r * w01 + c11.r * w11,
		c00.g * w00 + c10.g * w10 + c01.g * w01 + c11.g * w11,
		c00.b * w00 + c10.b * w10 + c01.b * w01 + c11.b * w11
	};
}

// ── Source MIP pyramid ─────────────────────────────────────────────────────
//
// Gather-based DoF sampling becomes sparse when the CoC is large: 1024
// Vogel samples spread across a ~300 px disc leave several pixels between
// neighbouring taps, which manifests as "small circles" inside big bokeh
// and hides any sub-pixel structure of the aperture texture. The fix is a
// pre-filtered mip pyramid: each sample reads from the mip level whose
// texel footprint matches the inter-sample spacing, so the bilinear taps
// already average the right source neighbourhood -- no aliasing, smooth
// bokeh, and the aperture texture survives even at huge radii.
//
// The pyramid is stored as linear-light float RGB so gamma decoding happens
// once per source pixel (at build time) instead of thousands of times per
// output pixel (at gather time). Level 0 is the original source converted
// to linear; each subsequent level halves both dimensions via a 2x2 box
// filter, which is the exact box-prefilter that matches linear bilinear
// upsampling.
struct MipLevel {
	std::vector<float> data; // interleaved RGBA, linear light (A = linear coverage)
	// Per-texel blur magnitude |signed CoC|, same w*h (one float). Drives the
	// EDGE-AWARE downsample: when two children straddle a depth discontinuity
	// (sharp foreground vs blurred background), the more-defocused child wins the
	// parent texel, so a coarse mip near a silhouette holds CLEAN background
	// colour instead of a dark FG+BG average. That average was the dark edge halo
	// ("doesn't look like real DOF"); the sharp side is rendered from its own
	// in-focus path so losing it from the mip costs nothing. Empty => plain box
	// downsample (bit-identical to the old behaviour; safe fallback).
	std::vector<float> coc;
	A_long w = 0;
	A_long h = 0;
};

struct SourcePyramid {
	std::vector<MipLevel> levels;
	A_long num_levels = 0;
	// True if the source carries any non-opaque pixel. When false the entire
	// premultiplied-alpha blur path is skipped: the gather never accumulates
	// alpha and RenderPixelImpl outputs the (constant 1.0) original alpha, so
	// ordinary opaque footage stays bit-identical and pays nothing for the
	// matte-feathering machinery. Set once at build time.
	bool has_alpha = false;
};

// Forward-declared; defined a bit further down alongside the other
// world-introspection helpers.
inline bool WorldIsFloat(const PF_EffectWorld* world);

// The bit depth After Effects is driving THIS render at (8, 16 or 32 bpc), taken
// straight from PF_SmartRenderInput::bitdepth. 0 means "not told", which is the
// legacy non-smart entry point; WorldIsFloat then falls back to inspecting
// rowbytes. Every world AE hands an effect in one render -- input, output, depth
// layer, aperture layers -- is at this depth, so one value answers for all of
// them.
//
// Thread-local because the UI preview and the MFR render threads can be driven
// at different depths concurrently, and a single global would let one clobber
// the other mid-frame.
inline int& WorkingBpc()
{
	static thread_local int bpc = 0;
	return bpc;
}

// Scoped setter -- restores the previous value so nested/reentrant renders (AE
// does re-enter the effect for the bokeh preview) cannot leak a depth outward.
struct ScopedWorkingBpc {
	int prev;
	explicit ScopedWorkingBpc(int bpc) : prev(WorkingBpc()) { WorkingBpc() = bpc; }
	~ScopedWorkingBpc() { WorkingBpc() = prev; }
};

inline A_long MirrorCoordSafe(A_long c, A_long size)
{
	if (size <= 1) return 0;
	if (c < 0) c = -c - 1;
	const A_long period = size * 2;
	c = c % period;
	if (c < 0) c += period;
	return (c >= size) ? (period - 1 - c) : c;
}

// Pyramid edge handling. v3.1: CLAMP, not mirror.
//
// A bokeh disc near the frame border reaches outside the plate, and mirroring
// answers those taps with a flipped copy of the image -- so a bright point a few
// pixels inside the edge is gathered TWICE, once at its real position and once
// as its reflection, and prints a phantom second bokeh outside the frame edge.
// Clamping extends the border row instead: still not physical (nothing is), but
// it invents no new features, which is the artifact that actually reads as a
// mistake on screen. The CUDA texture uses cudaAddressModeClamp to match.
inline A_long ClampCoordSafe(A_long c, A_long size)
{
	if (size <= 1) return 0;
	return (c < 0) ? 0 : ((c >= size) ? (size - 1) : c);
}

// Gather colour space. v2.21: LINEAR light is the default. Defocus is physically
// a linear-light convolution; the old perceptual (gamma-2.0) averaging faked a
// "solid" blur but turned a dark night sky with sparse bright lights into milky
// fog and dulled the highlight punch (the user's "больше fog" report). Linear
// keeps blurred backgrounds dark/clean and bokeh bright -- validated against the
// reference night shot. The "translucent bright-sky-over-dark-geometry" look that
// originally motivated perceptual is a compositing/coverage matter handled
// elsewhere, not a reason to gamma-bias the whole gather. ZLUX_PROFILE builds can
// force the legacy perceptual path with ZLUX_PERCEPTUAL=1 for A/B comparison.
#ifdef ZLUX_PROFILE
inline bool ZluxLinear() { static const bool v = std::getenv("ZLUX_PERCEPTUAL") == nullptr; return v; }
// A/B switch for the v3.1 weight-LOD split (see the mip_w block in GatherPass).
// The split is the entire justification for letting Bokeh Definition remove the
// footprint floor, so being able to turn it off and re-measure the speckle is
// worth a profile-only knob. Never present in a release build.
inline bool ZluxWeightLodSplit() { static const bool v = std::getenv("ZLUX_NOWLOD") == nullptr; return v; }
#else
inline bool ZluxLinear() { return true; }
inline bool ZluxWeightLodSplit() { return true; }
#endif
inline Color3 PercToLin(const Color3& p) {
	return ZluxLinear() ? p : Color3{ p.r * p.r, p.g * p.g, p.b * p.b };
}

// Perceptual-space bilinear sample of a pyramid level at normalized UV.
inline Color3 SampleMipLinear(const MipLevel& L, PF_FpLong u, PF_FpLong v)
{
	const A_long iw = L.w;
	const A_long ih = L.h;
	if (iw <= 0 || ih <= 0 || L.data.empty()) return {0.0, 0.0, 0.0};
	if (iw == 1 && ih == 1) {
		const float* p = L.data.data();
		return { static_cast<PF_FpLong>(p[0]),
		         static_cast<PF_FpLong>(p[1]),
		         static_cast<PF_FpLong>(p[2]) };
	}
	const PF_FpLong px = u * static_cast<PF_FpLong>(iw) - 0.5;
	const PF_FpLong py = v * static_cast<PF_FpLong>(ih) - 0.5;
	const A_long fx0 = static_cast<A_long>(std::floor(px));
	const A_long fy0 = static_cast<A_long>(std::floor(py));
	const A_long x0 = ClampCoordSafe(fx0, iw);
	const A_long y0 = ClampCoordSafe(fy0, ih);
	const A_long x1 = ClampCoordSafe(fx0 + 1, iw);
	const A_long y1 = ClampCoordSafe(fy0 + 1, ih);
	// Coordinate math stays in double (precise floor/mirror at 4K), but the
	// bilinear blend itself runs in single precision: the mip data is float, so
	// this avoids 12 float->double widenings per tap and uses float madds. The
	// result is sub-ULP-identical to the double blend and is the dominant
	// arithmetic in the gather (one call per Vogel sample per pixel).
	const float fx = static_cast<float>(px - std::floor(px));
	const float fy = static_cast<float>(py - std::floor(py));
	const float* p00 = L.data.data() + (static_cast<size_t>(y0) * iw + x0) * 4;
	const float* p10 = L.data.data() + (static_cast<size_t>(y0) * iw + x1) * 4;
	const float* p01 = L.data.data() + (static_cast<size_t>(y1) * iw + x0) * 4;
	const float* p11 = L.data.data() + (static_cast<size_t>(y1) * iw + x1) * 4;
	const float w00 = (1.0f - fx) * (1.0f - fy);
	const float w10 = fx * (1.0f - fy);
	const float w01 = (1.0f - fx) * fy;
	const float w11 = fx * fy;
	return {
		static_cast<PF_FpLong>(p00[0]*w00 + p10[0]*w10 + p01[0]*w01 + p11[0]*w11),
		static_cast<PF_FpLong>(p00[1]*w00 + p10[1]*w10 + p01[1]*w01 + p11[1]*w11),
		static_cast<PF_FpLong>(p00[2]*w00 + p10[2]*w10 + p01[2]*w01 + p11[2]*w11)
	};
}

// Pick the mip level whose bilinear footprint ≥ the Vogel inter-sample
// spacing, so adjacent taps cover a contiguous area with no gaps.
//
// Inter-sample radial spacing in a uniform Vogel disc of radius R with N
// points is cell = R * sqrt(pi/N). At mip M a single bilinear fetch reads
// a 2x2 neighbourhood of mip-texels, i.e. a footprint of 2 * 2^M source
// pixels in each axis. Require footprint >= cell ⇒ M >= log2(cell/2);
// we use ceil, not round, because rounding DOWN leaves microscopic gaps
// that tile into visible streaks the moment the kernel becomes elliptical
// (anamorphic aperture 2..4) and the gap lattice aligns along one axis.
inline A_long PickMipLevel(PF_FpLong eff_radius_px, A_long n_samples, A_long num_levels)
{
	if (num_levels <= 1) return 0;
	const PF_FpLong n = static_cast<PF_FpLong>(std::max<A_long>(1, n_samples));
	const PF_FpLong cell = eff_radius_px * 1.7725 / std::sqrt(n);
	if (cell <= 2.0) return 0;
	const PF_FpLong mip_f = std::log2(cell * 0.5);
	const A_long mip = static_cast<A_long>(std::ceil(mip_f - 1e-6));
	return ClampValue<A_long>(mip, 0, num_levels - 1);
}

// Continuous (fractional) mip level -- same footprint model as PickMipLevel but
// WITHOUT the ceil, so the gather can trilinear-blend the two bracketing levels.
// Single-level bilinear sampling of the ceil'd (always-coarser) mip is what made
// large bokeh look blocky / box-filtered; trilinear filtering over the exact
// fractional LOD is the correct texture reconstruction and removes it.
inline PF_FpLong PickMipLevelF(PF_FpLong eff_radius_px, A_long n_samples)
{
	const PF_FpLong n = static_cast<PF_FpLong>(std::max<A_long>(1, n_samples));
	const PF_FpLong cell = eff_radius_px * 1.7725 / std::sqrt(n);
	if (cell <= 2.0) return 0.0;
	return std::max<PF_FpLong>(0.0, std::log2(cell * 0.5));
}

// Trilinear pyramid sample: bilinear within each of the two bracketing mip
// levels, then linear blend by the fractional LOD.
inline Color3 SampleMipTrilinear(const SourcePyramid& pyr, PF_FpLong mip_f,
                                 PF_FpLong u, PF_FpLong v)
{
	const A_long maxL = pyr.num_levels - 1;
	if (maxL <= 0) return SampleMipLinear(pyr.levels[0], u, v);
	const PF_FpLong mf = ClampValue<PF_FpLong>(mip_f, 0.0, static_cast<PF_FpLong>(maxL));
	const A_long m0 = static_cast<A_long>(mf);          // floor (mf >= 0)
	const PF_FpLong t = mf - static_cast<PF_FpLong>(m0);
	const Color3 c0 = SampleMipLinear(pyr.levels[static_cast<size_t>(m0)], u, v);
	if (t <= 1e-4 || m0 >= maxL) return c0;
	const Color3 c1 = SampleMipLinear(pyr.levels[static_cast<size_t>(m0 + 1)], u, v);
	return { c0.r + (c1.r - c0.r) * t,
	         c0.g + (c1.g - c0.g) * t,
	         c0.b + (c1.b - c0.b) * t };
}

// Single-channel linear-light bilinear sample (ch: 0=R,1=G,2=B). The chromatic-
// aberration path samples each channel at a DIFFERENT offset and keeps only that
// one channel, so calling the full Color3 SampleMipLinear there computed (and
// threw away) two thirds of the bilinear blend. This variant runs the identical
// coordinate math but blends only the requested channel -- bit-identical output,
// ~3x less blend arithmetic on the CA hot path.
inline PF_FpLong SampleMipLinearCh(const MipLevel& L, PF_FpLong u, PF_FpLong v, int ch)
{
	const A_long iw = L.w;
	const A_long ih = L.h;
	if (iw <= 0 || ih <= 0 || L.data.empty()) return 0.0;
	if (iw == 1 && ih == 1) return static_cast<PF_FpLong>(L.data[static_cast<size_t>(ch)]);
	const PF_FpLong px = u * static_cast<PF_FpLong>(iw) - 0.5;
	const PF_FpLong py = v * static_cast<PF_FpLong>(ih) - 0.5;
	const A_long fx0 = static_cast<A_long>(std::floor(px));
	const A_long fy0 = static_cast<A_long>(std::floor(py));
	const A_long x0 = ClampCoordSafe(fx0, iw);
	const A_long y0 = ClampCoordSafe(fy0, ih);
	const A_long x1 = ClampCoordSafe(fx0 + 1, iw);
	const A_long y1 = ClampCoordSafe(fy0 + 1, ih);
	const float fx = static_cast<float>(px - std::floor(px));
	const float fy = static_cast<float>(py - std::floor(py));
	const float* p00 = L.data.data() + (static_cast<size_t>(y0) * iw + x0) * 4 + ch;
	const float* p10 = L.data.data() + (static_cast<size_t>(y0) * iw + x1) * 4 + ch;
	const float* p01 = L.data.data() + (static_cast<size_t>(y1) * iw + x0) * 4 + ch;
	const float* p11 = L.data.data() + (static_cast<size_t>(y1) * iw + x1) * 4 + ch;
	const float w00 = (1.0f - fx) * (1.0f - fy);
	const float w10 = fx * (1.0f - fy);
	const float w01 = (1.0f - fx) * fy;
	const float w11 = fx * fy;
	return static_cast<PF_FpLong>(p00[0]*w00 + p10[0]*w10 + p01[0]*w01 + p11[0]*w11);
}

// Single-channel trilinear pyramid sample (see SampleMipLinearCh).
inline PF_FpLong SampleMipTrilinearCh(const SourcePyramid& pyr, PF_FpLong mip_f,
                                      PF_FpLong u, PF_FpLong v, int ch)
{
	const A_long maxL = pyr.num_levels - 1;
	if (maxL <= 0) return SampleMipLinearCh(pyr.levels[0], u, v, ch);
	const PF_FpLong mf = ClampValue<PF_FpLong>(mip_f, 0.0, static_cast<PF_FpLong>(maxL));
	const A_long m0 = static_cast<A_long>(mf);
	const PF_FpLong t = mf - static_cast<PF_FpLong>(m0);
	const PF_FpLong c0 = SampleMipLinearCh(pyr.levels[static_cast<size_t>(m0)], u, v, ch);
	if (t <= 1e-4 || m0 >= maxL) return c0;
	const PF_FpLong c1 = SampleMipLinearCh(pyr.levels[static_cast<size_t>(m0 + 1)], u, v, ch);
	return c0 + (c1 - c0) * t;
}

// Populate Level 0 of a pyramid directly from an arbitrary-bit-depth source.
// For 8/16 bit (perceptual gamma) we apply a cheap gamma 2.0 → linear decode
// once, so the rest of the pipeline reads linear-light floats and we avoid
// per-tap gamma math downstream. 32bpc float is already linear-light (AE's
// convention) so we pass it through untouched -- any squaring here would
// darken the pyramid and kill HDR highlights.
template <typename PIX>
void PopulatePyramidLevel0(const PF_EffectWorld* src, MipLevel& L0,
                           PF_FpLong highlight_recovery = 0.0,
                           bool* out_has_alpha = nullptr,
                           const float* signed_coc_cache = nullptr,
                           A_long coc_w = 0, A_long coc_h = 0,
                           bool srgb = false)
{
	const A_long w = src->width;
	const A_long h = src->height;
	L0.w = w;
	L0.h = h;
	// Seed level-0 CoC magnitudes for the edge-aware downsample. Only when the
	// CoC cache matches the level-0 dimensions (it is built at output size); on
	// any mismatch we leave coc empty and the pyramid falls back to plain box
	// filtering (old behaviour).
	if (signed_coc_cache && coc_w == w && coc_h == h) {
		L0.coc.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
		const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
		for (size_t i = 0; i < n; ++i) L0.coc[i] = std::abs(signed_coc_cache[i]);
	}
	// RGBA interleaved: RGB is gamma-decoded to linear light; A is stored as
	// LINEAR coverage (no gamma decode -- alpha is not a perceptual quantity).
	// Carrying alpha lets the gather blur the matte alongside the colour so a
	// layer with transparency feathers its silhouette into the bokeh instead
	// of staying razor-cut. `any_transp` records whether feathering is even
	// needed (any non-opaque pixel) so opaque footage skips the whole path.
	L0.data.assign(static_cast<size_t>(w) * static_cast<size_t>(h) * 4, 0.0f);
	std::atomic<bool> any_transp{false};
	const PF_FpLong im = PixTraits<PIX>::inv_max;
	const float imf = static_cast<float>(im);
	const float recovery_f = static_cast<float>(highlight_recovery);
	constexpr bool kIsFloat = std::is_same_v<PIX, PF_PixelFloat>;
	// 32bpc renders from 3D packages (Octane, Redshift, Cycles) routinely
	// contain NaN / Inf pixels around motion-blur / DOF AOV edges. Left
	// unchecked, a single NaN propagates through the MIP downsample and
	// infects the whole pyramid (NaN + anything = NaN). Sanitize once here.
	auto sanitize = [](PF_FpLong v) -> float {
		return std::isfinite(v) ? static_cast<float>(v) : 0.0f;
	};
	// ── #10 Highlight Clipping Recovery ──────────────────────────────
	// Integer 8/16bpc formats clamp at 1.0 in linear, so specular cores
	// (sun discs, window catches, LED lights) all land at the same
	// saturated value and bokeh loses the punchy brightness differential
	// DOF PRO gets from true HDR sources. Heuristic recovery: detect
	// near-unity luma (> 0.92) and extrapolate the intensity using a
	// smoothstep-weighted power curve. highlight_recovery scales the
	// peak multiplier (0 = off, 1 = +~3.5× for fully clipped whites).
	// Float (32bpc) formats already carry HDR so the path is skipped.
	const bool do_recovery = !kIsFloat && highlight_recovery > 0.01;
	// Level-0 population is a pure per-pixel map (gamma decode + sanitize +
	// optional highlight recovery), so it parallelizes cleanly by rows.
	ParallelRows(h, 64, [&](A_long y0, A_long y1) {
	bool row_transp = false;
	for (A_long y = y0; y < y1; ++y) {
		const PIX* row = PixelPtr<PIX>(const_cast<PF_EffectWorld*>(src), 0, y);
		float* dst = L0.data.data() + static_cast<size_t>(y) * static_cast<size_t>(w) * 4;
		if constexpr (std::is_same_v<PIX, PF_PixelFloat>) {
			// 32bpc is linear-light; sanitize then perceptual-encode (sqrt) so
			// the pyramid is stored in the same gamma-2.0 space as 8/16bpc.
			// Linear-gather mode stores linear directly (no sqrt).
			const bool lin = ZluxLinear();
			auto perc = [&](PF_FpLong v) -> float {
				const float s = std::max(0.0f, sanitize(v));
				return lin ? s : std::sqrt(s);
			};
			for (A_long x = 0; x < w; ++x) {
				dst[x * 4 + 0] = perc(static_cast<PF_FpLong>(row[x].red)   * im);
				dst[x * 4 + 1] = perc(static_cast<PF_FpLong>(row[x].green) * im);
				dst[x * 4 + 2] = perc(static_cast<PF_FpLong>(row[x].blue)  * im);
				const float a = sanitize(static_cast<PF_FpLong>(row[x].alpha) * im);
				dst[x * 4 + 3] = a;
				if (a < 0.998f) row_transp = true;
			}
		} else {
			// 8/16bpc: gamma-2.0 decode in single precision so the inner loop
			// vectorizes (AVX2). The rare highlight-recovery path is a separate
			// loop so the common decode stays branch-free and auto-vectorizes.
			const bool lin = ZluxLinear();
			if (do_recovery) {
				for (A_long x = 0; x < w; ++x) {
					// Perceptual values (raw gamma); recovery gain is derived in
					// LINEAR luma then folded back into perceptual via sqrt(gain).
					const float rp = static_cast<float>(row[x].red)   * imf;
					const float gp = static_cast<float>(row[x].green) * imf;
					const float bp = static_cast<float>(row[x].blue)  * imf;
					if (lin) {
						// Linear storage: decode to linear (sRGB or gamma-2.0), then
						// apply the recovery gain in linear light.
						const float lr = srgb ? static_cast<float>(SrgbToLinearCh(rp)) : rp*rp;
						const float lg = srgb ? static_cast<float>(SrgbToLinearCh(gp)) : gp*gp;
						const float lb = srgb ? static_cast<float>(SrgbToLinearCh(bp)) : bp*bp;
						const float lum = lr * 0.2126f + lg * 0.7152f + lb * 0.0722f;
						const float t = static_cast<float>(SmoothStep(0.86, 0.99, lum));
						const float g = 1.0f + recovery_f * 2.5f * t;
						dst[x * 4 + 0] = lr * g;
						dst[x * 4 + 1] = lg * g;
						dst[x * 4 + 2] = lb * g;
					} else {
						const float lum = rp*rp * 0.2126f + gp*gp * 0.7152f + bp*bp * 0.0722f;
						const float t = static_cast<float>(SmoothStep(0.86, 0.99, lum));
						const float gp_ = std::sqrt(1.0f + recovery_f * 2.5f * t);
						dst[x * 4 + 0] = rp * gp_;
						dst[x * 4 + 1] = gp * gp_;
						dst[x * 4 + 2] = bp * gp_;
					}
					const float a = static_cast<float>(row[x].alpha) * imf;
					dst[x * 4 + 3] = a;
					if (a < 0.998f) row_transp = true;
				}
			} else if (srgb) {
				// True sRGB decode (opt-in). pow() per channel -- doesn't vectorize
				// like the gamma-2.0 path, but it is the correct linearization for
				// sRGB/Rec.709 footage so bokeh colour averages right outside 32bpc.
				for (A_long x = 0; x < w; ++x) {
					dst[x * 4 + 0] = static_cast<float>(SrgbToLinearCh(static_cast<float>(row[x].red)   * imf));
					dst[x * 4 + 1] = static_cast<float>(SrgbToLinearCh(static_cast<float>(row[x].green) * imf));
					dst[x * 4 + 2] = static_cast<float>(SrgbToLinearCh(static_cast<float>(row[x].blue)  * imf));
					const float a = static_cast<float>(row[x].alpha) * imf;
					dst[x * 4 + 3] = a;
					if (a < 0.998f) row_transp = true;
				}
			} else {
				for (A_long x = 0; x < w; ++x) {
					// Perceptual (gamma-2.0): raw 8/16bpc is already gamma -- store
					// direct. Linear mode squares it to linear light.
					const float rp = static_cast<float>(row[x].red)   * imf;
					const float gp = static_cast<float>(row[x].green) * imf;
					const float bp = static_cast<float>(row[x].blue)  * imf;
					dst[x * 4 + 0] = lin ? rp*rp : rp;
					dst[x * 4 + 1] = lin ? gp*gp : gp;
					dst[x * 4 + 2] = lin ? bp*bp : bp;
					const float a = static_cast<float>(row[x].alpha) * imf;
					dst[x * 4 + 3] = a;
					if (a < 0.998f) row_transp = true;
				}
			}
		}
	}
	if (row_transp) any_transp.store(true, std::memory_order_relaxed);
	});
	if (out_has_alpha) *out_has_alpha = any_transp.load(std::memory_order_relaxed);
}

// 4x4 tent downsample (v3.1; was a 2x2 box).
//
// The gather reads essentially all of its defocused colour out of this pyramid,
// so the pyramid IS the inside of every bokeh disc. A 2x2 box is a terrible
// low-pass: its stopband leaks badly, so each level keeps a share of the
// aliasing of the level above it. On a still frame that reads as blocky
// box-filter texels inside large discs; on a moving one the aliased energy
// beats against the sampling grid and the defocused texture crawls.
//
// A 4x4 tent -- the outer product of (1,3,3,1)/8, i.e. two box passes -- is
// centred on exactly the same point as the 2x2 box (offsets -1,0,+1,+2 around
// 2x are symmetric about the box centre), so nothing shifts; it just rolls off
// an octave earlier. 16 taps per destination texel instead of 4, but only over
// a quarter-resolution target and only once per frame.
//
// Deliberately NOT a Karis / luma-weighted average: that is the standard
// firefly fix, and it works by throwing away exactly the bright-highlight
// energy that makes bokeh worth having. Fireflies are handled where they
// actually originate -- the weight nonlinearity in the gather (see mip_w).
inline void DownsamplePyramidLevel(const MipLevel& prev, MipLevel& cur)
{
	cur.w = std::max<A_long>(1, prev.w / 2);
	cur.h = std::max<A_long>(1, prev.h / 2);
	cur.data.assign(static_cast<size_t>(cur.w) * static_cast<size_t>(cur.h) * 4, 0.0f);
	// Edge-aware path is enabled only when the parent level carries CoC info.
	const bool edge_aware = !prev.coc.empty();
	if (edge_aware)
		cur.coc.assign(static_cast<size_t>(cur.w) * static_cast<size_t>(cur.h), 0.0f);
	// Bias toward the more-defocused child. Small floor so equal-CoC regions
	// (no edge) collapse to the plain box average -- the bias only bites at a
	// real CoC discontinuity (a silhouette).
	constexpr float kEdgeBiasFloor = 0.03f;
	// 2x2 box filter: each destination row is independent, so parallelize by
	// rows. Levels are built sequentially (each depends on the previous) but
	// the within-level work fans out across cores.
	// (1,3,3,1) tent taps at offsets -1,0,+1,+2 from the 2x source position.
	static constexpr float kTent[4] = {1.0f / 8.0f, 3.0f / 8.0f, 3.0f / 8.0f, 1.0f / 8.0f};
	ParallelRows(cur.h, 64, [&](A_long y0, A_long y1) {
	// Row base pointers for the four vertical taps, resolved once per row.
	const float* rowp[4];
	const float* rowc[4];
	for (A_long y = y0; y < y1; ++y) {
		for (int ky = 0; ky < 4; ++ky) {
			const A_long yy = ClampCoordSafe(y * 2 + ky - 1, prev.h);
			rowp[ky] = prev.data.data() + static_cast<size_t>(yy) * prev.w * 4;
			rowc[ky] = edge_aware ? (prev.coc.data() + static_cast<size_t>(yy) * prev.w)
			                      : nullptr;
		}
		for (A_long x = 0; x < cur.w; ++x) {
			float* dst = cur.data.data() + (static_cast<size_t>(y) * cur.w + x) * 4;
			float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
			float wsum = 0.0f;
			float cmax = 0.0f;
			for (int ky = 0; ky < 4; ++ky) {
				for (int kx = 0; kx < 4; ++kx) {
					const A_long xx = ClampCoordSafe(x * 2 + kx - 1, prev.w);
					const float* p = rowp[ky] + static_cast<size_t>(xx) * 4;
					float w = kTent[ky] * kTent[kx];
					if (edge_aware) {
						// Bias toward the more-defocused child so a silhouette does
						// not drag sharp foreground colour up the pyramid. The tent
						// weight and the edge bias simply multiply.
						const float cv = rowc[ky][static_cast<size_t>(xx)];
						w *= (cv + kEdgeBiasFloor);
						// Propagate the MAX blur magnitude over the 2x2 CORE only.
						// Widening it to the whole tent would grow the "defocused"
						// flag by an extra texel per level and over-bias the chain.
						if ((ky == 1 || ky == 2) && (kx == 1 || kx == 2) && cv > cmax)
							cmax = cv;
					}
					acc0 += p[0] * w; acc1 += p[1] * w;
					acc2 += p[2] * w; acc3 += p[3] * w;
					wsum += w;
				}
			}
			const float inv = (wsum > 1e-20f) ? (1.0f / wsum) : 0.0f;
			dst[0] = acc0 * inv; dst[1] = acc1 * inv;
			dst[2] = acc2 * inv; dst[3] = acc3 * inv;
			if (edge_aware) cur.coc[static_cast<size_t>(y) * cur.w + x] = cmax;
		}
	}
	});
}

template <typename PIX>
void BuildSourcePyramidTyped(const PF_EffectWorld* src, SourcePyramid& pyr, PF_FpLong highlight_recovery = 0.0, A_long max_levels = 6,
                            const float* signed_coc_cache = nullptr, A_long coc_w = 0, A_long coc_h = 0,
                            bool srgb = false)
{
	pyr.levels.clear();
	pyr.levels.reserve(static_cast<size_t>(max_levels));
	pyr.levels.emplace_back();
	bool has_alpha = false;
	PopulatePyramidLevel0<PIX>(src, pyr.levels.back(), highlight_recovery, &has_alpha,
	                           signed_coc_cache, coc_w, coc_h, srgb);
	pyr.has_alpha = has_alpha;
	while (pyr.levels.size() < static_cast<size_t>(max_levels)) {
		const MipLevel& prev = pyr.levels.back();
		if (prev.w <= 32 || prev.h <= 32) break;
		pyr.levels.emplace_back();
		DownsamplePyramidLevel(pyr.levels[pyr.levels.size() - 2], pyr.levels.back());
	}
	pyr.num_levels = static_cast<A_long>(pyr.levels.size());
}

inline void BuildSourcePyramid(const PF_EffectWorld* src, SourcePyramid& pyr, PF_FpLong highlight_recovery = 0.0,
                               const float* signed_coc_cache = nullptr, A_long coc_w = 0, A_long coc_h = 0,
                               bool srgb = false)
{
	if (!src || !src->data) {
		pyr.levels.clear();
		pyr.num_levels = 0;
		return;
	}
	if (PF_WORLD_IS_DEEP(const_cast<PF_EffectWorld*>(src))) {
		BuildSourcePyramidTyped<PF_Pixel16>(src, pyr, highlight_recovery, 6, signed_coc_cache, coc_w, coc_h, srgb);
	} else if (WorldIsFloat(src)) {
		BuildSourcePyramidTyped<PF_PixelFloat>(src, pyr, highlight_recovery, 6, signed_coc_cache, coc_w, coc_h, srgb);
	} else {
		BuildSourcePyramidTyped<PF_Pixel8>(src, pyr, highlight_recovery, 6, signed_coc_cache, coc_w, coc_h, srgb);
	}
}

inline bool WorldIsFloat(const PF_EffectWorld* world)
{
	// PF_WorldFlag_DEEP marks 16 bpc, so a world carrying it is never float.
	if ((world->world_flags & PF_WorldFlag_DEEP) != 0) return false;

	// Exact answer whenever AE told us the working depth.
	if (const int bpc = WorkingBpc()) return bpc == 32;

	// Fallback for the legacy non-smart path: infer from the row stride. This is
	// ambiguous for very narrow worlds -- AE pads rows to a 64-byte boundary, so
	// a 4-px-wide 8 bpc row occupies 16 bytes padded to 64, which is exactly
	// width * sizeof(PF_PixelFloat) and reads as float out of a quarter-sized
	// buffer. Reachable on thumbnail-sized renders, which is why the smart path
	// above no longer relies on it.
	return world->rowbytes >= static_cast<A_long>(world->width * sizeof(PF_PixelFloat));
}

inline Color3 SampleColorUVWorld(const PF_EffectWorld* world, PF_FpLong u, PF_FpLong v)
{
	if (!world || !world->data) {
		return {0.0, 0.0, 0.0};
	}
	if (PF_WORLD_IS_DEEP(const_cast<PF_EffectWorld*>(world))) {
		return SampleColorUV<PF_Pixel16>(world, u, v);
	}
	if (WorldIsFloat(world)) {
		return SampleColorUV<PF_PixelFloat>(world, u, v);
	}
	return SampleColorUV<PF_Pixel8>(world, u, v);
}

// Maps a raw depth-channel value into [0,1]. When per-frame auto-ranging is
// active (non-normalized inputs such as a linear-Z EXR pass), the value is
// rescaled by the robust content range; otherwise the legacy clamp to [0,1] is
// used so conventional normalized maps behave exactly as before. Non-finite
// samples (EXR AOV garbage / infinity-far sentinels) collapse to the far plane
// under auto-range and to 0 otherwise -- matching how each path treats them.
inline PF_FpLong NormalizeRawDepth(PF_FpLong raw, const DOFSettings& s)
{
	if (s.depth_autorange) {
		if (!std::isfinite(raw)) return 1.0; // sentinel "infinitely far"
		return Clamp01((raw - s.depth_range_min) * s.depth_range_inv_span);
	}
	if (!std::isfinite(raw)) return 0.0;
	return Clamp01(raw);
}

inline PF_FpLong SampleDepthChannelRaw(const PF_EffectWorld* depth_world, PF_FpLong u, PF_FpLong v, const DOFSettings& s)
{
	const Color3 c = SampleColorUVWorld(depth_world, u, v);
	PF_FpLong raw;
	switch (s.depth_channel) {
		case 2:  raw = c.r; break;
		case 3:  raw = c.g; break;
		case 4:  raw = c.b; break;
		// Un-clamped Rec.601 luma so auto-range sees the true value range.
		// Depth passes are virtually always grayscale (R=G=B), so the exact
		// weighting is immaterial; this matches the legacy Luma() channel.
		default: raw = c.r * 0.299 + c.g * 0.587 + c.b * 0.114; break;
	}
	return NormalizeRawDepth(raw, s);
}

template <typename PIX>
inline PF_FpLong ReadDepthNN(const PF_EffectWorld* dw, A_long x, A_long y, int ch)
{
	const PIX* p = PixelPtr<PIX>(dw, x, y);
	const PF_FpLong im = PixTraits<PIX>::inv_max;
	switch (ch) {
		case 2: return static_cast<PF_FpLong>(p->red) * im;
		case 3: return static_cast<PF_FpLong>(p->green) * im;
		case 4: return static_cast<PF_FpLong>(p->blue) * im;
		default: {
			PF_FpLong r = static_cast<PF_FpLong>(p->red) * im;
			PF_FpLong g = static_cast<PF_FpLong>(p->green) * im;
			PF_FpLong b = static_cast<PF_FpLong>(p->blue) * im;
			return r * 0.2126 + g * 0.7152 + b * 0.0722;
		}
	}
}

// Reads one depth channel (raw, un-normalized) from an already-fetched pixel.
template <typename PIX>
inline PF_FpLong ReadDepthChannelRawPix(const PIX* p, int ch)
{
	const PF_FpLong im = PixTraits<PIX>::inv_max;
	switch (ch) {
		case 2:  return static_cast<PF_FpLong>(p->red)   * im;
		case 3:  return static_cast<PF_FpLong>(p->green) * im;
		case 4:  return static_cast<PF_FpLong>(p->blue)  * im;
		default: return (static_cast<PF_FpLong>(p->red)   * 0.299 +
		                 static_cast<PF_FpLong>(p->green) * 0.587 +
		                 static_cast<PF_FpLong>(p->blue)  * 0.114) * im;
	}
}

// Magnitude above which a depth sample is assumed to be an EXR "infinity-far"
// sentinel (empty/sky pixels in a Z AOV) rather than real geometry, and is
// excluded from auto-range estimation. Real scene Z is essentially never this
// large; such pixels still normalize to the far plane via NormalizeRawDepth.
constexpr PF_FpLong kDepthSentinelMag = 1.0e6;

// Scans the depth layer's selected channel for a robust content [min,max] used
// to normalize the depth onto the full [0,1] working range. Returns true (and
// fills out_min/out_max) whenever the data does not already fill ~[0,1] -- both
// out-of-range linear-Z EXR passes AND under-utilized maps that occupy only a
// sub-band (e.g. an AI disparity pass in [0, 0.34]). A map that already spans
// the full [0,1] returns false so conventional normalized projects are left on
// the exact legacy path.
//
// Robustness: two passes over the (subsampled) channel. Pass 1 finds the finite
// extent, ignoring values beyond kDepthSentinelMag (literal 1e30 "infinity"
// sentinels). Pass 2 bins the values into a histogram and trims to the
// [0.2 .. 99.8] percentile, so sparse outliers -- including the fp16 65504
// "far" sentinel that 3D apps stamp on empty pixels -- do not collapse the
// usable range onto the real geometry. Pixels outside the trimmed range simply
// clamp to the near/far plane in NormalizeRawDepth. Large maps are subsampled
// to ~1M taps; stepping cannot change whether the data is normalized and barely
// moves a percentile.
template <typename PIX>
bool ComputeDepthAutoRange(const PF_EffectWorld* dw, int ch, PF_FpLong& out_min, PF_FpLong& out_max)
{
	const A_long w = dw->width;
	const A_long h = dw->height;
	if (w <= 0 || h <= 0) return false;

	const A_long total = w * h;
	const A_long step = std::max<A_long>(1, total / 1000000);

	// IMPORTANT: this is the FIRST read of the checked-out depth world each
	// frame, and it MUST stay single-threaded. AE realizes a checked-out
	// layer's pixels lazily on first touch; doing that first touch from 32
	// worker threads at once (a v2.14 "optimization") raced AE's realization
	// and crashed with an access violation on RTX/EXR projects -- the depth
	// build below was safe before only because this serial scan touched the
	// world first. The whole scan is subsampled to ~1M taps, so it is a few
	// ms; parallelizing it saved ~7ms of a 1000ms+ frame and is not worth the
	// crash. Keep it serial.

	// Pass 1: finite extent (excluding hard infinity sentinels).
	// Both passes are order-invariant reductions (min, max, counts), so splitting
	// them across rows is bit-identical to the serial version -- and this
	// function was the last single-threaded stage of the depth chain, costing
	// ~8 ms of the frame purely for want of a parallel_for.
	// `idx` is the GLOBAL sample index, so each row recomputes its own start
	// rather than relying on a running counter.
	std::mutex reduce_mx;
	PF_FpLong mn =  1.0e30;
	PF_FpLong mx = -1.0e30;
	A_long n_valid = 0;
	ParallelRows(h, 16, [&](A_long y0, A_long y1) {
		PF_FpLong l_mn = 1.0e30, l_mx = -1.0e30;
		A_long l_n = 0;
		for (A_long y = y0; y < y1; ++y) {
			const PIX* row = PixelPtr<PIX>(const_cast<PF_EffectWorld*>(dw), 0, y);
			A_long idx = y * w;
			for (A_long x = 0; x < w; ++x, ++idx) {
				if (step > 1 && (idx % step) != 0) continue;
				const PF_FpLong v = ReadDepthChannelRawPix<PIX>(&row[x], ch);
				if (!std::isfinite(v) || std::abs(v) > kDepthSentinelMag) continue;
				if (v < l_mn) l_mn = v;
				if (v > l_mx) l_mx = v;
				++l_n;
			}
		}
		std::lock_guard<std::mutex> lk(reduce_mx);
		if (l_mn < mn) mn = l_mn;
		if (l_mx > mx) mx = l_mx;
		n_valid += l_n;
	});
	if (n_valid <= 0 || (mx - mn) < kEps) return false;

	// Pass 2: histogram over [mn, mx]; take the 0.2 / 99.8 percentile edges.
	constexpr int kBins = 512;
	A_long hist[kBins] = { 0 };
	const PF_FpLong inv_span = static_cast<PF_FpLong>(kBins) / (mx - mn);
	ParallelRows(h, 16, [&](A_long y0, A_long y1) {
		// Re-stated locally: a [&] capture makes the enclosing constexpr a
		// captured reference under MSVC, which is then not a constant expression.
		constexpr int kBinsLocal = 512;
		A_long l_hist[kBinsLocal] = { 0 };
		for (A_long y = y0; y < y1; ++y) {
			const PIX* row = PixelPtr<PIX>(const_cast<PF_EffectWorld*>(dw), 0, y);
			A_long idx = y * w;
			for (A_long x = 0; x < w; ++x, ++idx) {
				if (step > 1 && (idx % step) != 0) continue;
				const PF_FpLong v = ReadDepthChannelRawPix<PIX>(&row[x], ch);
				if (!std::isfinite(v) || std::abs(v) > kDepthSentinelMag) continue;
				int b = static_cast<int>((v - mn) * inv_span);
				if (b < 0) b = 0; else if (b >= kBins) b = kBins - 1;
				++l_hist[b];
			}
		}
		std::lock_guard<std::mutex> lk(reduce_mx);
		for (int b = 0; b < kBinsLocal; ++b) hist[b] += l_hist[b];
	});
	const A_long lo_rank = static_cast<A_long>(static_cast<double>(n_valid) * 0.002);
	const A_long hi_rank = static_cast<A_long>(static_cast<double>(n_valid) * 0.998);
	A_long acc = 0; int lo_bin = 0, hi_bin = kBins - 1;
	for (int b = 0; b < kBins; ++b) { acc += hist[b]; if (acc > lo_rank) { lo_bin = b; break; } }
	acc = 0;
	for (int b = 0; b < kBins; ++b) { acc += hist[b]; if (acc >= hi_rank) { hi_bin = b; break; } }
	const PF_FpLong bin_w = (mx - mn) / static_cast<PF_FpLong>(kBins);
	PF_FpLong rmin = mn + static_cast<PF_FpLong>(lo_bin) * bin_w;
	PF_FpLong rmax = mn + static_cast<PF_FpLong>(hi_bin + 1) * bin_w;
	if ((rmax - rmin) < kEps) { rmin = mn; rmax = mx; }

	// Normalize whenever the content does not already fill ~[0,1]. This covers
	// BOTH out-of-[0,1] data (linear-Z EXR) AND under-utilized maps that sit in
	// a sub-band -- e.g. an AI disparity / Depth Anything pass occupying only
	// [0, 0.34], which would otherwise leave Focus/Black/White operating in the
	// bottom third of their travel and make the DoF nearly untunable. A map that
	// already spans the full [0,1] (span >= 0.98 AND within bounds) keeps the
	// exact legacy path, so conventional normalized projects do not shift.
	const PF_FpLong span = rmax - rmin;
	if (rmin >= -1.0e-3 && rmax <= 1.0 + 1.0e-3 && span >= 0.98) return false;
	out_min = rmin;
	out_max = rmax;
	return true;
}

inline PF_FpLong FastBilateralWeight(PF_FpLong diff)
{
	const PF_FpLong x = std::abs(diff) * 120.0;
	return 1.0 / (1.0 + x * x);
}

// ── Circle-of-confusion (inverse-Z aperture model) ──────────────────────────
//
// The CoC is derived from a normalized inverse-Z curve rather than a flat
// linear |depth−focus| ramp, because 1/z is what gives a real lens its two
// signature behaviours: the near field grows far faster than the far field
// (CoC ∝ 1/z), and distant planes pool into one creamy value instead of racing
// to maximum blur. Blur Amount + Focus Distance drive it; the depth map is read
// as Normalized linear (white = near).

// Maps a normalized depth d∈[0,1] (0 = near plane, 1 = far plane) to inverse
// metric distance 1/z under the standard linear-in-distance encoding:
//   z = zn + d·(zf − zn)  ->  return 1/z.
// (The Depth Encoding popup -- linear-Z / disparity variants -- was retired in
// v2.22; the depth map is always read as Normalized linear, white = near.)
inline PF_FpLong DepthInvZ(PF_FpLong d, PF_FpLong zn, PF_FpLong zf)
{
	d = Clamp01(d);
	const PF_FpLong z = zn + d * (zf - zn);
	return 1.0 / std::max<PF_FpLong>(kEps, z);
}

// Pseudo near/far distances for the inverse-Z CoC curve. These reproduce the
// legacy 1/(0.25 + 0.75·d) curvature exactly (depth 0 -> 1/z = 4, depth 1 ->
// 1/z = 1): a fixed lens-like depth ratio so the background actually saturates
// into blur while Blur Amount keeps full authority at any focus position.
constexpr PF_FpLong kSimpleZNear = 0.25;
constexpr PF_FpLong kSimpleZFar  = 1.0;

// Signed CoC: negative = near (foreground), positive = far (background).
// focus_range acts as a *soft* DoF width; small CoCs inside it are smoothly
// attenuated instead of being hard-clipped to zero (no step at the focus plane).
inline PF_FpLong ComputeSignedCoC(PF_FpLong depth, const DOFSettings& s)
{
	// Per-side normalized inverse-Z. Each side of the focal plane is scaled by
	// its own reachable maximum, so Blur Amount keeps full authority at ANY
	// focus position (the background actually blurs when you pull focus to the
	// foreground, instead of staying near-sharp).
	const PF_FpLong inv_z    = DepthInvZ(depth,            kSimpleZNear, kSimpleZFar);
	const PF_FpLong inv_zf   = DepthInvZ(s.focal_distance, kSimpleZNear, kSimpleZFar);
	const PF_FpLong raw_dinv = inv_zf - inv_z; // >0 => far, <0 => near
	const PF_FpLong far_span  = std::max<PF_FpLong>(kEps,
		inv_zf - DepthInvZ(1.0, kSimpleZNear, kSimpleZFar));
	const PF_FpLong near_span = std::max<PF_FpLong>(kEps,
		DepthInvZ(0.0, kSimpleZNear, kSimpleZFar) - inv_zf);
	PF_FpLong raw = ((raw_dinv >= 0.0) ? (raw_dinv / far_span)     // [0, +1] background
	                                   : (raw_dinv / near_span))   // [-1, 0] foreground
	                * s.blur_strength;

	// Smooth DoF-zone attenuation (no hard dead band at the focal plane).
	const PF_FpLong abs_raw = std::abs(raw);
	const PF_FpLong dof_edge = std::max<PF_FpLong>(kEps, s.focus_range * 2.0);
	raw *= SmoothStep(0.0, dof_edge, abs_raw);

	// Artistic bias for the near field. Default 1.0 = neutral; 0 fully
	// disables the near pass downstream (~half the frame cost).
	if (raw < 0.0) {
		if (s.near_blur_factor <= 0.001) {
			return 0.0;
		}
		raw *= s.near_blur_factor;
	}
	return ClampValue<PF_FpLong>(raw, -kCocClamp, kCocClamp);
}

// ── Optical Vignetting (Cat's Eye) ─────────────────────────────────────────
//
// Real optical vignetting is the intersection of two circular apertures:
//   • the main iris (unit-radius disc, centered on the optical axis)
//   • the lens barrel / hood opening (offset toward the image center
//     by an amount proportional to field distance)
//
// This function returns a smooth 0..1 mask for a sample position `p` inside
// the normalized aperture (|p|<=1 is inside the iris). It must NEVER cause
// the aperture to close completely — otherwise all bokeh samples at frame
// edges get rejected and the frame appears "cropped", which is not what a
// real lens does. The two discs always retain a visible overlap.
//
// `strength` is signed: positive = classic cat's eye (real lenses),
// negative = inverted vignette used by catadioptric (mirror) lenses.
// `scale` models the size ratio of the lens barrel vs the iris:
//   scale = 1 : barrel matches iris (symmetric cat's eye).
//   scale > 1 : barrel is larger (hood ≫ aperture) — vignetting appears
//               later and with a flatter occlusion contour.
//   scale < 1 : tighter hood, vignetting kicks in closer to center.
inline PF_FpLong GetCatsEyeMask(
	const Vec2& p,
	PF_FpLong field_u, PF_FpLong field_v,
	PF_FpLong strength, PF_FpLong scale)
{
	const PF_FpLong abs_s = std::abs(strength);
	if (abs_s < 0.001) return 1.0;

	const PF_FpLong fcx = field_u - 0.5;
	const PF_FpLong fcy = field_v - 0.5;
	const PF_FpLong fd  = std::sqrt(fcx * fcx + fcy * fcy);
	if (fd < 0.02) return 1.0;

	// Field factor saturates near the frame corner (|(fcx,fcy)|_max ≈ 0.707).
	const PF_FpLong field = Clamp01(fd / 0.707);

	// Lens-barrel radius grows with `scale`. The exponent keeps it soft so a
	// scale-1 aperture is the reference cat's eye.
	const PF_FpLong barrel_r = std::max<PF_FpLong>(0.6, 1.0 + (scale - 1.0) * 0.5);

	// Offset of the barrel center relative to the iris. Hard-capped so the
	// two discs always overlap — otherwise the mask can collapse to zero and
	// frame edges would darken unnaturally.
	const PF_FpLong max_shift = (1.0 + barrel_r) * 0.88;
	PF_FpLong shift_mag = field * abs_s * 1.2;
	if (shift_mag > max_shift) shift_mag = max_shift;

	// Positive: barrel shifts toward image center (natural cat's eye).
	// Negative: barrel shifts outward (catadioptric / mirror inversion).
	const PF_FpLong sign = (strength >= 0.0) ? -1.0 : 1.0;
	const PF_FpLong bx = sign * (fcx / fd) * shift_mag;
	const PF_FpLong by = sign * (fcy / fd) * shift_mag;

	const PF_FpLong dpx = p.x - bx;
	const PF_FpLong dpy = p.y - by;
	const PF_FpLong dist = std::sqrt(dpx * dpx + dpy * dpy);
	return 1.0 - SmoothStep(barrel_r * 0.93, barrel_r * 1.02, dist);
}

// Hoisted cat's-eye: everything in GetCatsEyeMask except the final per-tap
// disc test depends only on the pixel's field position, not the sample, so the
// gather computes it ONCE per pixel (a field sqrt + barrel offset) and then
// evaluates only (sub, smoothstep) per tap. EvalCatsEye(MakeCatsEye(...), p)
// is bit-identical to GetCatsEyeMask(p, ...).
struct CatsEyeSetup {
	bool active;          // false -> mask is always 1 (off / on-axis)
	PF_FpLong bx, by;     // barrel-disc centre offset in iris coords
	PF_FpLong inner, outer; // SmoothStep edges (barrel_r * 0.93 / 1.02)
};

inline CatsEyeSetup MakeCatsEye(PF_FpLong field_u, PF_FpLong field_v,
                                PF_FpLong strength, PF_FpLong scale)
{
	CatsEyeSetup c{};
	c.active = false;
	const PF_FpLong abs_s = std::abs(strength);
	if (abs_s < 0.001) return c;
	const PF_FpLong fcx = field_u - 0.5;
	const PF_FpLong fcy = field_v - 0.5;
	const PF_FpLong fd  = std::sqrt(fcx * fcx + fcy * fcy);
	if (fd < 0.02) return c;
	const PF_FpLong field = Clamp01(fd / 0.707);
	const PF_FpLong barrel_r = std::max<PF_FpLong>(0.6, 1.0 + (scale - 1.0) * 0.5);
	const PF_FpLong max_shift = (1.0 + barrel_r) * 0.88;
	PF_FpLong shift_mag = field * abs_s * 1.2;
	if (shift_mag > max_shift) shift_mag = max_shift;
	const PF_FpLong sign = (strength >= 0.0) ? -1.0 : 1.0;
	c.bx = sign * (fcx / fd) * shift_mag;
	c.by = sign * (fcy / fd) * shift_mag;
	c.inner = barrel_r * 0.93;
	c.outer = barrel_r * 1.02;
	c.active = true;
	return c;
}

inline PF_FpLong EvalCatsEye(const CatsEyeSetup& c, PF_FpLong px, PF_FpLong py)
{
	if (!c.active) return 1.0;
	const PF_FpLong dpx = px - c.bx;
	const PF_FpLong dpy = py - c.by;
	const PF_FpLong dist = std::sqrt(dpx * dpx + dpy * dpy);
	return 1.0 - SmoothStep(c.inner, c.outer, dist);
}

inline Vec2 ApplyAstigmatism(const Vec2& offset_uv, PF_FpLong u, PF_FpLong v, PF_FpLong strength, PF_Boolean sagittal_mode)
{
	if (strength < 0.001) {
		return offset_uv;
	}
	const PF_FpLong cx = u - 0.5;
	const PF_FpLong cy = v - 0.5;
	const PF_FpLong dist = std::sqrt(cx * cx + cy * cy);
	if (dist < 0.025) {
		return offset_uv;
	}
	const PF_FpLong field_angle = std::atan2(cy, cx);
	const PF_FpLong ef = SmoothStep(0.05, 0.55, dist * 2.0);
	// Cap raised 1.0 -> 2.5 (v2.10) together with the Astigmatism slider's
	// 0..200 range: reference "zoom-burst" rigs elongate edge bokeh well
	// past the old 1.6x ceiling.
	const PF_FpLong edge_strength = std::min(strength * ef, 2.5);
	const Vec2 tangential{-std::sin(field_angle), std::cos(field_angle)};
	const Vec2 sagittal{std::cos(field_angle), std::sin(field_angle)};
	const PF_FpLong tang_comp = offset_uv.x * tangential.x + offset_uv.y * tangential.y;
	const PF_FpLong sag_comp = offset_uv.x * sagittal.x + offset_uv.y * sagittal.y;
	const PF_FpLong stretch = 1.0 + edge_strength * 0.6;
	const PF_FpLong squeeze = 1.0 / stretch;
	const PF_FpLong tang_scale = sagittal_mode ? squeeze : stretch;
	const PF_FpLong sag_scale = sagittal_mode ? stretch : squeeze;
	return {
		tangential.x * tang_comp * tang_scale + sagittal.x * sag_comp * sag_scale,
		tangential.y * tang_comp * tang_scale + sagittal.y * sag_comp * sag_scale
	};
}

// ── Polygonal aperture with blade notching ─────────────────────────────────
//
// Regular N-blade iris diaphragm, modelled as a *hard-clipped* polygon whose
// tips sit on the unit circle. Returns ~1 inside the shape, ~0 outside,
// with a narrow anti-aliasing band (~1 pixel at typical iris-preview /
// bokeh sizes).
//
// `curvature` is bipolar in [−1, +1] and models the full range of blade
// shapes seen in DOF PRO's reference:
//   • curvature = −1  → concave blades (star / flower silhouette, blade
//                       midpoints pulled all the way to the centre).
//   • curvature =  0  → straight-edged polygon.
//   • curvature = +1  → fully circular (blades bowed all the way out).
// Implemented by computing the shape's edge *radius* at the sample's
// angular sector and comparing it to the sample's actual radius. The edge
// radius is interpolated between three canonical curves (concave / polygon
// / circle) depending on the sign of curvature. This keeps the AA band
// uniformly narrow at every curvature value -- the old formulation coupled
// "curvature" with a wide SmoothStep(0.48, 1.0) soft-clip whose transition
// zone alone occupied >50% of the iris radius, producing the semi-
// transparent halo the user was seeing instead of a crisp iris silhouette.
//
// Blade notching carves a V-shaped triangular indent into every blade tip.
// `notch_scale` controls depth + width (0 = none, 1 = cut almost to the
// centre -- the DOF PRO "Notch Scale 100" flower / star look).
// `notch_bias` (radians) rotates the V to one side of the tip for the
// asymmetric "Notch Angle" look. The V is a hard mechanical cut (mask →
// 0) with its own narrow AA band.
inline PF_FpLong GetPolygonalAperture(const Vec2& normalized_offset,
                                      A_long blades,
                                      PF_FpLong curvature,
                                      PF_FpLong notch_scale = 0.0,
                                      PF_FpLong notch_bias = 0.0)
{
	if (blades < 3) {
		return 1.0;
	}
	const PF_FpLong angle = std::atan2(normalized_offset.y, normalized_offset.x) + kPi;
	const PF_FpLong radius = Length(normalized_offset);
	const PF_FpLong blade_angle = kTau / static_cast<PF_FpLong>(blades);
	const PF_FpLong half_ba = blade_angle * 0.5;
	const PF_FpLong cos_hba = std::cos(half_ba);

	// Sector ∈ [−half_ba, +half_ba]; 0 at a blade-edge midpoint, ±half_ba
	// at the two adjacent tips.
	const PF_FpLong sector = std::fmod(angle, blade_angle) - half_ba;
	const PF_FpLong abs_sector = std::abs(sector);

	// Three canonical edge-radius curves sharing tips at (sector = ±half_ba,
	// r = 1) so blending between them never moves the tip position:
	//   • Polygon: edge_r_poly = cos(half_ba) / cos(sector)
	//       → cos(half_ba) at midpoint, 1 at tips (straight-edged N-gon
	//         inscribed by its tips in the unit circle).
	//   • Circle:  edge_r_circ = 1 everywhere.
	//   • Concave: edge_r_conc = sin((|sector| / half_ba) · π/2)
	//       → 0 at midpoint, 1 at tips (blades bow fully *inward*, giving
	//         the star / flower silhouette seen on DOF PRO at curvature
	//         −100).
	const PF_FpLong edge_r_poly = cos_hba / std::max<PF_FpLong>(1e-6, std::cos(sector));
	const PF_FpLong edge_r_circ = 1.0;
	const PF_FpLong edge_r_conc = std::sin(abs_sector / std::max<PF_FpLong>(1e-6, half_ba) * (kPi * 0.5));

	PF_FpLong edge_r;
	if (curvature >= 0.0) {
		const PF_FpLong c = Clamp01(curvature);
		edge_r = edge_r_poly * (1.0 - c) + edge_r_circ * c;
	} else {
		const PF_FpLong c = Clamp01(-curvature);
		edge_r = edge_r_poly * (1.0 - c) + edge_r_conc * c;
	}

	// edge_dist normalised so "1 = exactly on edge". Guard against edge_r
	// collapsing to 0 in extreme concave settings (would produce Inf).
	const PF_FpLong edge_dist = radius / std::max<PF_FpLong>(1e-5, edge_r);

	// Narrow symmetric AA band. 0.015 on unit-iris scale ≈ 1 px at a
	// 60 px iris preview / bokeh render -- crisp but not aliased.
	constexpr PF_FpLong kAA = 0.015;
	PF_FpLong mask = 1.0 - SmoothStep(1.0 - kAA, 1.0 + kAA, edge_dist);
	if (mask <= 0.0) {
		return 0.0;
	}

	// ── V-shaped blade notching ──────────────────────────────────────────
	// A real optical iris with notched blades carves a SHALLOW triangular
	// bite out of the OUTER RIM of each blade tip. The V is a bounded
	// triangle (apex inside the blade, two arms meeting the polygon edge)
	// — NOT an unbounded wedge running to the iris centre. A previous
	// version used a centre-reaching wedge, which at high notch_scale
	// turned circular / high-curvature irises into "sunburst" shapes
	// (spokes radiating from the centre), which is not what DOF PRO
	// produces. The formulation below calibrates apex depth + arm
	// half-angle so ns=1 on the DOF PRO pentagon+circle+ns=100 reference
	// matches the "flower" silhouette, and every intermediate value
	// scales proportionally.
	if (notch_scale > 0.001) {
		const PF_FpLong ns = Clamp01(notch_scale);
		// Clamp bias so the apex never crosses the blade midpoint, beyond
		// which neighbouring V-cuts overlap and the geometry becomes
		// ambiguous. ±0.7 × half-sector is ample asymmetry for the
		// swirl / pinwheel Notch Angle looks in the DOF PRO references.
		const PF_FpLong bias_rad = ClampValue<PF_FpLong>(notch_bias,
		                                                 -half_ba * 0.7,
		                                                  half_ba * 0.7);
		const PF_FpLong sign_side = (sector >= 0.0) ? 1.0 : -1.0;
		const PF_FpLong tip_sec = sign_side * half_ba;
		const PF_FpLong apex_sec = tip_sec + bias_rad * sign_side;

		// Apex radial position: ns=0 → 1.0 (no cut), ns=1 → 0.45 (cut
		// 55% of the blade radius at max). Critically the apex never
		// approaches the iris centre, so the preserved region forms a
		// solid disc topped with petal-shaped arcs -- DOF PRO's flower
		// look, not a sunburst.
		constexpr PF_FpLong kTipR = 1.0;
		const PF_FpLong apex_r = kTipR * (1.0 - ns * 0.55);
		// Arm half-angle: very narrow at low ns (Zuiko-style vertex
		// cusps — tiny V at the corner of a clean hexagon) opening up
		// to ~half the blade sector at ns=1 (wide bites, DOF PRO
		// photos 3/6). Base 0.08 × half_ba keeps the default Notched
		// look subtle like the Olympus Zuiko OM reference; cap at 0.50
		// × half_ba so adjacent V-cuts never overlap at the midpoint.
		const PF_FpLong arm_half = half_ba * std::min<PF_FpLong>(0.50,
		                                                         0.08 + ns * 0.45);

		const PF_FpLong ax = apex_r * std::cos(apex_sec);
		const PF_FpLong ay = apex_r * std::sin(apex_sec);
		const PF_FpLong px = radius * std::cos(sector);
		const PF_FpLong py = radius * std::sin(sector);
		const PF_FpLong dx = px - ax;
		const PF_FpLong dy = py - ay;

		const PF_FpLong theta1 = apex_sec + arm_half;
		const PF_FpLong theta2 = apex_sec - arm_half;

		// Cross-product signed distances to each arm (positive = inside
		// the wedge from that arm's point of view). "Both positive = in
		// wedge".
		const PF_FpLong d1 =  dx * std::sin(theta1) - dy * std::cos(theta1);
		const PF_FpLong d2 = -dx * std::sin(theta2) + dy * std::cos(theta2);

		// Additional constraint: only cut when the sample is on the
		// TIP SIDE of apex (past the line perpendicular to apex_dir
		// through apex). Without this the wedge would leak toward the
		// centre whenever apex_r is small, producing the "spokes"
		// artefact at high ns. Projecting disp onto apex_dir and
		// requiring a positive component bounds the cut to the outer
		// rim.
		const PF_FpLong t_side = dx * std::cos(apex_sec) +
		                         dy * std::sin(apex_sec);

		const PF_FpLong notch_sd = std::min({d1, d2, t_side});
		const PF_FpLong cut = SmoothStep(-0.010, 0.010, notch_sd);
		mask *= 1.0 - cut;
	}

	return mask;
}

inline PF_FpLong GetCatadioptricMask(const Vec2& normalized_offset, PF_FpLong strength)
{
	if (strength < 0.1) {
		return 1.0;
	}
	const PF_FpLong dist = Length(normalized_offset);
	const PF_FpLong inner_radius = 0.08 + strength * 0.72;
	const PF_FpLong outer = 1.0 - SmoothStep(0.7, 1.0, dist);
	const PF_FpLong inner = SmoothStep(inner_radius * 0.85, inner_radius, dist);
	return outer * inner;
}

inline Color3 MixColor(const Color3& a, const Color3& b, PF_FpLong t)
{
	return {Mix(a.r, b.r, t), Mix(a.g, b.g, t), Mix(a.b, b.b, t)};
}

// ── Transfer functions ──────────────────────────────────────────────────────
// 8/16bpc footage is display-referred (project gamma). The whole bokeh gather
// runs in linear light, so we decode at the pipeline edges. Default is the fast
// gamma-2.0 approximation (square / sqrt -- branch-free, AVX2-friendly). When the
// user enables sRGB Accurate we use the true IEC 61966-2-1 piecewise curve, which
// is the correct decode for the sRGB / Rec.709-ish footage AE actually delivers,
// so defocused bokeh colour (and especially saturated highlights) average right.
inline PF_FpLong SrgbToLinearCh(PF_FpLong c)
{
	if (c <= 0.04045) return c * (1.0 / 12.92);
	return std::pow((c + 0.055) * (1.0 / 1.055), 2.4);
}
inline PF_FpLong LinearToSrgbCh(PF_FpLong c)
{
	c = std::max(0.0, c);
	if (c <= 0.0031308) return c * 12.92;
	return 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

inline Color3 GammaToLinear(const Color3& c, bool srgb = false)
{
	if (srgb) return { SrgbToLinearCh(c.r), SrgbToLinearCh(c.g), SrgbToLinearCh(c.b) };
	return {c.r * c.r, c.g * c.g, c.b * c.b};
}

inline Color3 LinearToGamma(const Color3& c, bool srgb = false)
{
	if (srgb) return { LinearToSrgbCh(c.r), LinearToSrgbCh(c.g), LinearToSrgbCh(c.b) };
	return {
		std::sqrt(std::max(0.0, c.r)),
		std::sqrt(std::max(0.0, c.g)),
		std::sqrt(std::max(0.0, c.b))
	};
}

inline PF_FpLong RemapDepth(PF_FpLong d, const DOFSettings& s)
{
	PF_FpLong t = (d - s.depth_blackpoint) / std::max<PF_FpLong>(0.001, s.depth_whitepoint - s.depth_blackpoint);
	t = Clamp01(t);
	// Depth Gamma: non-linear remap of the focus falloff (1 = linear). >1 pushes
	// the bulk of the depth toward the near plane, <1 toward the far plane, so
	// the user can bias how quickly blur ramps with distance (DOF PRO depth remap).
	if (std::abs(s.depth_gamma - 1.0) > 0.001 && s.depth_gamma > 0.0) {
		t = std::pow(t, s.depth_gamma);
	}
	// Depth convention (v2.19+): the STANDARD is white = NEAR, black = FAR, which
	// is how the depth passes we target are authored. Internally d = 0 is the near
	// plane and d = 1 the far plane (see DepthInvZ), so a standard white-near map
	// must have its normalized value flipped: white(t=1) -> near(0), black(t=0) ->
	// far(1). "Invert Depth" therefore now means the LEGACY / non-standard maps
	// where black = near and white = far (some linear-Z or engine AOVs).
	return s.depth_invert ? t : (1.0 - t);
}

template <typename PIX>
inline PF_FpLong SampleDepthMapped(const PF_EffectWorld* depth_world, PF_FpLong u, PF_FpLong v, const DOFSettings& s)
{
	return RemapDepth(SampleDepthChannelRaw(depth_world, u, v, s), s);
}

// Pre-baked (cos, sin) pairs for 128 evenly-spaced rotations on [0, 2π).
// Used by the custom-aperture-texture path to give every bokeh its own
// random orientation without paying for a trig call per Vogel tap.
// 128 buckets = 2.8° granularity, well below the eye's ability to tell
// one rotation from another at typical bokeh sizes.
static const std::array<std::pair<PF_FpLong, PF_FpLong>, 128> kBokehRotLUT = []() {
	std::array<std::pair<PF_FpLong, PF_FpLong>, 128> a{};
	for (int i = 0; i < 128; ++i) {
		const PF_FpLong phi = static_cast<PF_FpLong>(i) * (kTau / 128.0);
		a[static_cast<size_t>(i)] = { std::cos(phi), std::sin(phi) };
	}
	return a;
}();

// Evaluates the custom-aperture texture mask for a single bokeh sample.
//
//   norm    : sample position in aperture disc coordinates ([-1, 1]^2).
//   field_u,
//   field_v : on-screen position of the bokeh centre ([0, 1]^2).
//   s       : DOF settings (carries intensity / scale / invert / offset).
//   sample_u,
//   sample_v : UV of the source position this Vogel tap lands on. When both
//              are ≥ 0, they drive a deterministic per-bokeh rotation of
//              the texture (hashed on a coarse source-pixel grid so all
//              taps that contribute to the same bright source point get
//              the same rotation). Pass a negative sentinel to disable
//              rotation -- used by the iris preview where randomness would
//              scramble the illustration.
//
// The texture is sampled with these knobs:
//   * Scale   -- zooms into the texture (>1) or repeats it (<1) around the
//                aperture centre.
//   * Offset  -- shifts the texture UV by the bokeh's frame position, so
//                adjacent bokehs see different regions of the texture
//                (emulates real lens-dirt parallax; this is the DOF PRO
//                "offset" feature).
//   * Invert  -- inverts luminance before mixing.
//   * Intensity -- blends the textured mask against a plain (1.0) disc, so
//                  setting it to 0 disables the effect entirely.
//   * Per-bokeh rotation (implicit) -- makes every individual bokeh in the
//                  frame show the texture at a different orientation, which
//                  is what a real lens does (random dust orientation,
//                  coating scratches, oil streaks). Active whenever sample
//                  coordinates are supplied.
// The UV wraps around so neighbouring bokehs cross-fade seamlessly.
inline PF_FpLong SampleApertureTextureMask(
	const PF_EffectWorld* tex_world,
	const Vec2& norm,
	PF_FpLong field_u,
	PF_FpLong field_v,
	const DOFSettings& s,
	PF_FpLong sample_u = -1.0,
	PF_FpLong sample_v = -1.0)
{
	if (!tex_world || !tex_world->data) {
		return 1.0;
	}
	const PF_FpLong intensity = Clamp01(s.aperture_texture_intensity);
	if (intensity < 0.001) return 1.0;

	// Per-bokeh rotation randomization. We quantize the sample's source UV
	// to a coarse grid (~256 cells across the frame, i.e. ~8 source pixels
	// per cell at 4K) and hash that cell to a rotation bucket. Because
	// every output pixel that gathers the same bright source point lands
	// at (approximately) the same (sample_u, sample_v), they all hit the
	// same bucket and agree on the rotation -- the bokeh stays coherent.
	// Different source points get different rotations, so every bokeh in
	// the frame looks like an independent instance of the aperture.
	Vec2 rnorm = norm;
	if (sample_u >= 0.0 && sample_v >= 0.0) {
		const A_long gx = static_cast<A_long>(std::floor(sample_u * 256.0));
		const A_long gy = static_cast<A_long>(std::floor(sample_v * 256.0));
		uint32_t h = static_cast<uint32_t>(gx) * 374761393u
		           ^ static_cast<uint32_t>(gy) * 668265263u;
		h = (h ^ (h >> 13)) * 1274126177u;
		h ^= h >> 16;
		const size_t bucket = static_cast<size_t>(h & 127u);
		const auto& rot = kBokehRotLUT[bucket];
		rnorm.x = norm.x * rot.first  - norm.y * rot.second;
		rnorm.y = norm.x * rot.second + norm.y * rot.first;
	}

	const PF_FpLong inv_scale = 1.0 / std::max<PF_FpLong>(0.25, s.aperture_texture_scale);
	PF_FpLong u = rnorm.x * 0.5 * inv_scale + 0.5;
	PF_FpLong v = rnorm.y * 0.5 * inv_scale + 0.5;

	const PF_FpLong off = s.aperture_texture_offset;
	if (std::abs(off) > 0.001) {
		u += (field_u - 0.5) * off;
		v += (field_v - 0.5) * off;
	}

	// Seamless wrap -- negative fmod would break without the +floor.
	u -= std::floor(u);
	v -= std::floor(v);

	PF_FpLong mask = Luma(SampleColorUVWorld(tex_world, u, v));
	if (s.aperture_texture_invert) mask = 1.0 - mask;
	return Mix(1.0, Clamp01(mask), intensity);
}

inline PF_FpLong ComputeHighlightMask(const Color3& c, const DOFSettings& s)
{
	const PF_FpLong lum = Luma(c) * 255.0;
	const PF_FpLong lo = s.highlights_low;
	const PF_FpLong hi = std::max(lo + 1.0, s.highlights_high);
	const PF_FpLong soft = std::max<PF_FpLong>(0.5, s.highlights_softness * 255.0);
	const PF_FpLong a = SmoothStep(lo - soft, lo + soft, lum);
	const PF_FpLong b = 1.0 - SmoothStep(hi - soft, hi + soft, lum);
	return Clamp01(a * b);
}

inline PF_FpLong ComputeSphericalProfile(PF_FpLong radius01, PF_FpLong center_depth, const DOFSettings& s)
{
	const PF_FpLong plane_sign = (center_depth >= s.focal_distance) ? 1.0 : -1.0;
	const PF_FpLong aberration = s.spherical_aberration_amount * plane_sign;
	if (std::abs(aberration) < 0.001) {
		return 1.0;
	}
	const PF_FpLong t = Clamp01(std::abs(aberration));
	const PF_FpLong r = ClampValue<PF_FpLong>(radius01, 0.0, 1.0);
	const PF_FpLong sharpness = 2.0 + s.spherical_aberration_scale * 6.0;
	if (aberration > 0.0) {
		const PF_FpLong profile = std::pow(r, sharpness);
		const PF_FpLong norm = (sharpness + 1.0) * 0.5;
		return Mix(1.0, profile * norm, t);
	}
	const PF_FpLong profile = std::pow(1.0 - r, sharpness);
	const PF_FpLong norm = (sharpness + 1.0) * 0.5;
	return Mix(1.0, profile * norm, t);
}

template <typename PIX>
PF_FpLong SampleWeightedFocusDepth(
	const PF_EffectWorld* depth_world,
	const DOFSettings& s,
	PF_FpLong focus_u,
	PF_FpLong focus_v,
	PF_FpLong inv_w,
	PF_FpLong inv_h)
{
	const PF_FpLong center_depth = SampleDepthMapped<PIX>(depth_world, focus_u, focus_v, s);
	PF_FpLong sum = 0.0;
	PF_FpLong total = 0.0;
	for (A_long oy = -3; oy <= 3; ++oy) {
		for (A_long ox = -3; ox <= 3; ++ox) {
			const PF_FpLong du = static_cast<PF_FpLong>(ox) * inv_w;
			const PF_FpLong dv = static_cast<PF_FpLong>(oy) * inv_h;
			const PF_FpLong u = focus_u + du;
			const PF_FpLong v = focus_v + dv;
			const PF_FpLong d = SampleDepthMapped<PIX>(depth_world, u, v, s);
			const PF_FpLong sd2 = (du * du + dv * dv) * 16000.0;
			const PF_FpLong spatial = 1.0 / (1.0 + sd2);
			const PF_FpLong dd = std::abs(d - center_depth) * 85.0;
			const PF_FpLong depth_similarity = 1.0 / (1.0 + dd * dd);
			const PF_FpLong w = spatial * depth_similarity;
			sum += d * w;
			total += w;
		}
	}
	return (total > kEps) ? (sum / total) : center_depth;
}

// Per-sample matte-box occluder.
//
// Models the straight flags / lens-barrel edge that physically sit in the
// light cone and clip the aperture uniformly for every bokeh in the frame.
// Each fin is a straight half-plane; the slider directly encodes how far
// that fin has been pushed into the aperture disc:
//
//   slider = 0.0  -> fin sits just outside the disc (no effect).
//   slider = 0.5  -> fin has advanced halfway across the disc (half-moon).
//   slider = 1.0  -> fin covers the whole disc from that side.
//
// Aperture coords: norm ∈ [-1, 1]^2 with +y pointing DOWN in AE screen
// space (same as the offset.y used in GatherPass), so the TOP fin lives on
// the negative-y side of the disc.
//
// This is intentionally uniform across the frame -- progressive,
// position-dependent disc reshaping is already handled by Optical
// Vignetting (cat's eye) and is a different optical phenomenon.
inline PF_FpLong GetMatteBoxApertureMask(const Vec2& norm, const DOFSettings& s)
{
	if (s.matte_top < 0.001 && s.matte_bottom < 0.001 &&
	    s.matte_left < 0.001 && s.matte_right < 0.001) {
		return 1.0;
	}

	PF_FpLong mask = 1.0;
	constexpr PF_FpLong kSoft = 0.04; // soft chord width for smooth edges

	if (s.matte_top > 0.001) {
		const PF_FpLong fin = -1.0 + 2.0 * s.matte_top;
		mask *= SmoothStep(fin - kSoft, fin + kSoft, norm.y);
	}
	if (s.matte_bottom > 0.001) {
		const PF_FpLong fin = 1.0 - 2.0 * s.matte_bottom;
		mask *= 1.0 - SmoothStep(fin - kSoft, fin + kSoft, norm.y);
	}
	if (s.matte_left > 0.001) {
		const PF_FpLong fin = -1.0 + 2.0 * s.matte_left;
		mask *= SmoothStep(fin - kSoft, fin + kSoft, norm.x);
	}
	if (s.matte_right > 0.001) {
		const PF_FpLong fin = 1.0 - 2.0 * s.matte_right;
		mask *= 1.0 - SmoothStep(fin - kSoft, fin + kSoft, norm.x);
	}
	return Clamp01(mask);
}

inline PF_FpLong HashNoise(PF_FpLong x, PF_FpLong y, PF_FpLong seed)
{
	PF_FpLong v = std::sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453;
	return v - std::floor(v);
}

// Pick a sample count that scales with blur magnitude. The hot loop cost
// is strictly linear in N, so shrinking N for small kernels is the single
// cheapest speed-up we have. We never expand past the user-requested
// base_count any more -- for large bokeh the mip pyramid already provides
// every source pixel pre-filtered, so doubling the taps only trades CPU
// time for no visible gain. A final footprint-aware cap in GatherPass
// further trims redundant taps when the picked mip is coarse enough that
// each sample's bilinear kernel already covers multiple neighbours.
inline A_long PickVogelLUT(PF_FpLong blur_norm, A_long base_count)
{
	if (blur_norm < 0.06) return std::max<A_long>(16, static_cast<A_long>(base_count * blur_norm * 4.0));
	if (blur_norm < 0.18) return std::max<A_long>(48, static_cast<A_long>(base_count * (0.2 + blur_norm * 1.6)));
	if (blur_norm < 0.40) return std::max<A_long>(96, static_cast<A_long>(base_count * (0.45 + blur_norm * 0.9)));
	return base_count;
}

// ── Two-pass DoF kernel ─────────────────────────────────────────────────────
//
// The classic gather kernel is split into two independent passes that share
// all of the optical machinery (aperture shape, CA, vignette, astigmatism,
// highlights, spherical/catadioptric profiles). They differ only in which
// samples they accept and how the result is normalized:
//
//   Far pass   -- gathers within the center pixel's own far CoC radius and
//                 only admits samples that are themselves far. Returns a
//                 blurred-background color and a coverage factor used as the
//                 lerp amount against the sharp source.
//
//   Near pass  -- gathers within the tile-dilated maximum near CoC. A sample
//                 contributes only if its own CoC disc *reaches* this pixel.
//                 The accumulated weight vs. total aperture area yields the
//                 near-field alpha that governs occlusion over the far/sharp
//                 composite. This emulates true scatter/spreading without an
//                 intermediate buffer.
//
// Returned rgb is already in linear light; the compositor keeps the mix in
// linear and re-encodes at the end.
enum class DofPass { Far, Near };

struct PassOutput {
	Color3 rgb;
	PF_FpLong weight; // Far: coverage in [0..1]; Near: alpha in [0..1]
	// Blurred PREMULTIPLIED alpha (the matte feathered by the bokeh kernel),
	// accumulated with the same per-tap weights as rgb so the premult pair
	// stays consistent. Only meaningful when the source pyramid has alpha
	// (pyramid.has_alpha); otherwise left at a neutral sentinel and ignored.
	PF_FpLong matte;
};

// ── Bake per-sample frame-constant data ────────────────────────────────────
//
// The aperture masks (polygonal / notched, catadioptric, softness edge,
// matte-box flags) are expensive to evaluate: atan2, fmod, pow, several
// sqrt + smoothstep combinations each. They all depend only on the sample
// position on the unit disc and on settings that do not vary across the
// frame -- so evaluating them inside the per-pixel gather loop (tens of
// billions of times per 4K frame) is pure waste.
//
// This pass evaluates everything that is frame-constant once, stores the
// results directly in the VogelSample, and lets the hot loop read a single
// float per tap instead of re-deriving it. Only per-pixel masks survive
// the inner loop: CatsEye (needs field position), SphericalProfile (needs
// centre depth), custom aperture texture (needs UV for offset).
// `apmap` is the caller's snapshot of the selected aperture-map library shape
// (null when none is selected). Passed in rather than read from a global so the
// map cannot be swapped out underneath the bake by another render thread.
inline void FinalizeVogelLUT(VogelLUT& lut, const DOFSettings& s,
                             const zlux_apmap::ApMap* apmap)
{
	const PF_FpLong inv_anam = 1.0 / std::max<PF_FpLong>(0.1, s.anamorphic_ratio);
	const PF_FpLong cos_rot  = std::cos(s.bokeh_rotation_rad);
	const PF_FpLong sin_rot  = std::sin(s.bokeh_rotation_rad);
	// `blade_curve` is bipolar: −1 = concave blades (star/flower), 0 =
	// straight polygon, +1 = fully circular. `GetPolygonalAperture`
	// interprets the sign; we pass it through unmodified.
	const PF_FpLong blade_shape = ClampValue<PF_FpLong>(s.blade_curve, -1.0, 1.0);

	const bool is_poly  = (s.aperture_shape_mode == 2 || s.aperture_shape_mode == 3);
	const bool has_cata = s.catadioptric > 0.1;
	const bool has_soft = s.softness > 0.001;
	const bool has_matte = (s.matte_top > 0.001 || s.matte_bottom > 0.001 ||
	                        s.matte_left > 0.001 || s.matte_right > 0.001);
	// Built-in aperture-map library shape (baked per sample if one is loaded).
	// Sampled through the mip level matched to this LUT's tap density so the
	// map's fine texture is integrated, not aliased, into the bake (full-res
	// point sampling produced fingerprint moiré on the bokeh discs, amplified
	// by Bokeh Gamma's weight contrast).
	const bool has_apmap = (s.aperture_map_index > 0) && zlux_apmap::Active(apmap);
	const int apmap_level = has_apmap ? zlux_apmap::PickLevelForSamples(apmap, lut.count) : 0;
	const bool has_onion = s.onion_amount > 0.001;

	const PF_FpLong soft_edge_start = has_soft ? (1.0 - s.softness * 0.6) : 1.0;
	const PF_FpLong blade_angle_rad = kTau / static_cast<PF_FpLong>(std::max<A_long>(3, s.aperture_blades));
	// In "Notched" shape mode, force a subtle minimum V-cut (0.15) so
	// picking that mode without touching the slider immediately shows
	// a clean Olympus Zuiko OM 50mm-style hexagonal iris with tiny
	// vertex cusps (the canonical "notched blade" look). Users still
	// drive Notch Depth freely; at 0 the preset minimum kicks in.
	const PF_FpLong ns_val = (s.aperture_shape_mode == 3)
		? std::max<PF_FpLong>(0.15, s.notch_scale)
		: s.notch_scale;
	const PF_FpLong notch_bias = (s.notch_angle / kPi) * (blade_angle_rad * 0.5);

	// Spherical-aberration radial profile bake. sharpness is frame-constant, so
	// pow(fr, sharpness) / pow(1-fr, sharpness) can be evaluated once here and
	// reused for every pixel instead of per tap (see ComputeSphericalProfile /
	// the gather's SA branch). Must use the exact same sharpness expression.
	const bool has_spher_bake = std::abs(s.spherical_aberration_amount) > 0.001;
	const PF_FpLong sa_sharpness = 2.0 + s.spherical_aberration_scale * 6.0;

	for (A_long i = 0; i < lut.count; ++i) {
		VogelSample& vs = lut.samples[i];
		// Pre-rotation, anamorphic-squashed unit-disc position -- this is
		// what gets multiplied by the per-pixel effective_radius to form
		// the sample offset in pixels.
		vs.kx = vs.cos_a * vs.fr * inv_anam;
		vs.ky = vs.sin_a * vs.fr;
		// Rotated normalized position used by every aperture-shape mask.
		vs.norm_x = vs.kx * cos_rot - vs.ky * sin_rot;
		vs.norm_y = vs.kx * sin_rot + vs.ky * cos_rot;

		// Soft-edge roll-off at the disc boundary. Stored separately so the
		// custom-texture path (which cannot bake the final aperture mask)
		// can still combine it cheaply.
		vs.soft_edge = has_soft ? (1.0 - SmoothStep(soft_edge_start, 1.0, vs.fr)) : 1.0;

		PF_FpLong sm = 1.0;
		if (is_poly) {
			sm *= GetPolygonalAperture({vs.norm_x, vs.norm_y},
			                           s.aperture_blades, blade_shape,
			                           ns_val, notch_bias);
		}
		if (has_cata) {
			sm *= GetCatadioptricMask({vs.norm_x, vs.norm_y}, s.catadioptric);
		}
		sm *= vs.soft_edge;
		if (has_matte) {
			sm *= GetMatteBoxApertureMask({vs.norm_x, vs.norm_y}, s);
		}
		if (has_apmap) {
			sm *= zlux_apmap::SampleLevel(apmap, apmap_level, vs.norm_x, vs.norm_y);
		}
		if (has_onion) {
			sm *= OnionRingMask(vs.fr, s.onion_amount, s.onion_count);
		}
		vs.static_mask = sm;

		if (has_spher_bake) {
			vs.sa_pos = std::pow(vs.fr, sa_sharpness);
			vs.sa_neg = std::pow(1.0 - vs.fr, sa_sharpness);
		} else {
			vs.sa_pos = 1.0;
			vs.sa_neg = 1.0;
		}
	}
}

template <DofPass MODE>
PassOutput GatherPass(
	const SourcePyramid& pyramid,
	const PF_EffectWorld* aperture_tex_world,
	const PF_EffectWorld* iris_mod_world,
	PF_FpLong u,
	PF_FpLong v,
	PF_FpLong radius,
	const DOFSettings& s,
	PF_FpLong center_depth,
	PF_FpLong center_signed_coc,
	PF_FpLong inv_w,
	PF_FpLong inv_h,
	const VogelLUT* luts,
	A_long num_luts,
	const float* signed_coc_cache,
	A_long cache_w,
	A_long cache_h,
	// Optional precomputed distance-to-nearest-CoC-discontinuity field (full res,
	// cache_w x cache_h). When supplied, the silhouette-band test below reads it
	// once instead of doing the 16-tap scatter scan per pixel. Null => fall back
	// to the inline scan (e.g. if a caller has no field handy).
	const float* coc_disc_dist = nullptr)
{
	// Pyramid is always built before the gather pass runs (see RenderCore);
	// fall back to a safe neutral reply if the pyramid is unexpectedly empty.
	if (pyramid.num_levels <= 0) {
		if constexpr (MODE == DofPass::Far) {
			return { {0.0, 0.0, 0.0}, 0.0, 1.0 };
		} else {
			return { {0.0, 0.0, 0.0}, 0.0, 0.0 };
		}
	}

	const MipLevel& L0 = pyramid.levels[0];
	// NB: the pyramid is PERCEPTUAL space, so every colour sampled here
	// (center + ring) is gamma-2.0 encoded; the accumulation runs in that space
	// and PercToLin() converts back to linear at each return.
	const Color3 center_color_lin = SampleMipLinear(L0, u, v);
	// Premultiplied-alpha (matte) blur: only when the source actually has
	// transparency. center_alpha seeds the accumulator; opaque sources skip
	// the whole alpha path so the gather stays exactly as fast as before.
	const bool do_alpha = pyramid.has_alpha;
	const PF_FpLong center_alpha_lin = do_alpha ? SampleMipLinearCh(L0, u, v, 3) : 1.0;
	PF_FpLong a_acc = 0.0; // Σ sample_alpha * w  (matches rgb's acc / w_sum)

	if (radius <= 0.001) {
		if constexpr (MODE == DofPass::Far) {
			return { PercToLin(center_color_lin), 0.0, center_alpha_lin };
		} else {
			return { {0.0, 0.0, 0.0}, 0.0, 0.0 };
		}
	}

	const PF_FpLong min_dimension = 1.0 / std::max(inv_w, inv_h);
	const PF_FpLong effective_radius = radius * min_dimension * 0.15;
	const PF_FpLong blur_norm = Clamp01(radius / kCocClamp);

	// Energy-conserving (physical) gather: drop the brightness-weighting knobs
	// (Bokeh Brightness Boost + Bokeh Gamma) so colour is normalized by pure
	// geometric coverage -- a linear-light average of the source, exactly like a
	// real lens / 3D-render defocus. This also removes the high-variance weighting
	// that fireflies at strong boost. (Highlight Recovery + Scatter still apply.)
	const bool energy = s.energy_conserving;
	const bool has_ca = s.ca_strength > 0.001;
	const bool has_vig = std::abs(s.vignetting) > 0.001;
	const bool has_highlight = s.highlight_boost > 0.001 && !energy;
	const bool has_astig = s.astigmatism > 0.001;
	const bool has_cata = s.catadioptric > 0.1;
	const bool has_spher = std::abs(s.spherical_aberration_amount) > 0.001;
	// Per-pixel spherical-aberration setup. The radial profile is baked per
	// Vogel sample (sa_pos / sa_neg); here we resolve, once per pixel, which of
	// the two baked curves applies (by the sign of aberration × plane_sign) and
	// the frame-constant strength/normalization, so the tap loop only does a
	// select + lerp instead of a pow(). Reproduces ComputeSphericalProfile
	// exactly for vs.fr taps.
	const PF_FpLong sa_norm = (2.0 + s.spherical_aberration_scale * 6.0 + 1.0) * 0.5;
	const PF_FpLong sa_t = Clamp01(std::abs(s.spherical_aberration_amount));
	const bool sa_use_pos =
		(s.spherical_aberration_amount * ((center_depth >= s.focal_distance) ? 1.0 : -1.0)) > 0.0;
	const bool is_custom_tex = (s.aperture_shape_mode == 4);
	const bool uniform_blur = s.no_depth;
	// Bokeh Gamma amplifies bright samples non-linearly during the gather,
	// so specular highlights keep their peak energy through defocus (the
	// crisp-core DOF PRO signature) instead of being averaged away. The
	// boost applies a multiplicative weight of (1 + luma)^gamma -- gentle
	// on darks (luma=0 -> boost=1), strong on highlights (luma=1, gamma=3
	// -> boost=8). A flat 0 disables the code path so the common case
	// still runs the fast linear gather.
	const bool has_bokeh_gamma = s.bokeh_gamma > 0.001 && !energy;

	const PF_FpLong cache_w_f = static_cast<PF_FpLong>(cache_w);
	const PF_FpLong cache_h_f = static_cast<PF_FpLong>(cache_h);

	// ── Frischluft-style full-res taps in the silhouette band ───────────────
	// The mip pyramid is a colour-only box downsample, so a mip texel straddling
	// a depth discontinuity is a blend of foreground + background colour; the
	// per-tap depth gate then rejects that texel only AFTER its colour is already
	// contaminated, leaving the bright/dark fringe that makes zluxDOF silhouettes
	// lose to Frischluft DoF. FL reads full-res per tap and rejects by depth
	// BEFORE averaging. We emulate that exactly where it matters: when the local
	// signed-CoC field shows a real discontinuity (sign flip / large jump) around
	// this pixel, force the colour taps to mip 0 so each tap is a clean full-res
	// read and the existing depth gate culls the wrong-plane colour cleanly.
	// Bounded to small/medium CoC so the limited Vogel budget still covers the
	// disc -- huge bokeh stay on the mip path (the gather has no per-tap budget to
	// fill a giant disc at full res, and the contamination is far less visible at
	// that scale). The sample cap and mip forcing below both read this flag.
	// Cap scales with the per-pixel sample budget: a full-res disc only stays
	// hole-free while the Vogel spacing (~R*sqrt(pi/N)) is below the mip-0
	// footprint (~2px). Extreme (up to 1024 taps) covers ~32px radius cleanly;
	// Fast's tiny budget keeps the band small. Above the cap the disc is too big
	// to fill at full res, so large bokeh stay on the mip path.
	const PF_FpLong band_cap = (s.render_mode == 3) ? 32.0
	                          : (s.render_mode == 1) ? 10.0 : 22.0;
	bool fullres_band = false;
	if (!uniform_blur && effective_radius <= band_cap) {
		const A_long bcx = ClampValue<A_long>(static_cast<A_long>(u * cache_w_f), 0, cache_w - 1);
		const A_long bcy = ClampValue<A_long>(static_cast<A_long>(v * cache_h_f), 0, cache_h - 1);
		// The dark edge halo forms on every background pixel whose blur disc
		// OVERLAPS the silhouette (up to one radius away), so the band must fire
		// wherever a CoC discontinuity lies within the gather reach.
		const PF_FpLong reach = ClampValue<PF_FpLong>(effective_radius, 2.0, 48.0);
		if (coc_disc_dist) {
			// Precomputed in RenderCore (one chamfer distance transform over the
			// whole frame): the distance from this pixel to the nearest CoC
			// discontinuity. A single read replaces the 16-tap scatter scan, which
			// was a per-pixel cache-miss storm in the hottest pre-loop stage.
			fullres_band =
				(coc_disc_dist[static_cast<size_t>(bcy) * cache_w + bcx] <= reach);
		} else {
			// Fallback scan (callers that pass no field): 8 directions at half-
			// and full-reach; any depth discontinuity within reach flips this
			// pixel to clean full-res taps.
			const A_long pr = ClampValue<A_long>(
				static_cast<A_long>(effective_radius + 0.5), 2, 48);
			static const int dxs[8] = { 1, -1, 0, 0, 1, -1, 1, -1 };
			static const int dys[8] = { 0, 0, 1, -1, 1, -1, -1, 1 };
			PF_FpLong bmax = 0.0;
			for (int bk = 0; bk < 8; ++bk) {
				for (A_long frac = 1; frac <= 2; ++frac) {
					const A_long off = (pr * frac) / 2;
					const A_long sx = ClampValue<A_long>(bcx + dxs[bk] * off, 0, cache_w - 1);
					const A_long sy = ClampValue<A_long>(bcy + dys[bk] * off, 0, cache_h - 1);
					const PF_FpLong dd =
						signed_coc_cache[static_cast<size_t>(sy) * cache_w + sx] - center_signed_coc;
					const PF_FpLong ad = std::abs(dd);
					if (ad > bmax) bmax = ad;
				}
			}
			fullres_band = (bmax > 0.03);
		}
	}

	// Per-pixel field factor for the opponent-axis CA: real fringing grows
	// gently away from the optical centre. The per-tap defocus scaling and
	// the focal-plane sign flip happen in the tap loop.
	PF_FpLong ca_field = 0.0;
	// Frame-constant opponent-axis CA matrix rows (only the per-tap defocus
	// factor f varies inside the loop, so these never need recomputing there).
	const PF_FpLong ca_mr =  s.ca_rc       - 0.5 * s.ca_gm - 0.5 * s.ca_by;
	const PF_FpLong ca_mg = -0.5 * s.ca_rc + s.ca_gm       - 0.5 * s.ca_by;
	const PF_FpLong ca_mb = -0.5 * s.ca_rc - 0.5 * s.ca_gm + s.ca_by;
	if (has_ca) {
		const PF_FpLong ca_cu = u - 0.5;
		const PF_FpLong ca_cv = v - 0.5;
		const PF_FpLong ca_dist = std::sqrt(ca_cu * ca_cu + ca_cv * ca_cv);
		ca_field = 0.55 + ca_dist * 0.9;
	}
	// Per-pixel cat's-eye setup hoisted out of the tap loop (the field sqrt and
	// barrel offset are pixel-constant; only the disc test is per-tap).
	const CatsEyeSetup cats_eye = has_vig
		? MakeCatsEye(u, v, s.vignetting, s.vignetting_scale)
		: CatsEyeSetup{};

	A_long desired_N = PickVogelLUT(blur_norm, s.sample_count);

	// Anisotropy factor. Astigmatism stretches the iris tangentially by
	// (1 + 0.6*astig) at the frame edge, and anamorphic_ratio squeezes
	// the horizontal axis by that ratio -- Helios 58 (astig 0.8) alone is
	// a ~1.48x ellipse, and Helios + Anamorphic 2x combine to almost
	// 3x along the worst axis. The gather resamples from a single
	// isotropic mip level, so if we size the footprint for the short
	// axis the long axis gets under-sampled: the Vogel constellation
	// shows through as the horizontal "stripey tower" artifact and
	// every bokeh carries a visible sampling texture. Inflating the
	// effective radius by the worst-case stretch feeds the mip picker
	// and the footprint-based N cap with the *long-axis* footprint,
	// so the mip level rolls one or two steps higher and the
	// bilinear taps cover the longer inter-sample spacing cleanly.
	const PF_FpLong astig_stretch = has_astig ? (1.0 + std::min<PF_FpLong>(2.0, s.astigmatism) * 0.6) : 1.0;
	const PF_FpLong anam = std::max<PF_FpLong>(0.1, s.anamorphic_ratio);
	const PF_FpLong anam_stretch = std::max(anam, 1.0 / anam);
	const PF_FpLong aniso_factor = astig_stretch * anam_stretch;
	const PF_FpLong eff_radius_long_axis = effective_radius * aniso_factor;

	// Footprint-aware sample cap. Once the mip level is chosen, each tap's
	// bilinear footprint covers ≈ 2*2^M source pixels. Packing taps denser
	// than the footprint is pure waste (all extra taps sample the same
	// box-averaged texel). We compute N_needed from the mip-space disc area
	// and cap the tap count at ~1.6× that -- enough blue-noise overlap for
	// clean coverage without paying for redundant fetches. Typical 4K hit:
	// saves ~30-50% of taps on medium/large bokeh with zero quality loss.
	// We feed the *long-axis* footprint to the cap so elliptical bokeh do
	// not get their sample count clipped below what's required to fill
	// the stretched axis.
	{
		A_long mip_est = PickMipLevel(eff_radius_long_axis, desired_N, pyramid.num_levels);
		// Quality mip bias: Fast keeps the cheapest (coarsest) mip the sample
		// spacing allows; Final/Extreme step to a FINER mip so large bokeh stop
		// showing the coarse box-filter texels as blocky "stair-steps". A finer
		// mip needs proportionally more taps to stay gap-free, which the cap
		// below grants automatically because we shrink `footprint` by the same
		// bias. (mip_bias_footprint here; the colour mip is biased identically
		// further down.)
		const PF_FpLong mip_bias = 0.0;   // (rolled back: quality mip bias removed)
		// In the full-res silhouette band the colour taps are forced to mip 0
		// below, so size the budget for the mip-0 footprint (=2.0) -- otherwise
		// the coarse-mip footprint would cap N too low and the full-res disc
		// would show sampling holes.
		const PF_FpLong footprint = fullres_band
			? 2.0
			: 2.0 * std::pow(2.0, std::max<PF_FpLong>(0.0, mip_est - mip_bias));
		const PF_FpLong n_needed = kPi * effective_radius * effective_radius * aniso_factor
		                         / std::max<PF_FpLong>(1.0, footprint * footprint);
		// Extreme render mode relaxes the cap headroom: denser-than-footprint
		// taps are normally waste, but at huge CoC the extra blue-noise
		// overlap measurably cleans bokeh edge gradients on hero shots.
		const PF_FpLong cap_headroom = (s.render_mode == 3) ? 2.5 : 1.6;
		const A_long n_cap = static_cast<A_long>(std::ceil(n_needed * cap_headroom)) + 24;
		desired_N = std::min(desired_N, std::max<A_long>(48, n_cap));
	}

	// Pick the smallest LUT whose count is >= desired (ceil, not floor -- the
	// old floor path routinely dropped us to 512 when we asked for 600 and
	// produced visible under-coverage).
	A_long best_lut = num_luts - 1;
	for (A_long li = 0; li < num_luts; ++li) {
		if (luts[li].count >= desired_N) { best_lut = li; break; }
	}
	const VogelLUT& lut = luts[best_lut];
	const A_long actual_N = lut.count;

	// Mask angular sensitivity: concave star blades, notched irises and
	// aperture maps carry steep ANGULAR detail in the baked mask. The mask
	// weights stay at their baked angles while the blue-noise rotation spins
	// the sampling pattern per pixel, so a large rotation misaligns weight
	// vs position and the IGN pattern prints into the disc as structured
	// dithering (user-visible on star bokeh). Such irises get a token
	// rotation plus stochastic jitter instead; smooth irises keep the full
	// anti-constellation rotation.
	const bool mask_angular = (s.aperture_shape_mode == 3) ||
	                          (s.aperture_shape_mode == 2 && s.blade_curve < 0.15) ||
	                          (s.aperture_map_index > 0);

	// Stochastic per-sample jitter to break the regular Vogel lattice when the
	// iris is strongly anisotropic (anamorphic / astigmatism). A fixed-
	// orientation elliptical lattice otherwise shows up as a moiré "mesh" even
	// at 1024 samples; jittering each tap by up to ~half the inter-sample
	// spacing turns that structured pattern into fine, far less visible noise.
	// Off for round irises (aniso ≈ 1) so the common path stays identical.
	// Angular-detailed masks also jitter (their rotation budget is tiny).
	const bool jitter_samples = (aniso_factor > 1.05) || mask_angular;
	// Per-axis jitter: the iris semi-extent is effective_radius/anamorphic
	// horizontally and effective_radius vertically, so an elongated iris has
	// proportionally wider sample spacing along its long axis. One isotropic
	// amount cannot break the lattice there -- at Aspect Ratio 0 the iris is
	// stretched 10x in x and the x jitter fell 10x short, leaving the sampling
	// grid visible as a fine cross-hatch over defocused texture.
	const PF_FpLong jit_anam = std::max<PF_FpLong>(0.1, s.anamorphic_ratio);
	const PF_FpLong jit_base = 1.7725 * 0.55
	                         / std::sqrt(static_cast<PF_FpLong>(std::max<A_long>(1, actual_N)));
	const PF_FpLong jit_amt_u = (effective_radius / jit_anam) * jit_base * inv_w;
	const PF_FpLong jit_amt_v =  effective_radius             * jit_base * inv_h;

	// Pick the mip level that matches the inter-sample spacing on the
	// long axis. Each tap now reads from a texel that has already been
	// box-averaged over the right neighbourhood, so huge bokeh no longer
	// reveal the Vogel pattern as a constellation of tiny circles and
	// the custom aperture texture keeps its detail regardless of blur
	// size. `ceil` selection guarantees the footprint is never smaller
	// than the inter-sample spacing, which is what fixes the
	// horizontal/vertical streaking at anamorphic 2..4 and the
	// astigmatism banding at Helios: the former `round` mip dropped us
	// one level too low, leaving 1-2px gaps between taps that lined up
	// along the ellipse's long axis.
	// Integer `mip` (ceil) drives the anisotropic multi-tap path; `mip_f` is
	// the continuous LOD used for trilinear filtering of the colour taps,
	// which removes the blocky box-filter look on large bokeh.
	// Bias the colour mip to a FINER level in Final/Extreme (see the cap above):
	// the extra taps the cap granted now read smaller, sharper texels, so the
	// large-bokeh box-filter "blockiness" dissolves into smooth gradients.
	const PF_FpLong q_mip_bias = 0.0; // (rolled back: quality mip bias removed)
	A_long mip = std::max<A_long>(0,
		PickMipLevel(eff_radius_long_axis, actual_N, pyramid.num_levels)
		- static_cast<A_long>(q_mip_bias + 0.5));
	PF_FpLong mip_f = std::max<PF_FpLong>(0.0,
		PickMipLevelF(eff_radius_long_axis, actual_N) - q_mip_bias);
	// ── Radius-aware colour-mip floor (firefly / source-texture-echo fix) ────
	// The mip above is sized purely so the Vogel footprint has no GAPS, which
	// makes it go FINER as the sample count rises. At Extreme sample counts a
	// genuinely defocused region therefore samples the colour pyramid at (or
	// near) full-res, so every tap reads SHARP source detail instead of a
	// pre-averaged footprint: the sharp source texture is smeared across the
	// bokeh as a fingerprint "echo", and the per-tap highlight weighting
	// (boost + bokeh gamma) amplifies its bright textels into the crawling
	// "speckle" the user reported -- and *more* samples made it WORSE, not
	// better. The integral of a bokeh disc must average over a footprint that
	// is a fixed fraction of the disc radius, independent of how many taps we
	// throw at it; we enforce that with a floor on the footprint (hence the
	// mip). It is radius-tied, so tiny/near-focus CoC keeps mip 0 and stays
	// razor-sharp -- only real defocus gets the floor.
	//
	// v3.1: the floor is no longer a hard constant. It was only ever needed to
	// hide the WEIGHT nonlinearity -- boost and bokeh gamma raise a tap's weight
	// by a power of its luma, so one sharp bright texel fires as a crawling
	// speckle, and more samples made it worse because a higher tap count drives
	// the spacing mip FINER. The colour never needed the blur. So the weighting
	// luma is now read at `mip_w` (the old 0.35 footprint, rock steady) while the
	// colour keeps `mip_f`, and Bokeh Definition dials the colour floor away.
	// At definition 1 the floor is inert and the Vogel spacing -- which is
	// gap-free by construction -- is the only limit, so this can never introduce
	// sampling holes, only reveal how much tap budget the user has bought.
	PF_FpLong mip_w = mip_f;
	{
		// The legacy floor, still the reference for the WEIGHT LOD. Cranking
		// Bokeh Brightness Boost amplifies bright source texels in the weighted
		// average, so the stable LOD gets coarser as the boost climbs.
		const PF_FpLong k_legacy = 0.35 +
			(energy ? 0.0 : std::min<PF_FpLong>(0.30, s.highlight_boost * 0.08));
		const PF_FpLong fp_stable   = std::max<PF_FpLong>(2.0, effective_radius * k_legacy);
		mip_w = std::max(mip_f, std::log2(fp_stable * 0.5));

		const PF_FpLong k_floor = k_legacy * (1.0 - Clamp01(s.bokeh_definition));
		const PF_FpLong fp_floor   = std::max<PF_FpLong>(2.0, effective_radius * k_floor);
		const PF_FpLong mipf_floor = std::log2(fp_floor * 0.5); // >= 0
		if (mipf_floor > mip_f) mip_f = mipf_floor;
		const A_long mip_floor_i = static_cast<A_long>(std::ceil(mipf_floor - 1e-6));
		if (mip_floor_i > mip) mip = mip_floor_i;
		const A_long maxL = pyramid.num_levels - 1;
		if (mip   > maxL) mip   = maxL;
		if (mip_f > static_cast<PF_FpLong>(maxL)) mip_f = static_cast<PF_FpLong>(maxL);
		if (mip_w > static_cast<PF_FpLong>(maxL)) mip_w = static_cast<PF_FpLong>(maxL);
		if (mip_w < mip_f) mip_w = mip_f;
	}
	// Silhouette band: read every colour tap from full-res level 0 so the depth
	// gate rejects wrong-plane colour before it is averaged (Frischluft parity).
	// Forcing mip 0 here also disables the anisotropic multi-tap path below,
	// which requires mip > 0.
	// The weight LOD follows: a coarse texel straddling the silhouette is an
	// FG+BG blend, and the band radii are small enough that speckle is not a
	// risk there anyway.
	if (fullres_band) {
		mip = 0;
		mip_f = 0.0;
		mip_w = 0.0;
	}
	// One extra pyramid fetch per tap, so only pay it where a weight
	// nonlinearity actually exists AND the two LODs differ enough to matter.
	const bool weight_lod_split =
		ZluxWeightLodSplit()
		&& (has_highlight || has_bokeh_gamma || s.highlight_scatter > 0.001)
		&& (mip_w > mip_f + 0.05);

	Color3 acc{0.0, 0.0, 0.0};
	PF_FpLong w_sum = 0.0;        // for color normalization (brightness-weighted)
	PF_FpLong potential_sum = 0.0; // aperture-shape area (denominator for alpha)
	// GEOMETRIC accepted coverage -- the sum of (mask*gate) WITHOUT the
	// highlight-boost / bokeh-gamma weighting. The opacity alpha must be the
	// geometric fraction of the aperture filled, not the colour-weighted sum:
	// using w_sum there let a non-zero Bokeh Gamma (every vintage preset sets
	// it) inflate the alpha ~1.5x and harden silhouette edges. rgb keeps using
	// the weighted w_sum for the punchy-bokeh look.
	PF_FpLong cover_sum = 0.0;

	// Self-contribution -------------------------------------------------------
	// Far: if the center pixel is itself far-blurred, seed with its color.
	// Near: if the center pixel is itself in the near field, it is trivially
	//       fully covered (its disc contains itself).
	//
	// Center-seed thresholds are *deliberately* wider than the per-sample
	// rejection gate below (0.008 vs 0.006). Seeding the gather with the
	// sharp centre color when center_signed_coc is tiny (AA fringe, depth
	// ramp) is what produces the classic "bright halo around sharp edges"
	// that the user called out against DOF PRO's reference output. Letting
	// the gather pick its colors exclusively from ring samples kills the
	// halo even when the depth map bleeds a few pixels at edges.
	//
	// However the seed reads the center from mip 0 (full-res source) while
	// every ring sample reads from the *downsampled* mip selected for the
	// current blur radius. For deep defocus that single full-res tap lands
	// on top of the output pixel as a 1-pixel bright spike -- exactly the
	// "dot in the middle of every bokeh" artifact the user flagged. We
	// ramp the seed weight down as the CoC grows: full strength where it
	// suppresses halos (tiny CoC, edge transitions), silent for deeply
	// defocused highlights (CoC past ~0.04) where the gather already has
	// thousands of ring samples covering the bokeh. The spherical-profile
	// mask at r=0 is also folded in so bubble/donut irises don't carry a
	// stray central dot through the seed path.
	//
	// The `bokeh_gamma` non-linear boost is deliberately *not* applied to
	// the seed: gamma is a ring-sample knob for the defocused gather, and
	// boosting a single centre tap produces visible "punched-in" specks
	// rather than smooth highlight rolloff.
	// Near pass only: maximum splash reach observed among contributing taps
	// (plus the centre seed). The search disc radius comes from the tile-
	// dilated near maximum, which strong blur ANYWHERE nearby inflates far
	// beyond what the near content around this pixel can actually cover; the
	// alpha renormalization at the bottom uses this to stay independent of
	// that inflation (see comment there).
	PF_FpLong near_max_reach = 0.0;

	const PF_FpLong seed_coc_abs = std::abs(center_signed_coc);
	const PF_FpLong seed_gate = 1.0 - SmoothStep(0.015, 0.04, seed_coc_abs);
	const PF_FpLong seed_spher = has_spher ? ComputeSphericalProfile(0.0, center_depth, s) : 1.0;
	if constexpr (MODE == DofPass::Far) {
		if ((uniform_blur || center_signed_coc > 0.008) && seed_gate > 0.001) {
			PF_FpLong cw_geom = seed_gate * seed_spher;
			if (has_cata) cw_geom *= GetCatadioptricMask({0.0, 0.0}, s.catadioptric);
			const PF_FpLong cw = has_highlight
				? cw_geom * (1.0 + Luma(center_color_lin) * s.highlight_boost * 8.0) : cw_geom;
			acc.r += center_color_lin.r * cw;
			acc.g += center_color_lin.g * cw;
			acc.b += center_color_lin.b * cw;
			if (do_alpha) a_acc += center_alpha_lin * cw;
			w_sum += cw;
			potential_sum += cw_geom;
			cover_sum += cw_geom;
			near_max_reach = std::max(near_max_reach, std::max<PF_FpLong>(0.0, center_signed_coc));
		}
	} else {
		// Near pass keeps the seed at full strength: most ring samples fall
		// in the Far field at near-focus pixels (foreground subject against
		// distant background) and get rejected by the Near gate, so without
		// a centre seed the near gather can run nearly empty. The SA-at-r=0
		// attenuation is still applied for consistency with donut bokeh.
		if (!uniform_blur && center_signed_coc < -0.008) {
			PF_FpLong cw_geom = seed_spher;
			if (has_cata) cw_geom *= GetCatadioptricMask({0.0, 0.0}, s.catadioptric);
			const PF_FpLong cw = has_highlight
				? cw_geom * (1.0 + Luma(center_color_lin) * s.highlight_boost * 8.0) : cw_geom;
			acc.r += center_color_lin.r * cw;
			acc.g += center_color_lin.g * cw;
			acc.b += center_color_lin.b * cw;
			if (do_alpha) a_acc += center_alpha_lin * cw;
			w_sum += cw;
			potential_sum += cw_geom;
			cover_sum += cw_geom;
			near_max_reach = -center_signed_coc;
		}
	}

	// Pre-scaled coefficients for the inner loop: the sample UV offset is
	// just the baked kx/ky times the per-pass radius (and viewport size).
	const PF_FpLong pos_coef_x = effective_radius * inv_w;
	const PF_FpLong pos_coef_y = effective_radius * inv_h;

	// -------- Per-pixel blue-noise rotation (#3) -----------------------------
	// The Vogel spiral is deterministic: every output pixel gathers samples
	// along the same golden-angle lattice, so smooth defocused regions carry
	// the spiral's structure as a faint "constellation" texture visible once
	// the mip footprint drops below cell spacing. Rotating the sampling
	// pattern per-pixel by a blue-noise angle (Jimenez interleaved-gradient
	// noise is free, high quality, and temporally stable for any (u,v))
	// breaks up that coherence -- adjacent output pixels read different
	// sample positions, so the residual pattern averages out visually.
	//
	// We cap the rotation at ±10° so the aperture mask (baked at the global
	// rotation in FinalizeVogelLUT) still lines up with the per-pixel
	// rotated Vogel sample to within a few percent of its angular span --
	// even for polygonal irises with 6 blades the mask value changes by
	// less than 3% over a 10° sweep, which is invisible next to the
	// blue-noise win. Circular / custom-texture modes are rotationally
	// symmetric so the mask match is exact.
	const PF_FpLong px_f = u / std::max<PF_FpLong>(inv_w, 1e-9);
	const PF_FpLong py_f = v / std::max<PF_FpLong>(inv_h, 1e-9);
	const PF_FpLong ign_raw = std::fmod(
		52.9829189 * std::fmod(0.06711056 * px_f + 0.00583715 * py_f, 1.0),
		1.0);
	const PF_FpLong bn_angle = (ign_raw - 0.5) *
		(mask_angular ? (kPi / 120.0) : (kPi / 9.0)); // ±1.5° vs ±10°
	const PF_FpLong cos_bn = std::cos(bn_angle);
	const PF_FpLong sin_bn = std::sin(bn_angle);

	// -------- Anisotropic multi-tap major axis (#2) --------------------------
	// When the iris is strongly elongated (Helios astigmatism, anamorphic),
	// our mip picker uses eff_radius_long_axis and picks a blurrier level so
	// the long-axis inter-sample spacing stays covered. That's safe but
	// costs sharpness on the short axis. Multi-tap compensates: along the
	// ellipse's long axis we take 2 taps at a FINER mip (mip-1, half
	// footprint), so the composite footprint still covers the long axis
	// spacing but the texture detail along the short axis is resolved at
	// full mip-1 resolution. Net result: Helios + anamorphic bokeh keep
	// the crisp edge definition that the old single-tap path lost to the
	// aniso mip bump.
	const bool use_multitap = (aniso_factor > 1.25) && (mip > 0) && !has_ca;
	Vec2 major_axis = {1.0, 0.0};
	PF_FpLong tap_offset_px = 0.0;
	if (use_multitap) {
		if (has_astig) {
			const PF_FpLong cx = u - 0.5;
			const PF_FpLong cy = v - 0.5;
			const PF_FpLong d2 = cx * cx + cy * cy;
			if (d2 > 1e-6) {
				const PF_FpLong inv_d = 1.0 / std::sqrt(d2);
				if (s.astigmatism_type_sagittal) {
					major_axis = {cx * inv_d, cy * inv_d}; // radial = sagittal
				} else {
					major_axis = {-cy * inv_d, cx * inv_d}; // tangential swirl
				}
			}
		} else {
			// No astigmatism: long axis is determined by anamorphic squeeze.
			// anamorphic_ratio > 1 squeezes horizontally so vertical is long.
			major_axis = (s.anamorphic_ratio > 1.0) ? Vec2{0.0, 1.0} : Vec2{1.0, 0.0};
		}
		// Offset = half the *long-axis* cell spacing on mip-1 (one footprint
		// of the finer level). Stays well inside the mip-level footprint of
		// the current coarser mip, so there are no seams between taps.
		const PF_FpLong cell_long_px = eff_radius_long_axis * 1.7725
		                             / std::sqrt(static_cast<PF_FpLong>(std::max<A_long>(1, actual_N)));
		tap_offset_px = cell_long_px * 0.25;
	}
	const PF_FpLong tap_dx_u = major_axis.x * tap_offset_px * inv_w;
	const PF_FpLong tap_dy_v = major_axis.y * tap_offset_px * inv_h;

	// -------- Sprite-scatter accumulator (#1) --------------------------------
	// Physical gather divides by sum-of-weights, which flattens bright point
	// sources into the dim surrounding average. DOF PRO's crisp specular
	// bokeh comes from an additive scatter contribution: every sample that
	// clears the highlight threshold accumulates into a separate bucket
	// *without* normalisation, then that bucket is layered on top of the
	// normalised gather result. Point sources stay at their source intensity,
	// clusters of bright pixels additively summed. Strength is user-driven
	// via the new Highlight Scatter slider; at 0 the whole path compiles out
	// to a couple of floats doing nothing.
	const bool has_scatter = s.highlight_scatter > 0.001;
	// Preservative mode folds the specular emphasis into the gather WEIGHTS
	// instead of layering an un-normalised bucket on top. Because the result
	// stays Sigma(col*w)/Sigma(w), it is a convex combination of the sampled
	// colours: it can never exceed the brightest sample, so a bright point
	// source concentrates into a crisp iris without pushing the frame past its
	// own peak and clipping to flat white. Energy is redistributed, not added.
	const bool preservative = has_scatter && (s.highlight_mode == 1);
	const bool additive_scatter = has_scatter && !preservative;
	constexpr PF_FpLong kPreserveGain = 8.0;
	Color3 spec_acc{0.0, 0.0, 0.0};

	for (A_long i = 0; i < actual_N; ++i) {
		const VogelSample& vs = lut.samples[i];

		// Fast skip for samples that are fully masked by a frame-constant
		// aperture feature (notched blade tip, catadioptric ring, matte
		// flags, softness falloff). This is the biggest win of the bake:
		// whole classes of samples never touch the sampler or depth cache.
		if (!is_custom_tex && vs.static_mask < 0.001) continue;

		// Order of operations matters when anamorphic_ratio != 1.
		//   astigmatism stretches the iris tangentially around the frame
		//   centre (that's what produces the Helios-style swirl), while
		//   anamorphic_ratio is the cylindrical element that squeezes
		//   the projection horizontally. Composing them in the wrong
		//   order (anamorphic-then-astigmatism) feeds a squashed ellipse
		//   into the stretch and cancels the swirl along the anamorphic
		//   axis -- the exact failure the user flagged on Helios +
		//   Aspect Ratio 2.23.
		//
		// Correct composition: do astigmatism on the un-squashed iris
		// circle, then apply the anamorphic squash on the final offset
		// so the projection step is the last thing that touches the
		// sample position. vs.kx already has anamorphic baked in for
		// the fast non-astigmatism path, so we branch.
		// Apply the per-pixel blue-noise rotation to the Vogel direction
		// vector before astigmatism / anamorphic composition. This rotates
		// the sampling pattern without touching the aperture mask baked
		// into vs.static_mask, so the mask alignment stays consistent and
		// we simply read from slightly different source positions across
		// adjacent output pixels.
		Vec2 pos_uv;
		if (has_astig) {
			// The polar (cos_a, sin_a) rotation is ONLY consumed on the
			// astigmatism path (the round-iris branch rotates the baked kx/ky
			// instead), so compute it here rather than on every tap -- on the
			// dominant non-astigmatism path this saved a dead 4-mul/2-add per tap.
			const PF_FpLong cos_a_r = vs.cos_a * cos_bn - vs.sin_a * sin_bn;
			const PF_FpLong sin_a_r = vs.cos_a * sin_bn + vs.sin_a * cos_bn;
			const PF_FpLong anam = std::max<PF_FpLong>(0.1, s.anamorphic_ratio);
			Vec2 iris = {cos_a_r * vs.fr * pos_coef_x,
			             sin_a_r * vs.fr * pos_coef_y};
			iris = ApplyAstigmatism(iris, u, v, s.astigmatism, s.astigmatism_type_sagittal);
			pos_uv = {iris.x / anam, iris.y};
		} else {
			// Rotate on the UNSQUASHED disc, then re-apply the anamorphic
			// squeeze. vs.kx already carries the 1/anamorphic factor, so
			// rotating it directly is R(theta)*S*v, which rotates the ELLIPSE
			// itself. Harmless for a circular iris, but with an anamorphic one
			// every pixel gets its bokeh tilted by its own blue-noise angle and
			// the blur direction wobbles pixel to pixel -- visible as ribbing
			// along edges. S*R*v keeps the ellipse fixed and moves only the
			// samples, which is all the rotation was ever meant to do.
			const PF_FpLong anam_n = std::max<PF_FpLong>(0.1, s.anamorphic_ratio);
			const PF_FpLong kxu = vs.kx * anam_n;      // back to cos_a * fr
			const PF_FpLong rx  = kxu * cos_bn - vs.ky * sin_bn;
			const PF_FpLong ry  = kxu * sin_bn + vs.ky * cos_bn;
			pos_uv = {(rx / anam_n) * pos_coef_x, ry * pos_coef_y};
		}
		PF_FpLong su = u + pos_uv.x;
		PF_FpLong sv = v + pos_uv.y;
		if (jitter_samples) {
			// Cheap per-(pixel, sample) hash (golden-ratio additive fract, no
			// trig) -> ±0.5 jitter in cell units.
			const PF_FpLong h1 = std::fmod(ign_raw + i * 0.7548776662, 1.0);
			const PF_FpLong h2 = std::fmod(ign_raw * 1.3247179572 + i * 0.5698402910, 1.0);
			su += (h1 - 0.5) * jit_amt_u;
			sv += (h2 - 0.5) * jit_amt_v;
		}

		// Baked static mask covers polygonal / notch / catadioptric /
		// matte-box / softness. Per-pixel-only masks are folded on top.
		PF_FpLong mask = vs.static_mask;
		if (is_custom_tex) {
			// Custom-texture mode cannot bake the aperture mask (the
			// texture offset and the per-bokeh rotation both depend on
			// the sampled source position), so we still sample it here
			// and combine it with the baked softness roll-off. Passing
			// (su, sv) activates the per-bokeh rotation randomization so
			// every distinct source point shows the texture at its own
			// orientation instead of the whole frame tiling identically.
			mask = SampleApertureTextureMask(aperture_tex_world, {vs.norm_x, vs.norm_y}, u, v, s, su, sv) * vs.soft_edge;
			if (mask < 0.001) continue;
		} else if (iris_mod_world) {
			// Iris Texture modulator: multiply the baked polygonal /
			// circular / notched shape by a user-supplied texture layer
			// so dust, vintage coating, blade imperfections, etc. show
			// up on any iris shape. Uses the same sampling helper (and
			// the same Intensity / Scale / Offset / Invert sliders) as
			// Custom-shape mode; when the user loads the layer the
			// feature just lights up.
			mask *= SampleApertureTextureMask(iris_mod_world, {vs.norm_x, vs.norm_y}, u, v, s, su, sv);
			if (mask < 0.001) continue;
		}

		if (has_spher) {
			// Baked profile (sa_pos = pow(fr,sharp), sa_neg = pow(1-fr,sharp))
			// lerped by strength -- identical to ComputeSphericalProfile(vs.fr).
			const PF_FpLong sa_profile = sa_use_pos ? vs.sa_pos : vs.sa_neg;
			mask *= Mix(1.0, sa_profile * sa_norm, sa_t);
			if (mask < 0.001) continue;
		}

		if (has_vig) {
			mask *= EvalCatsEye(cats_eye, vs.norm_x, vs.norm_y);
			if (mask < 0.001) continue;
		}

		potential_sum += mask;

		// Pass-specific sample acceptance --------------------------------
		PF_FpLong sample_signed_coc;
		if (uniform_blur) {
			sample_signed_coc = radius; // everything behaves as far
		} else {
			const A_long sx = ClampValue<A_long>(static_cast<A_long>(su * cache_w_f), 0, cache_w - 1);
			const A_long sy = ClampValue<A_long>(static_cast<A_long>(sv * cache_h_f), 0, cache_h - 1);
			sample_signed_coc = signed_coc_cache[static_cast<size_t>(sy * cache_w + sx)];
		}

		PF_FpLong gate;
		// Soft plane-membership weight. OUTSIDE the silhouette band this stays a
		// hard cutoff (1.0 once admitted) so every prior halo/tint fix is intact
		// and interior bokeh is bit-identical. INSIDE the band we replace the hard
		// near/far cliff with a narrow smoothstep crossfade around the focus plane
		// so the Far and Near passes overlap slightly -- emulating Frischluft's
		// single unified gather, which leaves no sharp residual contour at the
		// foreground<->background boundary. The band is ~0.012 wide and centred
		// just past zero, so genuinely focused samples are still mostly rejected.
		PF_FpLong plane_w = 1.0;
		if constexpr (MODE == DofPass::Far) {
			// Reject near *and* focused samples. Threshold widened from
			// 0.003 -> 0.006 to filter out depth-map anti-aliased edge
			// pixels that previously leaked the sharp foreground colour
			// into the background gather, producing the "bright halo"
			// artefact around sharp subjects in the DOF PRO comparison
			// image the user flagged. Pairs with the 0.008 centre-seed
			// threshold above (always slightly wider to kill halo seed
			// bias without starving the gather).
			if (fullres_band) {
				plane_w = SmoothStep(-0.004, 0.010, sample_signed_coc);
				if (plane_w < 0.01) continue;
			} else if (sample_signed_coc < 0.006) {
				continue;
			}
			// A far sample contributes if its own CoC is at least as big as
			// the ring we are gathering from (classic depth-aware gather).
			const PF_FpLong req = vs.fr * radius;
			const PF_FpLong softness = std::max<PF_FpLong>(0.02, radius * 0.15);
			gate = SmoothStep(std::max<PF_FpLong>(0.0, req - softness),
			                  req + softness,
			                  std::max<PF_FpLong>(0.0, sample_signed_coc));
			// Occlusion gate (depth-aware edge-leak fix). The reach gate above
			// admits any far sample whose disc is big enough -- including a
			// CLOSER far surface in front of this (farther) centre. A real lens
			// cannot see behind that nearer surface, so admitting it drags the
			// nearer object's colour across the silhouette as a dark/bright
			// fringe (the classic background "edge leak"). Reject in proportion
			// to how much NEARER the sample sits than the centre, normalised by
			// the gather radius: a smooth far gradient (tiny step) passes
			// untouched, a true near<->far boundary is cut. The reverse
			// direction (a farther sample onto a nearer centre) is intentionally
			// NOT gated, so the physical background-over-foreground soft bleed
			// and the far bleed-over downstream survive.
			const PF_FpLong occl = SmoothStep(0.10, 0.32,
				(center_signed_coc - sample_signed_coc) / std::max<PF_FpLong>(0.04, radius));
			gate *= 1.0 - occl;
			gate *= plane_w;
			if (gate < 0.01) continue;
			// Track the largest accepted far reach (mirrors the Near pass):
			// feeds the area-coverage alpha used by the far bleed-over.
			if (sample_signed_coc > near_max_reach) near_max_reach = sample_signed_coc;
		} else {
			// Reject far *and* focused samples: the near layer must only
			// contain genuine foreground scatter, otherwise focused /
			// background samples tint the near bokeh and produce the
			// "everything is blurry" look. Threshold symmetric with the
			// Far gate (0.006) for consistent behaviour across passes.
			if (fullres_band) {
				plane_w = SmoothStep(-0.004, 0.010, -sample_signed_coc);
				if (plane_w < 0.01) continue;
			} else if (sample_signed_coc > -0.006) {
				continue;
			}
			// A near sample contributes only if its disc actually covers
			// this pixel, i.e. its CoC radius reaches our offset. (A band
			// sample admitted from just across the plane has near-zero reach
			// and is culled here, so the near bokeh never picks up background
			// tint -- the seam softening is effectively far-side only, which
			// is the dominant artifact direction.)
			const PF_FpLong sample_reach = -sample_signed_coc;
			const PF_FpLong required_reach = vs.fr * radius;
			const PF_FpLong softness = std::max<PF_FpLong>(0.02, radius * 0.15);
			gate = SmoothStep(std::max<PF_FpLong>(0.0, required_reach - softness),
			                  required_reach + softness,
			                  sample_reach);
			gate *= plane_w;
			if (gate < 0.01) continue;
			if (sample_reach > near_max_reach) near_max_reach = sample_reach;
		}

		PF_FpLong w = mask * gate;
		cover_sum += w;   // geometric coverage (pre brightness-weight) for alpha

		// Color sampling (with optional CA). All reads are against the
		// pre-linearized mip pyramid, so there is no per-tap gamma decode
		// and no per-tap int-to-float conversion -- the heavy lifting was
		// done once at pyramid-build time.
		Color3 col;
		if (has_ca) {
			const PF_FpLong plane_sign = (sample_signed_coc >= 0.0) ? 1.0 : -1.0;
			// Per-tap strength: a sample's per-channel focus spread grows
			// with ITS distance from the focal plane, so in-focus taps stay
			// clean, deep-bokeh taps fringe fully, and the sign flips across
			// the focal plane (LoCA behaviour).
			const PF_FpLong tap_defocus = Clamp01(std::abs(sample_signed_coc) * (1.0 / kCocClamp));
			const PF_FpLong f = ca_field * (0.35 + tap_defocus * 1.15) * plane_sign;
			// Opponent-axis mixing: each axis pushes its colour one way and
			// the complementary pair the other way at half weight, so every
			// axis is hue-balanced and the three sliders compose linearly.
			const PF_FpLong rs = ClampValue<PF_FpLong>(ca_mr * f, -2.8, 2.8);
			const PF_FpLong gs = ClampValue<PF_FpLong>(ca_mg * f, -2.8, 2.8);
			const PF_FpLong bs = ClampValue<PF_FpLong>(ca_mb * f, -2.8, 2.8);
			// Sample relative to the JITTERED tap position (su/sv) with the
			// per-channel scale added on top.
			col.r = SampleMipTrilinearCh(pyramid, mip_f, su + pos_uv.x * rs, sv + pos_uv.y * rs, 0);
			col.g = SampleMipTrilinearCh(pyramid, mip_f, su + pos_uv.x * gs, sv + pos_uv.y * gs, 1);
			col.b = SampleMipTrilinearCh(pyramid, mip_f, su + pos_uv.x * bs, sv + pos_uv.y * bs, 2);
		} else if (use_multitap) {
			// Anisotropic multi-tap: along the ellipse's long axis the
			// mip picker bumped us one level coarser to avoid holes
			// between samples, which softens short-axis detail. Replace
			// the single coarse tap with a symmetric pair at mip-1 offset
			// by ±tap_offset_px along the long axis; the averaged result
			// covers the same long-axis footprint but resolves texture on
			// the short axis at the finer mip's sharper filter kernel.
			const MipLevel& Lf = pyramid.levels[static_cast<size_t>(mip - 1)];
			const Color3 c0 = SampleMipLinear(Lf, su - tap_dx_u, sv - tap_dy_v);
			const Color3 c1 = SampleMipLinear(Lf, su + tap_dx_u, sv + tap_dy_v);
			col = { (c0.r + c1.r) * 0.5,
			        (c0.g + c1.g) * 0.5,
			        (c0.b + c1.b) * 0.5 };
		} else {
			col = SampleMipTrilinear(pyramid, mip_f, su, sv);
		}

		// ── Weighting colour: a SEPARATE, deliberately steadier LOD ─────────
		// Everything below this line that touches `w` is a nonlinearity in the
		// sample's brightness -- boost multiplies by luma, bokeh gamma raises a
		// power of it, the scatter mask is a hard-ish threshold on it. Feeding
		// those a razor-sharp mip means one blown texel inside an otherwise dull
		// neighbourhood grabs a huge share of the weighted average and fires as a
		// crawling speckle; that is what the old blanket footprint floor existed
		// to suppress, at the cost of every bokeh edge in the frame. Reading the
		// WEIGHT from the coarse, stable LOD and the COLOUR from the sharp one
		// separates the two concerns: weights vary smoothly across the disc while
		// the disc itself keeps a hard edge.
		const Color3 wcol = weight_lod_split
			? SampleMipTrilinear(pyramid, mip_w, su, sv)
			: col;
		// Cache luma once; used by highlight boost, bokeh gamma, and the
		// sprite-scatter threshold gate so we never recompute it.
		const PF_FpLong sample_luma = Luma(wcol);

		if (has_highlight) {
			// DOF PRO's "punchy bokeh" signature comes from preserving the
			// original highlight intensity through the gather. A stronger
			// luma-driven weight (8× vs 6×) re-concentrates the bright
			// samples in the weighted average, so defocused specular
			// highlights keep the crisp iris shape instead of melting into
			// the surrounding mid-tones (the "blurry bokeh" failure from
			// the user's screenshot).
			w *= 1.0 + sample_luma * s.highlight_boost * 8.0;
		}
		if (has_bokeh_gamma) {
			// (1 + luma)^gamma biases the weighted average toward bright
			// samples without suppressing darks: luma=0 gives boost=1 and
			// luma=1 at gamma=3 gives boost=8. Combined with the linear
			// highlight-boost term above this reproduces DOF PRO's punchy
			// specular cores (the bokeh donut keeps its iris shape instead
			// of averaging into a cloud) while leaving the mid-tone bokeh
			// intact. sample_luma is already in [0,1] (Luma clamps), so a
			// 256-interval LUT lerp replaces the per-tap pow() exactly within
			// interpolation error -- the curve is frame-constant.
			const PF_FpLong lf = sample_luma * 256.0;
			int li = static_cast<int>(lf);
			if (li > 255) li = 255;
			else if (li < 0) li = 0;
			const PF_FpLong lt = lf - static_cast<PF_FpLong>(li);
			w *= s.bokeh_gamma_lut[li] * (1.0 - lt) + s.bokeh_gamma_lut[li + 1] * lt;
		}

		if (preservative) {
			// Same specular test as the additive path, but it scales the tap's
			// weight rather than feeding a separate accumulator -- so it reads
			// the stable LOD too.
			const PF_FpLong sf = ComputeHighlightMask(wcol, s);
			if (sf > 0.001) w *= 1.0 + s.highlight_scatter * kPreserveGain * sf;
		}

		acc.r += col.r * w;
		acc.g += col.g * w;
		acc.b += col.b * w;
		w_sum += w;

		// Matte feathering: blur the premultiplied alpha with the SAME weight
		// `w` as the colour (highlight/bokeh-gamma boost included) so the
		// premult pair rgb=acc/w_sum, alpha=a_acc/w_sum stays consistent --
		// no dark/bright fringe at the silhouette. Sampled at the base tap
		// position (no chromatic offset on coverage). One extra single-channel
		// trilinear per accepted tap, compiled out entirely for opaque sources.
		if (do_alpha) {
			a_acc += SampleMipTrilinearCh(pyramid, mip_f, su, sv, 3) * w;
		}

		// Sprite-scatter: un-normalised accumulator. We reuse the already
		// gated mask * gate weight as the iris footprint (so the sprite has
		// the exact same polygonal/catadioptric/vignette shape as the gather
		// bokeh) and multiply in the highlight mask threshold so only
		// above-threshold samples contribute. Darks are not touched, so
		// the additive layer never brightens mid-tones -- only the sample
		// positions that the user considers "specular" splat into spec_acc.
		if (additive_scatter) {
			// Threshold on the stable LOD (it is a weight decision); the energy
			// that gets splatted is still the sharp colour.
			const PF_FpLong spec_factor = ComputeHighlightMask(wcol, s);
			if (spec_factor > 0.001) {
				const PF_FpLong sw = mask * gate * spec_factor;
				spec_acc.r += col.r * sw;
				spec_acc.g += col.g * sw;
				spec_acc.b += col.b * sw;
			}
		}
	}

	if (w_sum <= kEps) {
		if constexpr (MODE == DofPass::Far) {
			return { PercToLin(center_color_lin), 0.0, center_alpha_lin };
		} else {
			return { {0.0, 0.0, 0.0}, 0.0, 0.0 };
		}
	}

	// Perceptual-space weighted average; PercToLin() at the return converts the
	// whole (gather + scatter) result back to linear for the compositor.
	Color3 rgb = { acc.r / w_sum, acc.g / w_sum, acc.b / w_sum };

	// ── Info-density deficit fill (Frischluft coverage weighting) ───────────
	// Where the far gather is under-covered (cover_sum/potential_sum << 1: a far
	// pixel against a foreground silhouette, the thin wire/antenna case, or the
	// frame border where the disc is clipped) the few surviving samples read as a
	// foggy, washed patch. We blend the deficit back toward a fallback colour.
	//
	// CRITICAL: the fallback must itself be BLURRED. An earlier version filled
	// with the sharp centre pixel, which RE-SHARPENED every low-coverage region --
	// so frame edges and thin far structures (everything the Info-Density view
	// flags red) rendered crisp when they should be fully defocused. Sampling the
	// mip at the gather's own LOD (mip_f) gives a smooth, edge-aware-clean blurred
	// colour at the right scale, so the gap is filled with blur, not sharpness.
	if constexpr (MODE == DofPass::Far) {
		const PF_FpLong cov = Clamp01(cover_sum / std::max<PF_FpLong>(potential_sum, kEps));
		PF_FpLong deficit = Clamp01(1.0 - cov);
#ifdef ZLUX_PROFILE
		static const PF_FpLong kDeficitScale = []{ const char* e = std::getenv("ZLUX_DEFICIT"); return e ? atof(e) : 1.0; }();
		deficit *= kDeficitScale;
#endif
		if (deficit > 0.001) {
			const Color3 blurred_fill = SampleMipTrilinear(pyramid, mip_f, u, v);
			// Fill ONLY toward a DARKER background -- never brighter. The local
			// mip at a silhouette is contaminated by the (often bright) nearer
			// occluder; letting it brighten the accepted far average paints the
			// milky halo that hugs foreground edges. The accepted far samples are
			// already the correct background colour, so we only let the fill pull
			// an under-covered patch DOWN toward the local tone, never up into a
			// glow. Genuine far structures / frame borders (no bright occluder)
			// are unaffected -- there the local mip is the far content itself.
			rgb.r = Mix(rgb.r, std::min(rgb.r, blurred_fill.r), deficit);
			rgb.g = Mix(rgb.g, std::min(rgb.g, blurred_fill.g), deficit);
			rgb.b = Mix(rgb.b, std::min(rgb.b, blurred_fill.b), deficit);
		}
	}
	// Blurred matte: weighted-average alpha, the premult companion of rgb.
	// Sentinel for opaque sources stays neutral (Far: opaque center; Near: 0);
	// it is never read when !has_alpha.
	PF_FpLong matte;
	if (do_alpha) {
		matte = Clamp01(a_acc / w_sum);
	} else if constexpr (MODE == DofPass::Far) {
		matte = center_alpha_lin;
	} else {
		matte = 0.0;
	}
	if (additive_scatter) {
		// Normalise by actual sample count so the sprite layer is
		// independent of the Vogel LUT picked at this pixel (64 vs 1024
		// sample counts give the same visual intensity). At slider = 1
		// a 1-pixel specular source reaches the gather's peak exactly
		// (both contribute col/N_samples), so the bokeh peak doubles;
		// extended bright regions scale additively beyond that, which is
		// the expected "specular sprite stacks" behaviour. Users typically
		// dial 10-40% for a subtle punch, 60-100% for the full DOF PRO
		// crisp-iris look.
		const PF_FpLong scatter_scale = s.highlight_scatter
			/ std::max<PF_FpLong>(1.0, static_cast<PF_FpLong>(actual_N));
		rgb.r += spec_acc.r * scatter_scale;
		rgb.g += spec_acc.g * scatter_scale;
		rgb.b += spec_acc.b * scatter_scale;
	}

	if constexpr (MODE == DofPass::Far) {
		// Area-coverage alpha, same renormalized model as the Near pass:
		// the fraction of the aperture the accepted far content actually
		// covers from this pixel. The classic far compositing path ignores
		// this value (it blends by focus_mask); the far BLEED-OVER composite
		// reads it to wash defocused background over focused silhouettes
		// with a physically plausible falloff.
		const PF_FpLong r_loc = ClampValue<PF_FpLong>(near_max_reach, 1e-6, radius);
		const PF_FpLong area_frac = r_loc / std::max<PF_FpLong>(radius, 1e-6);
		const PF_FpLong alpha = (potential_sum > kEps)
			? Clamp01(cover_sum / std::max<PF_FpLong>(potential_sum * area_frac * area_frac, kEps))
			: 0.0;
		return { PercToLin(rgb), alpha, matte };
	} else {
		// Alpha = TRUE coverage fraction: how much of the local aperture disc
		// is filled by near content reaching this pixel. Energy conservation:
		// at the geometric silhouette ~half the aperture sees the subject, so
		// alpha = 0.5 -- the foreground is semi-transparent there and the
		// background shows through, exactly like a real out-of-focus edge.
		// (The old `* 0.5` saturated alpha at half-coverage, so the blurred
		// foreground stayed FULLY opaque half a CoC past its real silhouette
		// and then cut off hard -- the "black-blob-with-a-razor-edge" the
		// user kept hitting. Removing it restores the soft spreading edge.)
		//
		// Renormalization: `radius` is the tile-dilated near maximum, so a
		// strong-blur element ANYWHERE within the dilation reach (which is
		// global -- sized to the largest CoC in the frame) inflates the
		// search disc far beyond the splash range of the near content that
		// actually surrounds this pixel. Normalizing against the full
		// inflated disc diluted alpha by (local_reach / radius)² and made
		// every silhouette edge translucent at high Blur Amount -- the
		// "transparent edge band" artifact (background visibly leaking
		// through the subject's shoulder line). Scale the denominator down
		// to the disc the observed near content could plausibly cover (its
		// own maximum reach); when the disc was not inflated
		// (near_max_reach == radius) this reduces exactly to the old
		// behaviour.
		const PF_FpLong r_loc = ClampValue<PF_FpLong>(near_max_reach, 1e-6, radius);
		const PF_FpLong area_frac = r_loc / std::max<PF_FpLong>(radius, 1e-6);
		const PF_FpLong alpha = (potential_sum > kEps)
			? Clamp01(cover_sum / std::max<PF_FpLong>(potential_sum * area_frac * area_frac, kEps))
			: 0.0;
		return { PercToLin(rgb), alpha, matte };
	}
}

inline PF_FpLong DecodeSlider(PF_ParamDef* param)
{
	return param->u.fs_d.value;
}

// Anamorphic ratio for a named Aspect Preset popup (1-based). Returns a negative
// value for "Custom" (1), meaning the Aspect Ratio slider stays in control.
inline PF_FpLong AspectPresetRatio(A_long idx)
{
	switch (idx) {
		case 2:  return 1.0;    // Spherical 1.0
		case 3:  return 2.0;    // Anamorphic 2x
		case 4:  return 1.5;    // Anamorphic 1.5x
		case 5:  return 1.33;   // Anamorphic 1.33x
		case 6:  return 2.39;   // Cinemascope 2.39
		case 7:  return 1.85;   // Widescreen 1.85
		case 8:  return 1.43;   // IMAX 1.43
		case 9:  return 0.91;   // NTSC 0.91
		case 10: return 1.09;   // PAL 1.09
		default: return -1.0;   // Custom -> use the slider
	}
}

inline PF_FpLong DecodeAngleRad(PF_ParamDef* param)
{
	return FIX_2_FLOAT(param->u.ad.value) * (kPi / 180.0);
}

// Decode a PF_Point parameter into a normalized [0..1] UV.
//
// The gotcha: `in_data->width`/`height` are the *full-resolution* layer size
// (per AE_Effect.h), while PF_Point values (td.x_value/y_value) are expressed
// in the *current render* coordinate system — i.e. already scaled by the UI
// downsample. Normalizing the raw fixed value by `in_data->width` therefore
// yields garbage at anything other than Full-res preview (Half gives 0.5×,
// Quarter gives 0.25× the expected UV) and the Set Focus picker lands on a
// completely different pixel than the one the user clicked. Fix: build the
// current-render width/height via the downsample ratio and normalize there.
inline Vec2 DecodeAutoFocusPoint(PF_ParamDef* param, const PF_InData* in_data)
{
	const PF_FpLong raw_x = FIX_2_FLOAT(param->u.td.x_value);
	const PF_FpLong raw_y = FIX_2_FLOAT(param->u.td.y_value);
	if (std::abs(raw_x) < kEps && std::abs(raw_y) < kEps) {
		return {0.5, 0.5};
	}
	const PF_FpLong dsx_n = static_cast<PF_FpLong>(std::max<A_long>(1, in_data->downsample_x.num));
	const PF_FpLong dsx_d = static_cast<PF_FpLong>(std::max<A_u_long>(1u, in_data->downsample_x.den));
	const PF_FpLong dsy_n = static_cast<PF_FpLong>(std::max<A_long>(1, in_data->downsample_y.num));
	const PF_FpLong dsy_d = static_cast<PF_FpLong>(std::max<A_u_long>(1u, in_data->downsample_y.den));
	const PF_FpLong cur_w = std::max<PF_FpLong>(1.0,
		static_cast<PF_FpLong>(in_data->width)  * dsx_n / dsx_d);
	const PF_FpLong cur_h = std::max<PF_FpLong>(1.0,
		static_cast<PF_FpLong>(in_data->height) * dsy_n / dsy_d);
	return {Clamp01(raw_x / cur_w), Clamp01(raw_y / cur_h)};
}

// ── Lens-character presets ─────────────────────────────────────────────────
//
// Each preset re-writes the subset of DOFSettings that defines the "look" of
// a real lens: blade geometry, softness, spherical / chromatic aberrations,
// optical vignetting, catadioptric ring, astigmatism, anamorphic ratio.
// Everything else (focal distance, aperture size, sample count, focus range,
// noise, matte box, depth mapping) stays under user control, so presets act
// like swappable lens bodies instead of fully locking the effect.
//
// Applied *after* the slider decoder; preset 0 ("Manual") is a no-op so the
// raw slider values are preserved verbatim.
// ── Lens preset system (v2.6, "bake to panel") ─────────────────────────────
// Instead of overriding the decoded settings at render time (invisible to the
// user), a preset now writes its values straight into the real param sliders
// (see ApplyPresetToParams, driven from PF_Cmd_USER_CHANGED_PARAM) and the
// popup snaps back to "Manual" -- so every value lands in the panel and stays
// fully editable. Values below are in PARAM units (slider 0..100, etc.), not
// the decoded 0..1 units the old ApplyLensPreset used.
struct LensPreset {
	A_long    shape;        // Iris Shape popup: 1 Circular, 2 Polygonal, 3 Notched, 4 Custom
	A_long    apmap;        // Aperture Map popup: 1 Off, else Map(N) = N+1
	PF_FpLong blades;       // 3..16
	PF_FpLong blade_curve;  // -100..100
	PF_FpLong notch_scale;  // 0..100
	PF_FpLong softness;     // 0..100
	PF_FpLong vignetting;   // -100..100
	PF_FpLong vig_scale;    // 0.1..4
	PF_FpLong astig;        // 0..100
	A_long    astig_sag;    // 0/1  (Astigmatism: Sagittal)
	PF_FpLong aspect;       // 0..4  (anamorphic ratio)
	PF_FpLong sa_amount;    // -100..100
	PF_FpLong sa_scale;     // 0..100
	PF_FpLong ca_rc;        // -100..100 Red/Cyan %
	PF_FpLong ca_gm;        // -100..100 Green/Magenta %
	PF_FpLong ca_by;        // -100..100 Blue/Yellow %
	A_long    cata_on;      // 0/1  (Mirror / Catadioptric)
	PF_FpLong cata_scale;   // 0..100
	A_long    tint_r, tint_g, tint_b; // 0..255 highlight tint
	PF_FpLong bloom;        // 0..100 (Bloom / Halation)
	// Highlight punch (v2.8.1): without these two the crisp edge-defined
	// bokeh discs of real vintage glass (fairy-light test shots) cannot be
	// reproduced by a preset -- the gather averages the specular energy away.
	PF_FpLong bokeh_gamma;  // 0..3   (Bokeh Gamma; panel default 0.8)
	PF_FpLong scatter;      // 0..100 (Highlight Scatter)
	PF_FpLong onion;        // 0..100 (procedural Onion Rings, v2.9)
	PF_FpLong field_curv;   // 0..100 (Field Curvature edge blur, v2.10)
	PF_FpLong field_sweet;  // 0..100 (Sweet Spot Size, v2.10)
};

// Index 0 = popup value 2 (first entry after "Manual"). Order MUST match
// StrID_Preset_Choices.
static const LensPreset g_lens_presets[] = {
	// ── v2.9.2: real-lens lineup. Every entry models a documented optical
	// character of the actual glass (similar/duplicate presets removed).
	//      shp apmap bld curv ntch soft  vig vigsc astig sag aspect  sa  sasc  carc  cagm caby cata cscl   R   G   B  blm    bg sct  onion

	// KMZ Helios 44-2 58/2 (biotar copy): edge-rimmed discs with machining
	// onion rings, strong cat's-eye toward frame edges, moderate swirl,
	// whisper of red/blue fringe. THE Soviet portrait classic.
	/*Helios 44-2 58mm*/ {1, 1, 8, 0, 0, 10, 75, 1.55, 45, 0, 1.00, 40, 50, 0, 9, 0, 0, 0, 255, 255, 255, 8, 1.10, 25, 18},
	// Cyclop 85/1.5 (night-vision block = Helios-40 with NO aperture
	// blades, permanently wide open): the wildest swirl in Soviet glass --
	// max tangential astigmatism + max cat's-eye, circular iris, soft glow.
	/*Cyclop 85mm f/1.5*/ {1, 1, 8, 0, 0, 15, 100, 1.90, 90, 0, 1.00, 25, 35, 0, 11, 0, 0, 0, 255, 255, 255, 12, 1.00, 15, 0, 25, 45},
	// Mir-1V 37/2.8 (Grand Prix Brussels 1958): warm enveloping vintage
	// softness, gentle swirl, mild fringing -- the "warm hug" wide angle.
	/*Mir-1V 37mm*/ {2, 1, 10, 80, 0, 70, 40, 1.35, 15, 0, 1.00, 50, 30, 0, 16, 0, 0, 0, 255, 238, 215, 30, 0.85, 5, 0, 20, 50},
	// Industar-61 L/Z 50/2.8: lanthanum glass, razor sharp, and the famous
	// 6-point STAR bokeh from its concave blade shape at f/5.6-f/8 --
	// modelled with 6 blades at strong negative curvature.
	/*Industar-61 L/Z*/ {2, 1, 6, -75, 0, 4, 20, 1.20, 0, 0, 1.00, 10, 40, 7, 0, -7, 0, 0, 255, 255, 255, 5, 1.05, 30, 0},
	// MTO-500 catadioptric 500/8: the mirror-lens donut. Hollow-centre
	// discs, zero optical vignetting (central obstruction dominates).
	/*MTO-500 Mirror*/ {1, 1, 6, 100, 0, 15, 0, 1.00, 0, 0, 1.00, 0, 50, 0, 0, 0, 1, 80, 255, 255, 255, 0, 1.00, 25, 0},
	// Meyer-Optik Görlitz Trioplan 100/2.8: the soap-bubble king. The
	// triplet's heavy overcorrected SA turns every highlight into a bright-
	// rimmed, hard-edged perfect circle; faint rings inside, crisp punch.
	/*Trioplan 100mm*/ {1, 1, 15, 100, 0, 5, 55, 1.35, 8, 0, 1.00, 90, 80, 0, 18, 0, 0, 0, 255, 255, 255, 15, 1.15, 35, 12},
	// Lomography Petzval 85 (modern reissue): razor centre, massive
	// "tunnel" swirl at the edges -- the strongest vignette+swirl combo
	// short of the Cyclop, with cleaner, more defined discs.
	/*Petzval 85 (Lomo)*/ {1, 1, 6, 100, 0, 12, 95, 1.80, 65, 0, 1.00, 35, 45, 0, 10, 0, 0, 0, 255, 255, 255, 10, 1.05, 20, 0, 30, 45},
	// Canon 50/0.95 "Dream Lens": the legendary glow -- dreamy softness,
	// strong (but sane: the slider is 0.02x per unit) magenta/green fringing
	// wide open, lifted highlights. v2.9.5: was 70/10 mode 1, which drove
	// the channel scale ~1.5x past the disc -- broken red wash.
	/*Canon 0.95 Dream*/ {1, 1, 10, 90, 0, 45, 55, 1.40, 18, 0, 1.00, 30, 30, 0, 28, 0, 0, 0, 255, 248, 240, 25, 0.95, 10, 0},
	// Lensbaby Velvet 56: deliberate UNDERcorrected SA (negative) = bright
	// soft-glow cores instead of rims -- the velvet halation portrait look.
	/*Lensbaby Velvet 56*/ {1, 1, 8, 0, 0, 40, 25, 1.25, 0, 0, 1.00, -70, 45, 8, 0, -8, 0, 0, 255, 250, 242, 40, 0.80, 0, 0},
	// Sigma 105/1.4 Art "Bokeh Master": the modern clinical reference --
	// perfectly clean crisp discs, near-zero aberrations, specular punch.
	// (Also the go-to for fairy-light / garland shots.)
	/*Sigma 105mm Art*/ {1, 1, 9, 100, 0, 6, 25, 1.20, 0, 0, 1.00, 12, 60, 3, 0, -3, 0, 0, 255, 255, 255, 8, 1.35, 50, 0},
	// LOMO 35-NAP-2-3 projection anamorphot on a 50mm prime: true 2x
	// squeeze, green/magenta fringe, warm-gold flare tint of the old
	// projection coatings.
	/*LOMO 35-NAP 2x*/ {2, 1, 6, 15, 0, 25, 60, 1.45, 12, 0, 2.00, 14, 38, 0, 24, 0, 0, 0, 255, 225, 170, 15, 0.95, 18, 0},
	// Laowa Nanomorph 1.5x (Amber): modern compact anamorphic -- gentle
	// 1.5x ovals, restrained G/M fringe, signature amber flare tint.
	/*Laowa Nanomorph 1.5*/ {2, 1, 9, 35, 0, 30, 38, 1.22, 10, 0, 1.50, 12, 42, 0, 11, 0, 0, 0, 255, 189, 97, 12, 0.90, 15, 0},
	// Texture look (not a lens): dusty, scratched uncoated-glass discs from
	// library Map 78 -- kept as the one non-lens "character" preset.
	/*Bokeh: Grungy Vint.*/ {1, 79, 6, 100, 0, 40, 55, 1.40, 8, 0, 1.00, 15, 40, 23, 0, -23, 0, 0, 255, 235, 210, 10, 0.90, 10, 0, 0, 45},
	// ── v2.10: sweet-spot / field-curvature rigs (from the user's psyche-
	// drelic reference shots: sharp centre, radial streaks at the edges
	// regardless of depth). field_curv drives the depth-independent edge
	// blur; Astigmatism past 100 (new 0..200 range) elongates it into
	// streaks -- sagittal=1 = radial zoom-burst, 0 = swirl.
	/*Lensbaby Sweet 50*/ {1, 1, 8, 0, 0, 20, 30, 1.25, 140, 1, 1.00, 15, 35, 10, 0, -10, 0, 0, 255, 255, 255, 0, 0.80, 0, 0, 60, 40},
	/*Lensbaby Burnside35*/ {1, 1, 8, 0, 0, 25, 85, 1.70, 130, 0, 1.00, 20, 35, 12, 0, -12, 0, 0, 255, 255, 255, 0, 0.80, 0, 0, 50, 45},
	/*Wide Macro Rig*/ {1, 1, 8, 0, 0, 35, 70, 1.60, 180, 1, 1.00, 35, 30, 0, 20, 0, 0, 0, 255, 255, 255, 0, 0.80, 0, 0, 85, 25},

	// ── v2.15: astigmatism showcase trio (cool/unusual, all astig-driven) ──
	//      shp apmap bld curv ntch soft  vig vigsc astig sag aspect  sa  sasc  carc cagm caby cata cscl   R    G    B  blm   bg  sct onion fcrv fswt
	// "Swirl-o-Tron 58" -- a Cyclop dialled past 11: near-max TANGENTIAL
	// astigmatism (sag=0) wraps the whole frame into a hypnotic vortex, huge
	// cat's-eye pulls the discs into commas toward the rim, machining onion
	// rings + warm tint give it that haunted-Soviet-glass soul. Pure character.
	/*Swirl-o-Tron 58*/  {1, 1, 8,  0, 0, 12, 95, 1.90, 165, 0, 1.00, 22, 45,  0, 12,  0, 0, 0, 255, 250, 236, 12, 1.05, 22, 32,  0, 45},
	// "Starburst Zoom 35" -- SAGITTAL astigmatism (sag=1) fires every highlight
	// into a radial streak shooting away from the optical centre, and a strong
	// field-curvature edge-blur turns the frame corners into a zoom-burst
	// explosion while the centre stays sharp. Undercorrected SA (negative) adds
	// a soft glowing core; faint blue/yellow fringe on the streak tips.
	/*Starburst Zoom 35*/{1, 1, 6, 70, 0, 18, 45, 1.30, 195, 1, 1.00, -28, 40, 0, 0, 16, 0, 0, 240, 246, 255, 22, 0.85, 14, 0, 72, 28},
	// "Anamorphic Comet 2x" -- 2x squeeze + heavy TANGENTIAL astigmatism shears
	// each oval bokeh into a leaning comet streak with a bright head and a
	// trailing tail, green/magenta projection fringe and a warm-gold flare tint.
	// The closest thing to a sci-fi cine look this plugin can throw.
	/*Anamorphic Comet*/ {2, 1, 6, 20, 0, 16, 60, 1.55, 125, 0, 2.00, 16, 40,  0, 24,  0, 0, 0, 255, 232, 198, 16, 0.95, 18,  0,  0, 45},

	// ── v3.1: gaps in the lineup ───────────────────────────────────────────
	// Each of these occupies an optical axis the list above did not reach --
	// checked against the existing entries rather than added for the count.
	//      shp apmap bld curv ntch soft  vig vigsc astig sag aspect   sa sasc  carc cagm caby cata cscl   R    G    B  blm   bg  sct onion fcrv fswt

	// Sony 135/2.8 [T4.5] STF -- the apodization lens. A radially graded ND
	// element inside the barrel makes the disc fade to nothing at its rim, so
	// there is no bokeh EDGE at all. Nothing above does this: the softest entry
	// was Mir-1V at 70, and it spends that softness alongside heavy swirl and
	// fringing. Here softness is the whole point and everything else is near
	// zero -- the smoothest defocus ever put in a lens.
	/*Sony 135 STF (Apod.)*/ {1, 1, 9, 100, 0, 92,  8, 1.10,   0, 0, 1.00,   5, 40,  0,  0,  0, 0,  0, 255, 255, 255, 0, 0.75,  0,  0,  0, 45},
	// Cooke S4/i -- "the Cooke Look". Rounded 8-blade cine prime, deliberately
	// left slightly UNDERcorrected so highlights bloom into their discs rather
	// than ringing, with a warm cast. The counterpoint to the Sigma Art above,
	// which is the same cleanliness taken the other way (overcorrected, neutral).
	/*Cooke S4/i (Cine)*/    {2, 1, 8,  85, 0, 28, 32, 1.25,   6, 0, 1.00, -18, 45,  4,  0, -4, 0,  0, 255, 246, 232, 0, 0.90,  8,  0,  0, 45},
	// CCTV 25/1.4 C-mount -- the cheap security lens adapted to mirrorless. Hard
	// hexagonal iris (no blade rounding), violent swirl and vignetting, fringing
	// nobody corrected. Distinct from the Cyclop, whose swirl comes from a
	// bladeless CIRCULAR aperture and stays soft: this one is all hard edges.
	/*CCTV 25mm f/1.4*/      {2, 1, 6, -10, 0,  8, 90, 1.85, 120, 0, 1.00,  45, 55, 18,  0,-18, 0,  0, 255, 252, 245, 0, 1.10, 20,  8, 25, 35},
	// Rodenstock Imagon -- the classic portrait soft-focus head. Extreme
	// UNDERcorrected spherical aberration wraps every highlight in a glowing
	// halo instead of a disc. Same axis as the Lensbaby Velvet above but taken
	// to the end of it (-92/72 against -70/45), which is a different look rather
	// than a stronger one: the disc stops reading as a disc.
	/*Rodenstock Imagon*/    {1, 1,12, 100, 0, 72, 15, 1.15,   0, 0, 1.00, -92, 72,  0,  6,  0, 0,  0, 255, 250, 244, 0, 0.78,  0,  0,  0, 45},
	// Reflex 1000mm -- the BAD mirror lens, as against the MTO-500 above. Same
	// central obstruction, but the cheap catadioptrics stack a coarse machining
	// texture and real astigmatism on top, so the donuts swirl and ring instead
	// of sitting clean. The reason mirror bokeh has the reputation it does.
	/*Reflex 1000 (Donut)*/  {1, 1, 6, 100, 0, 20,  0, 1.00,  60, 0, 1.00,  12, 50,  0,  0,  0, 1, 92, 255, 255, 255, 0, 1.15, 28, 42,  0, 45},
	// Angenieux 25-250 -- the vintage cine zoom. Its element count is what makes
	// it: every ground surface prints another ring, so the onion structure is far
	// heavier than the Helios's machining marks (55 against 18), over a moderate
	// swirl and the cool cast of the old coatings.
	/*Angenieux 25-250*/     {1, 1, 9,  60, 0, 22, 58, 1.45,  35, 0, 1.00,  28, 45,  6,  8,  0, 0,  0, 246, 250, 255, 0, 1.00, 18, 55,  0, 45},
	// Tilt-Shift Miniature -- not a lens but the toy-town rig, and the one use
	// of Field Curvature the entries above do not cover: they all pair it with
	// heavy astigmatism to get streaks. This is pure depth-independent edge
	// blur with a tight sweet spot and clean optics, which is what reads as
	// "miniature" rather than "broken lens".
	/*Tilt-Shift Miniature*/ {1, 1, 9, 100, 0, 15, 20, 1.15,   0, 0, 1.00,   8, 40,  0,  0,  0, 0,  0, 255, 255, 255, 0, 0.95, 12,  0, 95, 22},
};

// Writes a preset's values into the live param sliders (so they show up in the
// panel and remain editable) and snaps the Lens Preset popup back to Manual.
// Called from PF_Cmd_USER_CHANGED_PARAM. preset_idx is the 1-based popup value.
static void ApplyPresetToParams(A_long preset_idx, PF_ParamDef* params[])
{
	if (preset_idx < 2) return;
	const A_long n = static_cast<A_long>(sizeof(g_lens_presets) / sizeof(g_lens_presets[0]));
	if (preset_idx - 2 >= n) return;
	const LensPreset& p = g_lens_presets[preset_idx - 2];

	auto set_f = [&](A_long idx, PF_FpLong v) {
		params[idx]->u.fs_d.value = v; params[idx]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE; };
	auto set_p = [&](A_long idx, A_long v) {
		params[idx]->u.pd.value = v; params[idx]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE; };
	auto set_b = [&](A_long idx, A_long v) {
		params[idx]->u.bd.value = v; params[idx]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE; };

	set_p(ZLUXDOF_APERTURE_SHAPE, p.shape);
	set_p(ZLUXDOF_APERTURE_MAP, p.apmap);
	set_f(ZLUXDOF_APERTURE_BLADES, p.blades);
	set_f(ZLUXDOF_BLADE_CURVE, p.blade_curve);
	set_f(ZLUXDOF_NOTCH_SCALE, p.notch_scale);
	set_f(ZLUXDOF_SOFTNESS, p.softness);
	set_f(ZLUXDOF_OPTICAL_VIGNETTING, p.vignetting);
	set_f(ZLUXDOF_OPTICAL_VIGNETTING_SCALE, p.vig_scale);
	set_f(ZLUXDOF_ASTIGMATISM, p.astig);
	set_b(ZLUXDOF_SPHERICAL_ABERRATION_PLUS, p.astig_sag);
	set_p(ZLUXDOF_ASPECT_PRESET, 1);             // Custom -> the ratio slider drives it
	set_f(ZLUXDOF_ASPECT_RATIO, p.aspect);
	set_f(ZLUXDOF_SPHERICAL_ABERRATION, p.sa_amount);
	set_f(ZLUXDOF_SPHERICAL_ABERRATION_SCALE, p.sa_scale);
	set_f(ZLUXDOF_CA_RED_CYAN, p.ca_rc);
	set_f(ZLUXDOF_CA_GREEN_MAGENTA, p.ca_gm);
	set_f(ZLUXDOF_CA_BLUE_YELLOW, p.ca_by);
	set_b(ZLUXDOF_CATADIOPTRIC_LENS, p.cata_on);
	set_f(ZLUXDOF_CATADIOPTRIC_LENS_SCALE, p.cata_scale);
	set_f(ZLUXDOF_ONION_RINGS, p.onion);
	set_f(ZLUXDOF_FIELD_CURVATURE, p.field_curv);
	set_f(ZLUXDOF_FIELD_SWEET, (p.field_curv > 0.001) ? p.field_sweet : 45.0);
	// Presets are OPTICS-ONLY (v2.9.4): they no longer touch the Highlights
	// group (tint / bloom / bokeh gamma / scatter / thresholds). Writing
	// those alongside the iris params kept stomping the user's highlight
	// grade and behaved erratically -- highlight shaping is the user's
	// grade, the preset is the glass. The LensPreset tint/bloom/gamma/
	// scatter columns are retained in the table as reference values only.

	// The popup KEEPS showing the chosen lens (v2.9.5). It used to snap back
	// to Manual so the baked values would read as "yours to edit" -- but in
	// practice that read as "the preset didn't apply". The values are still
	// baked into the sliders and fully editable; the popup is simply a label
	// for the last lens applied. (Re-applying the same lens after tweaks:
	// switch to Manual and back.)
}

struct RenderRefcon {
	PF_EffectWorld* src_world;
	PF_EffectWorld* depth_world;
	PF_EffectWorld* aperture_tex_world;
	// Optional Iris Texture modulator -- multiplied into the aperture mask
	// on every shape mode (polygonal, circular, catadioptric). Distinct
	// from aperture_tex_world, which REPLACES the aperture shape when the
	// user picks "Custom (Layer)" mode. iris_mod_world lets the user add
	// dust / blade imperfections / vintage coating texture on top of any
	// polygonal iris, which is the feature the dead APTEX layer parameter
	// was originally meant to drive.
	PF_EffectWorld* iris_mod_world;
	const DOFSettings* settings;
	// The frame's aperture-map snapshot. Borrowed; RenderCore owns the
	// shared_ptr that keeps it alive for the whole render, which is what lets
	// the worker threads read it without locking.
	const zlux_apmap::ApMap* apmap;
	const PF_FpLong* raw_depth_cache;
	const float* depth_cache;
	const float* signed_coc_cache;
	// Per-pixel distance (in output px) to the nearest CoC discontinuity, built
	// once per frame by a chamfer distance transform. The gather's silhouette-
	// band test reads this instead of scatter-scanning the CoC cache 16x/pixel.
	const float* coc_disc_dist;
	const PF_FpLong* protect_mask_cache; // per-pixel foreground-protect weight
	const CoCTileData* coc_tiles_dilated;
	const VogelLUT* vogel_luts;
	A_long num_vogel_luts;
	const SourcePyramid* pyramid;
	// Optional half-resolution prefiltered far-blur buffer. When non-null,
	// the per-pixel Far gather is replaced with a bilinear tap into this
	// buffer, which is prepared once up-front across all CPU cores. Because
	// far blur is by definition smooth, the visible quality loss is nil and
	// the per-frame gather cost for Far is effectively quartered. Only used
	// when depth-aware DOF is active and the pyramid is built.
	const Color3* far_halfres;
	// Far area-coverage alpha matching far_halfres (Fast mode): lets the far
	// bleed-over composite tap a prefiltered weight instead of running its own
	// full-res gather -- the third gather was a large share of frame cost.
	const float* far_alpha_halfres;
	// Blurred PREMULTIPLIED matte of the far layer (Fast mode, only when the
	// source has transparency): the half-res companion to far_halfres so the
	// full-res pass feathers the far silhouette without a per-pixel gather.
	const float* far_matte_halfres;
	// Half-res near layer (Fast mode): the near field is smooth bokeh by
	// definition, so -- like the far layer -- it upsamples visually clean while
	// quartering the cost of the near gather, which dominates frames whose
	// foreground is strongly blurred.
	const Color3* near_halfres;
	const float* near_alpha_halfres;
	// Blurred premultiplied matte of the near layer (Fast mode, transparency).
	const float* near_matte_halfres;
	// Per-tile "provably sharp" early-out map (all modes). 1 = no gather, no
	// probe and no bleed can produce anything but the source pixel here, so
	// the per-pixel pass copies the source directly. Bit-identical by
	// construction (see the eligibility conditions where it is built).
	const unsigned char* tile_skip;
	// ── CUDA gather results ────────────────────────────────────────────────
	// Non-null when the GPU ran this frame. Each is one element per output
	// pixel: rgb + the pass weight, matching what GatherPass returns. The
	// composite below keeps every one of its own gating conditions and simply
	// reads these instead of calling GatherPass, so the GPU and CPU paths differ
	// only in where the gather arithmetic happened.
	const struct float4_gpu* gpu_far;
	const struct float4_gpu* gpu_near;
	const struct float4_gpu* gpu_bleed;
	const struct float4_gpu* gpu_matte;  // x far, y near, z bleed
	A_long halfres_w;
	A_long halfres_h;
	PF_FpLong inv_w;
	PF_FpLong inv_h;
	A_long out_w;
	A_long out_h;
	A_long tiles_x;
};

// Bilinear tap into the half-res far buffer using output-space normalized UV.
// Border pixels are clamped to the edge sample; undefined pixels (null buffer)
// should be handled by the caller before invoking this helper.
inline Color3 SampleFarHalfres(const Color3* buf, A_long hw, A_long hh,
                               PF_FpLong u, PF_FpLong v)
{
	const PF_FpLong px = u * static_cast<PF_FpLong>(hw) - 0.5;
	const PF_FpLong py = v * static_cast<PF_FpLong>(hh) - 0.5;
	const A_long x0 = ClampValue<A_long>(static_cast<A_long>(std::floor(px)), 0, hw - 1);
	const A_long y0 = ClampValue<A_long>(static_cast<A_long>(std::floor(py)), 0, hh - 1);
	const A_long x1 = std::min<A_long>(x0 + 1, hw - 1);
	const A_long y1 = std::min<A_long>(y0 + 1, hh - 1);
	const PF_FpLong fx = Clamp01(px - std::floor(px));
	const PF_FpLong fy = Clamp01(py - std::floor(py));
	const Color3& c00 = buf[static_cast<size_t>(y0) * hw + x0];
	const Color3& c10 = buf[static_cast<size_t>(y0) * hw + x1];
	const Color3& c01 = buf[static_cast<size_t>(y1) * hw + x0];
	const Color3& c11 = buf[static_cast<size_t>(y1) * hw + x1];
	const PF_FpLong w00 = (1.0 - fx) * (1.0 - fy);
	const PF_FpLong w10 = fx * (1.0 - fy);
	const PF_FpLong w01 = (1.0 - fx) * fy;
	const PF_FpLong w11 = fx * fy;
	return {
		c00.r * w00 + c10.r * w10 + c01.r * w01 + c11.r * w11,
		c00.g * w00 + c10.g * w10 + c01.g * w01 + c11.g * w11,
		c00.b * w00 + c10.b * w10 + c01.b * w01 + c11.b * w11
	};
}

// Scalar variant of SampleFarHalfres for the float weight/alpha planes.
inline PF_FpLong SampleHalfresF(const float* buf, A_long hw, A_long hh,
                                PF_FpLong u, PF_FpLong v)
{
	const PF_FpLong px = u * static_cast<PF_FpLong>(hw) - 0.5;
	const PF_FpLong py = v * static_cast<PF_FpLong>(hh) - 0.5;
	const A_long x0 = ClampValue<A_long>(static_cast<A_long>(std::floor(px)), 0, hw - 1);
	const A_long y0 = ClampValue<A_long>(static_cast<A_long>(std::floor(py)), 0, hh - 1);
	const A_long x1 = std::min<A_long>(x0 + 1, hw - 1);
	const A_long y1 = std::min<A_long>(y0 + 1, hh - 1);
	const PF_FpLong fx = Clamp01(px - std::floor(px));
	const PF_FpLong fy = Clamp01(py - std::floor(py));
	const PF_FpLong c00 = buf[static_cast<size_t>(y0) * hw + x0];
	const PF_FpLong c10 = buf[static_cast<size_t>(y0) * hw + x1];
	const PF_FpLong c01 = buf[static_cast<size_t>(y1) * hw + x0];
	const PF_FpLong c11 = buf[static_cast<size_t>(y1) * hw + x1];
	return (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy)
	     + (c01 * (1.0 - fx) + c11 * fx) * fy;
}

// ── Occlusion-sliver (depth-discontinuity) detection ────────────────────────
// On the few-pixel band where an anti-aliased / smoothed depth edge
// interpolates between a blurred foreground and a blurred background, the
// signed CoC sweeps through zero. Those pixels claim "in focus" even though
// no surface at that depth exists; left alone they resurrect a sharp ghost
// stripe between two blurred fields. A genuinely focused surface cannot show
// BOTH a clearly-near and a clearly-far CoC within a few pixels of itself, so
// probing the signed-CoC cache on both axes at ±3 / ±6 px cleanly separates
// the two cases (the double offset keeps detection alive when Depth Smoothing
// widens the transition band). Returns 0 for ordinary pixels, ramping to 1 on
// a confident sliver.
inline PF_FpLong DetectCocSliver(const float* coc, A_long w, A_long h,
                                 A_long x, A_long y, PF_FpLong center_coc,
                                 PF_FpLong* out_far_max = nullptr)
{
	if (std::abs(center_coc) >= 0.06) {
		if (out_far_max) *out_far_max = 0.0;
		return 0.0;
	}
	PF_FpLong c_min = center_coc;
	PF_FpLong c_max = center_coc;
	const A_long offs[2] = {3, 6};
	for (int k = 0; k < 2; ++k) {
		const A_long o = offs[k];
		const A_long xm = std::max<A_long>(0, x - o);
		const A_long xp = std::min<A_long>(w - 1, x + o);
		const A_long ym = std::max<A_long>(0, y - o);
		const A_long yp = std::min<A_long>(h - 1, y + o);
		const size_t row = static_cast<size_t>(y) * w;
		const PF_FpLong c1 = coc[row + xm];
		const PF_FpLong c2 = coc[row + xp];
		const PF_FpLong c3 = coc[static_cast<size_t>(ym) * w + x];
		const PF_FpLong c4 = coc[static_cast<size_t>(yp) * w + x];
		c_min = std::min(c_min, std::min(std::min(c1, c2), std::min(c3, c4)));
		c_max = std::max(c_max, std::max(std::max(c1, c2), std::max(c3, c4)));
	}
	// Hand the probed far reach back to the caller: it is the per-pixel,
	// spatially smooth radius for the background reconstruction (the old
	// 16px-tile max produced visible blocky jumps along silhouettes).
	if (out_far_max) *out_far_max = std::max<PF_FpLong>(0.0, c_max);
	const PF_FpLong fields = std::min(-c_min, c_max);
	// Soft roll-off on the centre CoC instead of a hard cutoff, so the
	// sliver treatment fades out smoothly instead of switching at a contour.
	return SmoothStep(0.012, 0.04, fields)
	     * (1.0 - SmoothStep(0.03, 0.06, std::abs(center_coc)));
}

// Largest clearly-far signed CoC within ±4 / ±8 px. The half-res Far pre-pass
// uses it to decide whether a near/focused pixel borders the far field and
// therefore needs a background *reconstruction* stored in the buffer (instead
// of the sharp source colour, which would otherwise smear into the
// neighbouring blurred background through bilinear sampling).
inline PF_FpLong ProbeNeighborFarCoc(const float* coc, A_long w, A_long h, A_long x, A_long y)
{
	PF_FpLong c_max = 0.0;
	const A_long offs[2] = {4, 8};
	for (int k = 0; k < 2; ++k) {
		const A_long o = offs[k];
		const A_long xm = std::max<A_long>(0, x - o);
		const A_long xp = std::min<A_long>(w - 1, x + o);
		const A_long ym = std::max<A_long>(0, y - o);
		const A_long yp = std::min<A_long>(h - 1, y + o);
		const size_t row = static_cast<size_t>(y) * w;
		c_max = std::max(c_max, static_cast<PF_FpLong>(coc[row + xm]));
		c_max = std::max(c_max, static_cast<PF_FpLong>(coc[row + xp]));
		c_max = std::max(c_max, static_cast<PF_FpLong>(coc[static_cast<size_t>(ym) * w + x]));
		c_max = std::max(c_max, static_cast<PF_FpLong>(coc[static_cast<size_t>(yp) * w + x]));
	}
	return c_max;
}

// Wide-ladder variant for the far bleed-over composite: the background's
// wash over a focused silhouette extends as far as the background's own CoC
// reach, so far content must be detectable from tens of pixels away.
//
// Reach-gated (perf): a probe hit only counts if the far content's own blur
// disc can physically reach this pixel -- CoC c at distance o contributes
// nothing unless c * px_per_coc covers o (the gather's depth gate would
// reject every tap anyway, returning bleed = 0 after a full wasted gather).
// The 1.5x + 3px margin over-covers the SmoothStep gate's softness tail, so
// every pixel that could produce a non-zero bleed still runs the gather and
// the output is identical; only the guaranteed-zero gathers are skipped.
// px_per_coc = min_frame_dim * 0.15 (the engine's CoC -> pixel scale).
inline PF_FpLong ProbeFarReachWide(const float* coc, A_long w, A_long h,
                                   A_long x, A_long y, PF_FpLong px_per_coc)
{
	PF_FpLong c_max = 0.0;
	const A_long offs[5] = {3, 6, 12, 24, 48};
	for (int k = 0; k < 5; ++k) {
		const A_long o = offs[k];
		const A_long xm = std::max<A_long>(0, x - o);
		const A_long xp = std::min<A_long>(w - 1, x + o);
		const A_long ym = std::max<A_long>(0, y - o);
		const A_long yp = std::min<A_long>(h - 1, y + o);
		const size_t row = static_cast<size_t>(y) * w;
		PF_FpLong c = static_cast<PF_FpLong>(coc[row + xm]);
		c = std::max(c, static_cast<PF_FpLong>(coc[row + xp]));
		c = std::max(c, static_cast<PF_FpLong>(coc[static_cast<size_t>(ym) * w + x]));
		c = std::max(c, static_cast<PF_FpLong>(coc[static_cast<size_t>(yp) * w + x]));
		if (c > c_max && c * px_per_coc * 1.5 + 3.0 >= static_cast<PF_FpLong>(o)) {
			c_max = c;
		}
	}
	return c_max;
}


#ifdef ZLUX_CUDA
// Folds an aperture layer to a single-channel luma image for the device.
// Luma is a linear combination of RGB and bilinear filtering is linear, so
// luma-then-filter equals filter-then-luma exactly -- doing it here costs one
// pass over a small layer and saves three quarters of the per-tap bandwidth.
template <typename PIX>
static void ApertureLayerToLumaTyped(const PF_EffectWorld* w, std::vector<float>& out)
{
	out.resize(static_cast<size_t>(w->width) * w->height);
	ParallelRows(w->height, 32, [&](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; ++y) {
			for (A_long x = 0; x < w->width; ++x) {
				const Color3 c = ColorFromPix(*PixelPtr<PIX>(w, x, y));
				out[static_cast<size_t>(y) * w->width + x] = static_cast<float>(Luma(c));
			}
		}
	});
}

static bool ApertureLayerToLuma(const PF_EffectWorld* w, std::vector<float>& out)
{
	if (!w || !w->data || w->width <= 0 || w->height <= 0) return false;
	if (PF_WORLD_IS_DEEP(const_cast<PF_EffectWorld*>(w)))      ApertureLayerToLumaTyped<PF_Pixel16>(w, out);
	else if (WorldIsFloat(w))                                  ApertureLayerToLumaTyped<PF_PixelFloat>(w, out);
	else                                                       ApertureLayerToLumaTyped<PF_Pixel8>(w, out);
	return true;
}
#endif

// ── GPU gather bridge ───────────────────────────────────────────────────────
//
// Materialises the per-pixel gather radii, flattens the pyramid / Vogel ladder
// into the plain-C shapes the kernel expects, and runs one launch covering all
// three gathers. The radius expressions here are copied verbatim from
// RenderPixelImpl so the GPU computes exactly what the CPU would have; the
// composite keeps its own gating and just reads the result.
struct float4_gpu { float x, y, z, w; };

namespace zlux_gpu {

// ── Which path actually ran, for the panel badge ────────────────────────────
// Whether the gather ran on the GPU has been invisible to the user, and that
// ambiguity has already caused a "it feels faster" report while the GPU path
// was in fact disabled. RenderCore records what it really did; the custom-UI
// banner reads it. Deliberately reports the OBSERVED path, not the configured
// one, so a silent CPU fallback (unsupported feature, unhealthy device) is
// visible rather than merely slow.
enum class Path : int { Unknown = 0, Cpu = 1, Gpu = 2 };

inline std::atomic<int>& LastPath()
{
	static std::atomic<int> p{static_cast<int>(Path::Unknown)};
	return p;
}

// Last gather-stage wall time in milliseconds (kernel + transfers, or the CPU
// gather). Written by RenderCore, read by the badge.
inline std::atomic<float>& LastGatherMs()
{
	static std::atomic<float> ms{0.0f};
	return ms;
}

inline void RecordPath(Path p, float ms)
{
	LastPath().store(static_cast<int>(p), std::memory_order_relaxed);
	LastGatherMs().store(ms, std::memory_order_relaxed);
}


#ifdef ZLUX_CUDA
// ── GPU path gate ───────────────────────────────────────────────────────────
//
// ── CUDA runtime availability ───────────────────────────────────────────────
//
// cudart is DELAY-LOADED (see /DELAYLOAD in zluxDOF.vcxproj), so no CUDA symbol
// is resolved until the first call. That matters because a hard import means the
// Windows loader refuses to map the plug-in at all when the DLL is absent, and
// After Effects then does not list the effect -- a user with no CUDA toolkit
// would lose zluxDOF entirely instead of falling back to the CPU renderer.
//
// So: try to load the runtime ourselves, first from the folder the .aex lives in
// (which is NOT on the DLL search path, but is where the DLL ships next to the
// plug-in), then from the default search order. If neither works, no CUDA call
// is ever made and the CPU path runs exactly as it always did.
inline bool CudaRuntimePresent()
{
	static const bool ok = []() -> bool {
		// Already mapped? A host that links cudart normally (the dof_png harness)
		// or an earlier probe has it in the process already, and asking the
		// loader to find it again BY NAME fails whenever the CUDA bin directory
		// is not on PATH -- which is exactly what silently disabled the GPU path
		// here after delay-loading was introduced.
		if (::GetModuleHandleW(L"cudart64_12.dll")) return true;

		// Next to the .aex: where the runtime ships, and NOT on the DLL search
		// path, so it needs an explicit full path.
		wchar_t dir[MAX_PATH] = {0};
		if (::GetModuleFileNameW(zlux_banner::GetPluginModule(), dir, MAX_PATH)) {
			if (wchar_t* slash = wcsrchr(dir, L'\\')) {
				*(slash + 1) = 0;
				std::wstring beside = std::wstring(dir) + L"cudart64_12.dll";
				if (::LoadLibraryExW(beside.c_str(), nullptr,
				                     LOAD_WITH_ALTERED_SEARCH_PATH)) return true;
			}
		}
		return ::LoadLibraryW(L"cudart64_12.dll") != nullptr;
	}();
	return ok;
}

// ON by default. The AE crash that prompted a temporary opt-in default was
// traced to the version-field overflow (code 2.12 vs PiPL 3.12) plus a duplicate
// .aex registered from two plug-in folders -- not to CUDA. Both are fixed, and
// the failure latch below covers a genuinely unhealthy device.
//
//   ZLUX_NOGPU=1   force the CPU gather (also how the harness renders the
//                  reference image the GPU output is diffed against)
//
// Built against CUDA 12.9, whose cudart.lib is a genuine import library for
// cudart64_12.dll. CUDA 13 must NOT be used: both of its libs route through a
// hybrid runtime loader that access-violates on the first cudaGetDeviceCount
// inside After Effects (minidump: read of 0x0 in MSVCP140, stack RunGather ->
// Ctx() -> initializeLoader -> initializeHybridRuntimeLoaderOnce).
//
// Note the short-circuit order: ZLUX_NOGPU is checked before
// zluxGpuAvailable(), so forcing the CPU path also means the CUDA runtime is
// never initialised at all.
inline bool Enabled()
{
	// Order matters twice over: ZLUX_NOGPU short-circuits before anything CUDA
	// happens, and CudaRuntimePresent() must succeed before zluxGpuAvailable()
	// touches a delay-loaded import.
	static const bool v = (std::getenv("ZLUX_NOGPU") == nullptr) &&
	                      CudaRuntimePresent() &&
	                      (zluxGpuAvailable() != 0);
	return v;
}

// Set when any CUDA call fails. The GPU path is then abandoned for the rest of
// the session rather than retried per frame: a device that failed once will
// usually keep failing, and retrying turns one bad frame into a stalled render.
inline std::atomic<bool>& Disabled()
{
	static std::atomic<bool> d{false};
	return d;
}

// One context for the process. Creating it costs ~70-105 ms, far too much to
// pay per frame of a sequence, and it owns ~150 MB of device buffers -- so a
// per-thread context is not an option either.
//
// That single shared context is the catch: zluxDOF sets
// PF_OutFlag2_SUPPORTS_THREADED_RENDERING, so After Effects renders several
// FRAMES concurrently on different threads, and they would all scribble over
// the same device buffers. The GPU work is ~37 ms per frame, so serialising it
// costs little: the depth preprocessing and the composite (the other ~70 ms)
// still run fully in parallel across MFR threads, and the GPU is far from
// saturated at ~27 gathers/second.
// Deliberately NOT `static ZluxGpuContext* c = zluxGpuCreate();`. A function-
// local static runs its initialiser exactly once for the life of the process,
// so once Shutdown() nulled the slot the context could never come back: Ctx()
// returned null forever, the caller latched Disabled(), and every later render
// fell to the CPU gather. After Effects does pair GlobalSetup/GlobalSetdown more
// than once per session (reloading effects, purging), which made that reachable.
inline ZluxGpuContext*& CtxSlot()
{
	static ZluxGpuContext* c = nullptr;
	return c;
}

// Creates the context on first use. MUST be called with DeviceMutex held --
// zluxGpuCreate is not reentrant and the slot is shared across MFR threads.
inline ZluxGpuContext* CtxLocked()
{
	ZluxGpuContext*& c = CtxSlot();
	if (!c) c = zluxGpuCreate();
	return c;
}

// Non-creating peek, for callers that only want to touch a context that already
// exists (the result-set release guard). Never resurrects the device.
inline ZluxGpuContext* CtxIfAlive() { return CtxSlot(); }

// ── Crash breadcrumbs ───────────────────────────────────────────────────────
// The GPU path crashed inside After Effects while running clean in the
// standalone harness, and AE's own error reporter swallows the fault before
// Windows records it. This writes one line per stage, flushed and closed
// immediately, so the LAST line in the file names the stage that died. Enabled
// only when ZLUX_TRACE is set, so a normal render pays nothing.
inline const char* TracePath()
{
	static const char* p = std::getenv("ZLUX_TRACE");
	return p;
}

inline void Trace(const char* stage, long long a = -1, long long b = -1)
{
	const char* path = TracePath();
	if (!path) return;
	static std::mutex m;
	std::lock_guard<std::mutex> lk(m);
	if (FILE* f = std::fopen(path, "a")) {
		std::fprintf(f, "[tid %5lu] %-22s %lld %lld\n",
		             static_cast<unsigned long>(GetCurrentThreadId()), stage, a, b);
		std::fclose(f);   // close every time: a crash must not lose the tail
	}
}

// Serialises device access across MFR render threads. Held for the whole
// upload -> launch -> readback sequence, because the context's buffers are
// single-instance.
inline std::mutex& DeviceMutex()
{
	static std::mutex m;
	return m;
}

// Releases the device context. Safe to call when nothing was ever created, and
// safe to call more than once: the next render recreates the context through
// CtxLocked(). Clearing the failure latch is part of that -- a teardown is not
// evidence of a bad device, and leaving it set would make the reload permanent.
inline void Shutdown()
{
	std::lock_guard<std::mutex> lk(DeviceMutex());
	if (ZluxGpuContext* c = CtxSlot()) {
		zluxGpuDestroy(c);
		CtxSlot() = nullptr;
	}
	Disabled().store(false, std::memory_order_relaxed);
}

inline bool RunGather(const ZluxGatherParams& gp,
                      const DOFSettings& s,
                      const PF_EffectWorld* aperture_tex_world,
                      const PF_EffectWorld* iris_mod_world,
                      const SourcePyramid& pyramid,
                      const std::vector<VogelLUT>& luts,
                      const std::vector<float>& signed_coc,
                      const std::vector<float>& disc_dist,
                      const std::vector<float>& depth_cache,
                      const std::vector<CoCTileData>& tiles,
                      A_long tiles_x, A_long out_w, A_long out_h,
                      PF_FpLong inv_w, PF_FpLong inv_h,
                      const float4_gpu*& out_far,
                      const float4_gpu*& out_near,
                      const float4_gpu*& out_bleed,
                      const float4_gpu*& out_matte,
                      int& out_slot)
{
#ifdef ZLUX_PROFILE
	auto t_enter = std::chrono::steady_clock::now();
#endif
	// First call creates the CUDA context, which costs ~75 ms. That is a
	// once-per-process cost (AE pays it on the first rendered frame of a
	// session), NOT a per-frame one -- the profile below breaks it out so a
	// single-frame harness run is not mistaken for steady-state cost.
	Trace("enter", out_w, out_h);
	if (Disabled().load(std::memory_order_relaxed)) return false;

	// A genuine DEVICE failure below latches the GPU path off for the session.
	// Wrapped in a helper so every early return goes through it -- a silent
	// `return false` would retry the same failing device on the next frame.
	//
	// A TRANSIENT failure must not latch. The only one is ZLUX_GPU_BUSY: every
	// pooled result set is held by a concurrently compositing frame, which is
	// ordinary MFR contention on a many-core machine and clears by itself. It
	// used to latch, so one busy moment silently dropped the rest of the session
	// to the CPU gather (~10x slower) with nothing but the panel badge to say so.
	// Disarm via latch.excuse() for those.
	struct Latch {
		bool tripped = true;
		void excuse() { tripped = false; }
		~Latch() { if (tripped) Disabled().store(true, std::memory_order_relaxed); }
	} latch;
	// Everything from here to the readback touches the shared device buffers.
	std::lock_guard<std::mutex> device_lock(DeviceMutex());
	Trace("lock acquired");

	// Under the lock: CtxLocked() may create the context, and two MFR threads
	// arriving together must not both run zluxGpuCreate.
	ZluxGpuContext* ctx = CtxLocked();
	if (!ctx) { Trace("FAIL ctx"); return false; }

	const size_t n = static_cast<size_t>(out_w) * static_cast<size_t>(out_h);
#ifdef ZLUX_PROFILE
	auto t0 = t_enter;
	auto lap = [&](const char* what) {  // NOLINT
		auto t1 = std::chrono::steady_clock::now();
		std::fprintf(stderr, "  [prof]   gpu:%-16s %7.1f ms\n", what,
			std::chrono::duration<double, std::milli>(t1 - t0).count());
		t0 = t1;
	};
#else
	auto lap = [](const char*) {};
#endif

	lap("ctx init (1x)");

	// ── Radii ───────────────────────────────────────────────────────────────
	// Only the centre-depth field still comes from the host: it is produced by
	// the CPU depth chain, which has not moved to the device yet. The three
	// radii are computed on the GPU right after the upload -- see below.
	// depth_cache is already std::vector<float>, so it uploads directly -- the
	// staging copy this used to make was pure memcpy for no reason.
	// Only the per-tile near reach needs packing, and that is a few hundred
	// values, not a frame.
	std::vector<float> tile_min(tiles.size());
	for (size_t i = 0; i < tiles.size(); ++i)
		tile_min[i] = static_cast<float>(tiles[i].min_coc);
	const PF_FpLong px_per_coc = 0.15 / std::max(inv_w, inv_h);
	const PF_FpLong uniform_base = std::max<PF_FpLong>(0.0, s.blur_strength);

	lap("radii");

	// ── Pyramid + LUT flattening ────────────────────────────────────────────
	const float* lvl_data[8]; int lvl_w[8], lvl_h[8];
	const int nlev = std::min<int>(pyramid.num_levels, 8);
	for (int L = 0; L < nlev; ++L) {
		lvl_data[L] = pyramid.levels[static_cast<size_t>(L)].data.data();
		lvl_w[L] = pyramid.levels[static_cast<size_t>(L)].w;
		lvl_h[L] = pyramid.levels[static_cast<size_t>(L)].h;
	}
	Trace("upload pyramid", nlev, lvl_w[0]);
	if (zluxGpuUploadPyramid(ctx, lvl_data, lvl_w, lvl_h, nlev) != 0) { Trace("FAIL pyramid"); return false; }
	lap("upload pyramid");

	std::vector<ZluxVogelHot>     hot;
	std::vector<ZluxVogelCold>    cold;
	std::vector<ZluxVogelLutDesc> descs;
	descs.reserve(luts.size());
	for (const VogelLUT& L : luts) {
		ZluxVogelLutDesc d{ static_cast<int>(L.count), static_cast<int>(hot.size()) };
		descs.push_back(d);
		for (A_long i = 0; i < L.count; ++i) {
			const VogelSample& vs = L.samples[i];
			ZluxVogelHot h{ static_cast<float>(vs.kx), static_cast<float>(vs.ky),
			                static_cast<float>(vs.fr), static_cast<float>(vs.static_mask) };
			ZluxVogelCold c{ static_cast<float>(vs.cos_a), static_cast<float>(vs.sin_a),
			                 static_cast<float>(vs.norm_x), static_cast<float>(vs.norm_y),
			                 static_cast<float>(vs.soft_edge),
			                 static_cast<float>(vs.sa_pos), static_cast<float>(vs.sa_neg), 0.0f };
			hot.push_back(h);
			cold.push_back(c);
		}
	}
	// Aperture layers + the per-bokeh rotation table. Uploaded every frame: the
	// layers are small and may be animated, and re-uploading is far cheaper than
	// tracking their validity.
	//
	// The results are CHECKED. They used to be discarded, so a failed upload left
	// the previous frame's iris texture bound and the gather quietly rendered the
	// wrong aperture -- the one outcome the GPU path is supposed to make
	// impossible, since its whole contract is that enabling a GPU never changes
	// output. A failure here is a device error, so it latches like any other.
	{
		float rot[256];
		for (int i = 0; i < 128; ++i) {
			rot[i * 2 + 0] = static_cast<float>(kBokehRotLUT[i].first);
			rot[i * 2 + 1] = static_cast<float>(kBokehRotLUT[i].second);
		}
		std::vector<float> lum;
		int arc;
		if (aperture_tex_world && ApertureLayerToLuma(aperture_tex_world, lum)) {
			arc = zluxGpuUploadApertureTex(ctx, 0, lum.data(),
			                               aperture_tex_world->width, aperture_tex_world->height, rot);
		} else {
			arc = zluxGpuUploadApertureTex(ctx, 0, nullptr, 0, 0, rot);
		}
		if (arc != ZLUX_GPU_OK) { Trace("FAIL aptex0"); return false; }

		if (iris_mod_world && ApertureLayerToLuma(iris_mod_world, lum)) {
			if (zluxGpuUploadApertureTex(ctx, 1, lum.data(),
			                             iris_mod_world->width, iris_mod_world->height,
			                             nullptr) != ZLUX_GPU_OK) {
				Trace("FAIL aptex1"); return false;
			}
		}
	}

	float bg[257];
	for (int i = 0; i < 257; ++i) bg[i] = static_cast<float>(s.bokeh_gamma_lut[i]);
	if (zluxGpuUploadLuts(ctx, hot.data(), cold.data(), static_cast<int>(hot.size()),
	                      descs.data(), static_cast<int>(descs.size()), bg) != 0) { Trace("FAIL luts"); return false; }
	lap("upload luts");

	ZluxGatherFields F{};
	F.signed_coc    = signed_coc.data();
	// Null: the device builds this itself right after the upload (see below), so
	// there is nothing to transfer.
	F.coc_disc_dist = disc_dist.empty() ? nullptr : disc_dist.data();
	F.far_radius    = nullptr;   // built on the device
	F.near_radius   = nullptr;
	F.bleed_radius  = nullptr;
	F.center_depth  = depth_cache.data();
	Trace("upload fields");
	{
		// Can report BUSY when the device is simply short of free VRAM this
		// instant (see the cudaMemGetInfo guard in zluxGpuUploadFields).
		const int frc = zluxGpuUploadFields(ctx, &gp, &F);
		if (frc != ZLUX_GPU_OK) {
			if (frc == ZLUX_GPU_BUSY) { Trace("BUSY fields"); latch.excuse(); }
			else                      { Trace("FAIL fields"); }
			return false;
		}
	}
	lap("upload fields");

	Trace("radii (gpu)");
	if (zluxGpuBuildRadii(ctx, &gp, tile_min.data(), tiles_x,
	                      static_cast<int>(tiles.size()) / std::max(1, tiles_x),
	                      static_cast<int>(kCocTileSize), static_cast<float>(px_per_coc),
	                      static_cast<float>(uniform_base),
	                      static_cast<float>(s.field_curvature),
	                      static_cast<float>(s.field_sweet)) != 0) {
		Trace("FAIL radii"); return false;
	}
	lap("radii gpu");

	// Only when the CPU stage was skipped; otherwise the uploaded field stands.
	if (disc_dist.empty()) {
		Trace("disc dist (gpu)");
		if (zluxGpuBuildDiscDist(ctx, &gp) != 0) { Trace("FAIL disc dist"); return false; }
		lap("disc dist gpu");
	}

	// No host allocation here: zluxGpuGather fills its own page-locked staging
	// buffers and returns pointers into them. Sizing std::vectors for this cost
	// ~95 MB of zero-fill per frame on top of the transfer itself.
	ZluxGatherOutputs O{};

	float kernel_ms = 0.0f;
	Trace("launch");
	const int grc = zluxGpuGather(ctx, &gp, &O, &kernel_ms);
	if (grc != ZLUX_GPU_OK) {
		// BUSY means the device is healthy and simply out of free result sets;
		// this frame goes to the CPU and the next one retries the GPU.
		if (grc == ZLUX_GPU_BUSY) { Trace("BUSY gather"); latch.excuse(); }
		else                      { Trace("FAIL gather"); }
		return false;
	}
	Trace("gather ok");
	latch.excuse();   // full success -- keep the GPU path armed

	// The pool hands this frame its own page-locked set, so the compositor can
	// read it after the device lock drops without the next frame overwriting it
	// -- and without the ~75 MB host-to-host copy the previous fix needed.
	out_far   = reinterpret_cast<const float4_gpu*>(O.far_rgba);
	out_near  = reinterpret_cast<const float4_gpu*>(O.near_rgba);
	out_bleed = reinterpret_cast<const float4_gpu*>(O.bleed_rgba);
	out_matte = reinterpret_cast<const float4_gpu*>(O.mattes);
	out_slot  = O.slot;
	Trace("copied out");
	return true;
}
#else
inline bool Enabled() { return false; }
#endif // ZLUX_CUDA

} // namespace zlux_gpu

template <typename PIX>
static PF_Err RenderPixelImpl(void* refcon, A_long x, A_long y, PIX* inP, PIX* outP)
{
	RenderRefcon* r = static_cast<RenderRefcon*>(refcon);
#ifdef ZLUX_PROFILE
	static const bool kSkipFar  = std::getenv("ZLUX_SKIP_FAR")  != nullptr;
	static const bool kSkipNear = std::getenv("ZLUX_SKIP_NEAR") != nullptr;
	static const bool kSkipBleed = std::getenv("ZLUX_SKIP_BLEED") != nullptr;
#else
	constexpr bool kSkipFar = false, kSkipNear = false, kSkipBleed = false;
#endif
	const DOFSettings& local = *r->settings;
	const PF_FpLong u = (static_cast<PF_FpLong>(x) + 0.5) * r->inv_w;
	const PF_FpLong v = (static_cast<PF_FpLong>(y) + 0.5) * r->inv_h;
	const Color3 original = ColorFromPix<PIX>(*inP);
	const PF_FpLong alpha = static_cast<PF_FpLong>(inP->alpha) * PixTraits<PIX>::inv_max;
	Color3 final_color = original;
	// Output alpha. For opaque sources (no transparency in the layer) this
	// stays the unchanged original alpha and the whole matte-feathering path
	// below is skipped -- bit-identical to the pre-alpha behaviour. When the
	// layer has transparency the blur branches overwrite it with the matte
	// blurred in lockstep with the colour so the silhouette feathers.
	const bool has_alpha = r->pyramid && r->pyramid->has_alpha;
	PF_FpLong out_alpha = alpha;

	// 32bpc is scene-linear already (AE's convention); 8/16bpc is in project
	// gamma. The whole bokeh accumulation is done in linear light so gamma
	// is only (de)applied at the edges of the pipeline for non-float paths.
	constexpr bool kIsFloat = std::is_same_v<PIX, PF_PixelFloat>;

	const bool display_only = (local.display_mode >= 2);

	if (!display_only) {
		const size_t pidx = static_cast<size_t>(y * r->out_w + x);
		const PF_FpLong depth = r->depth_cache[pidx];
		const PF_FpLong signed_coc = r->signed_coc_cache[pidx];
		const A_long tx = x / kCocTileSize;
		const A_long ty = y / kCocTileSize;
		const CoCTileData& tile = r->coc_tiles_dilated[static_cast<size_t>(ty * r->tiles_x + tx)];

		// ── Layered DoF compositing (GPU Gems 3 §28.5/§28.7) ───────────────
		//
		// The old pipeline cross-faded between the sharp source and a
		// possibly-blurred "far" result driven by a microscopic (0..0.012)
		// smoothstep on the signed CoC. Any tiny CoC tipped the pixel to
		// fully blurred even inside the focus zone, which is why focused
		// objects visibly lost detail.
		//
		// New pipeline draws three layers in linear light:
		//   1. Far layer  : gathered from clearly-far samples only.
		//                   Acts as the "blurred background" base.
		//   2. Sharp layer: the untouched source pixel, composited *on top
		//                   of* the far layer inside the focus zone.
		//                   Width of the fully-sharp region is driven by
		//                   the focus_range slider, giving the user a real
		//                   in-focus band.
		//   3. Near layer : gathered from clearly-near samples only, with
		//                   its own coverage alpha. Painted last so that
		//                   foreground bokeh floats over everything,
		//                   focused subject included.
		//
		// Rejecting focused samples from the blur layers (tight 0.003 gate
		// in GatherPass) means the sharp overlay *is* the only source of
		// in-focus detail, exactly as §28.7 suggests: "render everything
		// that should not be blurred on top of the already-blurred image".
		// All math stays in linear light; gamma is only (un)applied for
		// 8/16bpc at the pipeline edges.
		const Color3 src_lin = kIsFloat ? original : GammaToLinear(original);

		Color3 final_lin;

		if (local.no_depth) {
			// Uniform blur mode: no depth information, the whole frame
			// participates in a single far layer and nothing is kept sharp.
			// Field curvature still works here (a Lensbaby-style sweet-spot
			// blur needs no depth map at all).
			PF_FpLong uniform_r = std::max<PF_FpLong>(0.0, local.blur_strength);
			if (local.field_curvature > 0.001) {
				uniform_r += FieldCurvatureCoc(u, v, local.field_curvature, local.field_sweet);
			}
			Color3 blurred = src_lin;
			if (uniform_r > 0.001) {
				if (r->gpu_far) {
					const float4_gpu g = r->gpu_far[pidx];
					blurred = { g.x, g.y, g.z };
					if (has_alpha && r->gpu_matte) out_alpha = r->gpu_matte[pidx].x;
				} else {
				PassOutput f = GatherPass<DofPass::Far>(
					*r->pyramid, r->aperture_tex_world, r->iris_mod_world,
					u, v, uniform_r, local, depth, uniform_r,
					r->inv_w, r->inv_h,
					r->vogel_luts, r->num_vogel_luts,
					r->signed_coc_cache, r->out_w, r->out_h, r->coc_disc_dist);
				blurred = f.rgb;
				// Whole frame is one blurred layer -> the matte is just its
				// feathered alpha. This is the path the transparent-viewmodel
				// case (no depth map) takes, so the silhouette now softens.
				if (has_alpha) out_alpha = f.matte;
				}
			}
			final_lin = blurred;
		} else if (r->tile_skip &&
		           r->tile_skip[static_cast<size_t>(ty) * r->tiles_x + tx]) {
			// Provably-sharp tile: every gather, probe and composite below
			// reduces exactly to the source pixel (the eligibility conditions
			// in RenderCore guarantee no near reach, no far CoC, no sliver and
			// no bleed-capable far content anywhere their probes can see), so
			// skip the whole machinery. This is the fast path for ordinary
			// "subtle DOF" comps where most of the frame stays in focus.
			final_lin = src_lin;
		} else {
			// Far gather (background blur). Only runs when this pixel is
			// itself far; otherwise there is no background to build.
			//
			// Background-inpaint base: behind a near-field pixel there is no real
			// background in the plate. The far base is just the sharp source;
			// far pixels overwrite it with the gather and focused pixels cover it
			// with the sharp overlay.
			Color3 far_rgb = src_lin;
			// Far layer's blurred matte. Default = original alpha (the far
			// base is the sharp source until a gather replaces it).
			PF_FpLong far_matte = alpha;
			const PF_FpLong center_far = std::max<PF_FpLong>(0.0, signed_coc);
			// Occlusion-sliver rescue: on the interpolated depth edge between
			// a blurred foreground and a blurred background the CoC crosses
			// zero, so these pixels read as "in focus", keep the sharp source
			// as their base layer, and show up as a ghost stripe along
			// silhouettes (the band the user flagged on the shoulder). Detect
			// the sliver from the local CoC field, give it a real background
			// reconstruction here, and suppress its sharp-retention mask in
			// the focus_mask computation below.
			PF_FpLong sliver_far = 0.0;
			const PF_FpLong sliver = DetectCocSliver(
				r->signed_coc_cache, r->out_w, r->out_h, x, y, signed_coc, &sliver_far);
			const PF_FpLong far_need = std::max(center_far, sliver * sliver_far);
			if (far_need > 0.001 && !kSkipFar) {
				if (r->far_halfres) {
					// Prefiltered far buffer hit: bilinear sample. The far
					// layer is smooth by construction, so the 2× upsample
					// is visually indistinguishable from a full-res gather
					// while costing roughly one-quarter the sample work.
					far_rgb = SampleFarHalfres(
						r->far_halfres, r->halfres_w, r->halfres_h, u, v);
					if (has_alpha && r->far_matte_halfres) {
						far_matte = SampleHalfresF(
							r->far_matte_halfres, r->halfres_w, r->halfres_h, u, v);
					}
				} else if (r->gpu_far) {
					const float4_gpu g = r->gpu_far[pidx];
					far_rgb = { g.x, g.y, g.z };
					if (has_alpha && r->gpu_matte) far_matte = r->gpu_matte[pidx].x;
				} else {
					PassOutput f = GatherPass<DofPass::Far>(
						*r->pyramid, r->aperture_tex_world, r->iris_mod_world,
						u, v, far_need, local, depth, signed_coc,
						r->inv_w, r->inv_h,
						r->vogel_luts, r->num_vogel_luts,
						r->signed_coc_cache, r->out_w, r->out_h, r->coc_disc_dist);
					far_rgb = f.rgb;
					if (has_alpha) far_matte = f.matte;
				}
			}

			// Near gather (foreground scatter emulated as inverted-gather).
			// Radius is the tile-dilated max near CoC so a nearby near pixel
			// can still splash into this one even if it is focused itself.
			Color3 near_rgb = {0.0, 0.0, 0.0};
			PF_FpLong near_alpha = 0.0;
			PF_FpLong near_matte = 0.0; // blurred matte of the near (fg) layer
			const PF_FpLong tile_near_reach = std::max<PF_FpLong>(0.0, -tile.min_coc);
			const PF_FpLong self_near = std::max<PF_FpLong>(0.0, -signed_coc);
			const PF_FpLong near_r = std::max(tile_near_reach, self_near);
			if (r->near_halfres) {
				// Fast mode: prefiltered half-res near layer (rgb + coverage
				// alpha), bilinearly upsampled -- same shortcut philosophy as
				// the far buffer. The near field is smooth bokeh, so the 2x
				// upsample costs ~nothing visually but quarters the gather
				// work that dominates strongly-near-blurred frames.
				near_alpha = SampleHalfresF(
					r->near_alpha_halfres, r->halfres_w, r->halfres_h, u, v);
				if (near_alpha > 0.002) {
					near_rgb = SampleFarHalfres(
						r->near_halfres, r->halfres_w, r->halfres_h, u, v);
					if (has_alpha && r->near_matte_halfres) {
						near_matte = SampleHalfresF(
							r->near_matte_halfres, r->halfres_w, r->halfres_h, u, v);
					}
				} else {
					near_alpha = 0.0;
				}
			} else if (near_r > 0.001 && !kSkipNear && r->gpu_near) {
				const float4_gpu g = r->gpu_near[pidx];
				near_rgb = { g.x, g.y, g.z };
				near_alpha = g.w;
				if (has_alpha && r->gpu_matte) near_matte = r->gpu_matte[pidx].y;
			} else if (near_r > 0.001 && !kSkipNear) {
				PassOutput n = GatherPass<DofPass::Near>(
					*r->pyramid, r->aperture_tex_world, r->iris_mod_world,
					u, v, near_r, local, depth, signed_coc,
					r->inv_w, r->inv_h,
					r->vogel_luts, r->num_vogel_luts,
					r->signed_coc_cache, r->out_w, r->out_h, r->coc_disc_dist);
				near_rgb = n.rgb;
				near_alpha = n.weight;
				if (has_alpha) near_matte = n.matte;
			}

			// Sharp-retention mask. |signed_coc| inside [0, focus_inner] is
			// fully sharp; the tail between focus_inner and focus_outer is
			// the soft DoF edge. focus_range (UI slider) widens both; the
			// inner half is the fully in-focus band and the outer edge is
			// where the pixel is considered completely out of focus.
			//
			// The ramp is kept narrow (outer - inner ≈ focus_range * 0.25)
			// so the linear blend between sharp src and the far gather
			// doesn't linger long enough to create the DOF PRO-style
			// "bright halo on sharp edges against a blurry bg" we see in
			// the user's reference image. Keeping the transition tight
			// also hands all soft-falloff duty to the gather itself,
			// which already ramps smoothly via its own depth-aware gate.
			const PF_FpLong focus_inner = std::max<PF_FpLong>(0.006, local.focus_range * 0.40);
			const PF_FpLong focus_outer = std::max<PF_FpLong>(focus_inner + 0.006,
			                                                  focus_inner + local.focus_range * 0.35);
			PF_FpLong focus_mask = 1.0 -
				SmoothStep(focus_inner, focus_outer, std::abs(signed_coc));
			// The occlusion sliver's "in focus" claim is bogus (no surface at
			// that depth exists) -- suppress sharp retention so the background
			// reconstruction + near splash cover the band instead.
			focus_mask *= (1.0 - sliver);

			// Foreground Protect: force sharp retention and deny incoming
			// near-splash overlay at pixels flagged as thin foreground
			// detail during the depth pass. Without this second half the
			// wire pixel is "sharp" inside the layer but then the near
			// composite on top paints blurred neighbours over it and the
			// protection silently fails. Empirically validated in the
			// debug harness (see debug/dof_debug.cpp).
			const PF_FpLong protect = r->protect_mask_cache[pidx];
			focus_mask = std::max(focus_mask, protect);
			const PF_FpLong near_alpha_eff = near_alpha * (1.0 - protect);

			// Composite order: far (bg) → overlay sharp focused →
			// far bleed-over → near (fg). The matte (layer_a) is composited
			// in lockstep with the colour using the identical blend factors
			// so the premultiplied pair stays consistent at every step; it is
			// only consumed when the layer carries transparency.
			Color3 layer = far_rgb;
			PF_FpLong layer_a = far_matte;
			layer = MixColor(layer, src_lin, focus_mask);
			layer_a = Mix(layer_a, alpha, focus_mask); // sharp overlay = original matte

			// Far bleed-over: a defocused background physically washes over
			// a nearer in-focus silhouette -- every visible background point
			// spreads its light across its full blur disc, including sensor
			// positions that image the focused subject. Without this the
			// subject looks razor "cut out" against the blurred background
			// (the Frischluft-style soft edge the user asked for). Run a
			// reach-gated far gather at focused pixels bordering far content
			// and lay it over the sharp layer with its area-coverage alpha.
			if (focus_mask > 0.02 && !kSkipBleed) {
				if (r->far_alpha_halfres) {
					// Fast mode: the half-res pre-pass already gathered the
					// background reconstruction AND its area-coverage weight
					// at focused pixels bordering far content -- tap both
					// instead of running a third full-res gather here.
					const PF_FpLong wtap = SampleHalfresF(
						r->far_alpha_halfres, r->halfres_w, r->halfres_h, u, v);
					if (wtap > 0.002) {
						const Color3 frgb = SampleFarHalfres(
							r->far_halfres, r->halfres_w, r->halfres_h, u, v);
						const PF_FpLong bleed = Clamp01(wtap) * focus_mask * 0.85;
						layer = MixColor(layer, frgb, bleed);
						if (has_alpha && r->far_matte_halfres) {
							const PF_FpLong fm = SampleHalfresF(
								r->far_matte_halfres, r->halfres_w, r->halfres_h, u, v);
							layer_a = Mix(layer_a, fm, bleed);
						}
					}
				} else {
					const PF_FpLong px_per_coc = 0.15 / std::max(r->inv_w, r->inv_h);
					const PF_FpLong nb_far = ProbeFarReachWide(
						r->signed_coc_cache, r->out_w, r->out_h, x, y, px_per_coc);
					if (nb_far > 0.012) {
						Color3 fb_rgb; PF_FpLong fb_w, fb_matte;
						if (r->gpu_bleed) {
							const float4_gpu g = r->gpu_bleed[pidx];
							fb_rgb = { g.x, g.y, g.z };
							fb_w = g.w;
							// gpu_matte is only allocated for sources that carry
							// transparency -- the CPU path reads fb.matte from a
							// struct field, which is always valid, so translating
							// it to a pointer read needs the same guard the other
							// three matte reads have.
							fb_matte = (has_alpha && r->gpu_matte)
								? r->gpu_matte[pidx].z : alpha;
						} else {
							PassOutput fb = GatherPass<DofPass::Far>(
								*r->pyramid, r->aperture_tex_world, r->iris_mod_world,
								u, v, nb_far, local, depth, signed_coc,
								r->inv_w, r->inv_h,
								r->vogel_luts, r->num_vogel_luts,
								r->signed_coc_cache, r->out_w, r->out_h, r->coc_disc_dist);
							fb_rgb = fb.rgb; fb_w = fb.weight; fb_matte = fb.matte;
						}
						const PF_FpLong bleed = Clamp01(fb_w) * focus_mask * 0.85;
						layer = MixColor(layer, fb_rgb, bleed);
						if (has_alpha) layer_a = Mix(layer_a, fb_matte, bleed);
					}
				}
			}

			final_lin = MixColor(layer, near_rgb, near_alpha_eff);
			// Near (fg) over the composite: matte follows the same blend so the
			// foreground bokeh's feathered silhouette lands on the output alpha.
			if (has_alpha) out_alpha = Mix(layer_a, near_matte, near_alpha_eff);
		}

		final_color = kIsFloat ? final_lin : LinearToGamma(final_lin);

		if (local.enable_highlight) {
			const PF_FpLong hm = ComputeHighlightMask(final_color, local);
			const PF_FpLong lum = Luma(final_color);
			const Color3 gray{lum, lum, lum};
			const PF_FpLong sat = ClampValue<PF_FpLong>(1.0 + local.highlights_saturation, 0.0, 2.0);
			const Color3 sat_color = MixColor(gray, final_color, sat);
			// For 32bpc we must preserve HDR -- the tint multiplication has
			// to stay over-range so bright bokeh cores don't get crushed to
			// 1.0 and go dull.
			const Color3 tinted = kIsFloat
				? Color3{ sat_color.r * local.highlights_tint.r,
				          sat_color.g * local.highlights_tint.g,
				          sat_color.b * local.highlights_tint.b }
				: Color3{ Clamp01(sat_color.r * local.highlights_tint.r),
				          Clamp01(sat_color.g * local.highlights_tint.g),
				          Clamp01(sat_color.b * local.highlights_tint.b) };
			final_color = MixColor(final_color, tinted, hm * Clamp01(local.highlight_boost));
		}

		// ── Grain / Film Noise ────────────────────────────────────────────
		// Adds bipolar grain to the rendered image in display (gamma) space.
		// The whole Grain group was decoded but never applied; it is wired up
		// here. Distribution controls follow the popup semantics:
		//   Luma Distribution: Uniform | Photometric (film-like, peaks in
		//     midtones and fades to nothing at pure black / white).
		//   Grain Map: Fixed | Focus Map (grain only where in focus) |
		//     Blur Amount (grain only in the defocused region -- grain that
		//     lives inside the bokeh, the cinematic look).
		if (local.noise_amount > 0.001) {
			const PF_FpLong seed = local.noise_animated
				? (static_cast<PF_FpLong>(local.current_time) * 0.6180339887 + 11.0)
				: 17.0;
			const PF_FpLong xf = static_cast<PF_FpLong>(x);
			const PF_FpLong yf = static_cast<PF_FpLong>(y);
			PF_FpLong gr, gg, gb;
			if (local.noise_monochromatic) {
				const PF_FpLong g = HashNoise(xf, yf, seed) - 0.5;
				gr = gg = gb = g;
			} else {
				gr = HashNoise(xf, yf, seed)         - 0.5;
				gg = HashNoise(xf, yf, seed + 37.13) - 0.5;
				gb = HashNoise(xf, yf, seed + 71.37) - 0.5;
			}
			PF_FpLong luma_w = 1.0;
			if (local.noise_luma_distribution == 2) {            // Photometric
				const PF_FpLong lum = Luma(final_color);
				luma_w = 4.0 * lum * (1.0 - lum);                // peak at 0.5
			}
			PF_FpLong map_w = 1.0;
			if (local.noise_map_distribution == 2 ||             // Focus Map
			    local.noise_map_distribution == 3) {             // Blur Amount
				const PF_FpLong blur_w = Clamp01(std::abs(signed_coc) * 12.0);
				map_w = (local.noise_map_distribution == 2) ? (1.0 - blur_w) : blur_w;
			}
			const PF_FpLong amt = local.noise_amount * 0.5 * luma_w * map_w;
			final_color.r += gr * amt * local.noise_tint.r;
			final_color.g += gg * amt * local.noise_tint.g;
			final_color.b += gb * amt * local.noise_tint.b;
			if constexpr (!kIsFloat) {
				final_color.r = Clamp01(final_color.r);
				final_color.g = Clamp01(final_color.g);
				final_color.b = Clamp01(final_color.b);
			}
		}
	}

	if (local.display_mode == 2) {
		// Depth Map: the raw depth channel exactly as read (white = near).
		const PF_FpLong d = r->raw_depth_cache[static_cast<size_t>(y * r->out_w + x)];
		final_color = {d, d, d};
	} else if (local.display_mode == 3) {
		// Depth Stabilized: the depth AFTER black/white points, gamma, smoothing
		// and guided edge-snap -- what the CoC is actually derived from. Compare
		// against Depth Map (2) while tuning the depth controls.
		const PF_FpLong d = r->depth_cache[static_cast<size_t>(y * r->out_w + x)];
		final_color = {d, d, d};
	} else if (local.display_mode == 4) {
		// CoC Heat Map: the FINAL signed-CoC field the gather actually consumes
		// -- after auto-range, depth gamma/smoothing, edge protect, boundary snap
		// and field curvature. Orange = near blur, cyan = far blur, black = in
		// focus; brightness = blur magnitude.
		const PF_FpLong sc = r->signed_coc_cache[static_cast<size_t>(y * r->out_w + x)];
		const PF_FpLong mag = std::sqrt(Clamp01(std::abs(sc) * (1.0 / 0.35)));
		final_color = (sc >= 0.0)
			? Color3{mag * 0.10, mag * 0.75, mag}
			: Color3{mag, mag * 0.45, mag * 0.10};
	} else if (local.display_mode == 5) {
		// Focus Peaking: the untouched source with a green overlay on the
		// in-focus band -- aim Focus Distance like on a camera monitor.
		const PF_FpLong sc = r->signed_coc_cache[static_cast<size_t>(y * r->out_w + x)];
		const PF_FpLong t = (1.0 - SmoothStep(0.006, 0.022, std::abs(sc))) * 0.85;
		final_color = {
			original.r * (1.0 - t),
			original.g * (1.0 - t) + t,
			original.b * (1.0 - t)
		};
	} else if (local.display_mode == 7 || local.display_mode == 8) {
		const PF_FpLong out_w_f = static_cast<PF_FpLong>(r->out_w);
		const PF_FpLong out_h_f = static_cast<PF_FpLong>(r->out_h);
		PF_FpLong field_u, field_v;
		PF_FpLong nx, ny;

		if (local.display_mode == 7) {   // single Iris (Iris Array = 8)
			const PF_FpLong min_dim = std::min(out_w_f, out_h_f);
			const PF_FpLong iris_px_r = min_dim * 0.38;
			nx = (u - 0.5) * out_w_f / iris_px_r;
			ny = (v - 0.5) * out_h_f / iris_px_r;
			field_u = 0.5;
			field_v = 0.5;
		} else {
			const PF_FpLong cols = 11.0;
			const PF_FpLong cell_px = out_w_f / cols;
			const PF_FpLong rows = std::max(3.0, std::ceil(out_h_f / cell_px));
			const PF_FpLong cell_w = 1.0 / cols;
			const PF_FpLong cell_h = 1.0 / rows;
			const PF_FpLong iris_px_r = cell_px * 0.40;

			// Tight AA by default. Old 0.08 floor produced the wide
			// semi-transparent halo visible when softness = 0. Softness
			// still widens the fade linearly up to 0.35 for the classic
			// soft-bokeh look.
			const PF_FpLong edge_width = 0.02 + local.softness * 0.33;
			const PF_FpLong edge_start = 1.0 - edge_width;
			// Opponent-axis CA preview (same matrix as the gather).
			const PF_FpLong r_s = 1.0 + ClampValue<PF_FpLong>(( local.ca_rc       - 0.5 * local.ca_gm - 0.5 * local.ca_by) * 0.4, -0.45, 0.45);
			const PF_FpLong g_s = 1.0 + ClampValue<PF_FpLong>((-0.5 * local.ca_rc + local.ca_gm       - 0.5 * local.ca_by) * 0.4, -0.45, 0.45);
			const PF_FpLong b_s = 1.0 + ClampValue<PF_FpLong>((-0.5 * local.ca_rc - 0.5 * local.ca_gm + local.ca_by      ) * 0.4, -0.45, 0.45);

			const A_long base_cx = static_cast<A_long>(std::floor(u * cols));
			const A_long base_cy = static_cast<A_long>(std::floor(v * rows));

			PF_FpLong best_r = 0.0, best_g = 0.0, best_b = 0.0;
			for (A_long oy = -1; oy <= 1; ++oy) {
				for (A_long ox = -1; ox <= 1; ++ox) {
					const A_long icx = base_cx + ox;
					const A_long icy = base_cy + oy;
					if (icx < 0 || icx >= static_cast<A_long>(cols) ||
					    icy < 0 || icy >= static_cast<A_long>(rows)) continue;

					const PF_FpLong cu = (static_cast<PF_FpLong>(icx) + 0.5) * cell_w;
					const PF_FpLong cv = (static_cast<PF_FpLong>(icy) + 0.5) * cell_h;
					const PF_FpLong lnx = (u - cu) * out_w_f / iris_px_r;
					const PF_FpLong lny = (v - cv) * out_h_f / iris_px_r;

					Vec2 lp = Rotate({lnx, lny}, local.bokeh_rotation_rad);
					lp.x *= std::max<PF_FpLong>(0.1, local.anamorphic_ratio);

					if (local.astigmatism > 0.001) {
						const PF_FpLong fcx2 = cu - 0.5;
						const PF_FpLong fcy2 = cv - 0.5;
						const PF_FpLong fd2 = std::sqrt(fcx2 * fcx2 + fcy2 * fcy2);
						const PF_FpLong ef2 = SmoothStep(0.05, 0.55, fd2 * 2.0);
						if (ef2 > 0.001) {
							const PF_FpLong fa2 = std::atan2(fcy2, fcx2);
							// Cap matches the gather (ApplyAstigmatism: 2.5) so the
								// iris-array preview shows the real streak/swirl of
								// strong-astig presets instead of clamping at 1.0.
								const PF_FpLong as2 = std::min(local.astigmatism * ef2, 2.5);
							const Vec2 tang2{-std::sin(fa2), std::cos(fa2)};
							const Vec2 sag2{std::cos(fa2), std::sin(fa2)};
							const PF_FpLong tc2 = lp.x * tang2.x + lp.y * tang2.y;
							const PF_FpLong sc2 = lp.x * sag2.x + lp.y * sag2.y;
							const PF_FpLong stretch = 1.0 + as2 * 0.6;
							const PF_FpLong squeeze = 1.0 / stretch;
							const PF_FpLong ts2 = local.astigmatism_type_sagittal ? stretch : squeeze;
							const PF_FpLong ss2 = local.astigmatism_type_sagittal ? squeeze : stretch;
							lp = {tang2.x * tc2 * ts2 + sag2.x * sc2 * ss2,
							      tang2.y * tc2 * ts2 + sag2.y * sc2 * ss2};
						}
					}

					auto iris_m = [&](PF_FpLong rs) -> PF_FpLong {
						const Vec2 cp2{lp.x * rs, lp.y * rs};
						const PF_FpLong cd2 = Length(cp2);
						if (cd2 >= 1.05) return 0.0;
						const PF_Boolean is_poly2 =
							(local.aperture_shape_mode == 2 || local.aperture_shape_mode == 3);
						PF_FpLong m2;
						if (is_poly2) {
							const PF_FpLong ba_rad = kTau / static_cast<PF_FpLong>(local.aperture_blades);
							const PF_FpLong ns2 = (local.aperture_shape_mode == 3)
								? std::max<PF_FpLong>(0.15, local.notch_scale)
								: local.notch_scale;
							const PF_FpLong nb2 = (local.notch_angle / kPi) * (ba_rad * 0.5);
							// Polygon defines its own edge with narrow AA.
							// Applying the circular SmoothStep on top of it
							// produced the semi-transparent halo the user
							// saw, so only the polygon mask drives the shape
							// here. Optional extra softness is applied below
							// when the user explicitly raises the slider.
							m2 = GetPolygonalAperture(cp2, local.aperture_blades,
								ClampValue<PF_FpLong>(local.blade_curve, -1.0, 1.0),
								ns2, nb2);
							if (local.softness > 0.001) {
								m2 *= 1.0 - SmoothStep(edge_start, 1.0, cd2);
							}
						} else {
							if (cd2 >= 1.0) return 0.0;
							m2 = 1.0 - SmoothStep(edge_start, 1.0, cd2);
							if (local.aperture_shape_mode == 4) {
								m2 *= SampleApertureTextureMask(r->aperture_tex_world,
									cp2, cu, cv, local);
							}
						}
						// Built-in aperture-map library shape.
						if (local.aperture_map_index > 0 && zlux_apmap::Active(r->apmap)) {
							m2 *= zlux_apmap::Sample(r->apmap, cp2.x, cp2.y);
						}
						// Iris Texture modulator (any shape mode).
						if (r->iris_mod_world && local.aperture_shape_mode != 4) {
							m2 *= SampleApertureTextureMask(r->iris_mod_world,
								cp2, cu, cv, local);
						}
						if (local.catadioptric > 0.1)
							m2 *= GetCatadioptricMask(cp2, local.catadioptric);
						if (std::abs(local.spherical_aberration_amount) > 0.001)
							m2 *= ClampValue<PF_FpLong>(
								ComputeSphericalProfile(ClampValue<PF_FpLong>(cd2, 0.0, 1.0), 1.0, local),
								0.0, 2.8);
						if (std::abs(local.vignetting) > 0.001)
							m2 *= GetCatsEyeMask(lp, cu, cv, local.vignetting, local.vignetting_scale);
						m2 *= GetMatteBoxApertureMask(cp2, local);
						return Clamp01(m2);
					};

					best_r = std::max(best_r, iris_m(r_s));
					best_g = std::max(best_g, iris_m(g_s));
					best_b = std::max(best_b, iris_m(b_s));
				}
			}
			final_color = {best_r, best_g, best_b};
			goto iris_done;
		}

		{
		Vec2 p = Rotate({nx, ny}, local.bokeh_rotation_rad);
		p.x *= std::max<PF_FpLong>(0.1, local.anamorphic_ratio);

		// Tight AA by default; softness widens for soft-bokeh look.
		const PF_FpLong edge_width = 0.02 + local.softness * 0.33;
		const PF_FpLong edge_start = 1.0 - edge_width;
		auto iris_mask = [&](PF_FpLong rs) -> PF_FpLong {
			const Vec2 cp{p.x * rs, p.y * rs};
			const PF_FpLong cd = Length(cp);
			const PF_Boolean is_poly = (local.aperture_shape_mode == 2 || local.aperture_shape_mode == 3);
			if (cd >= 1.05) return 0.0;
			PF_FpLong m;
			if (is_poly) {
				const PF_FpLong ba_rad = kTau / static_cast<PF_FpLong>(local.aperture_blades);
				const PF_FpLong ns = (local.aperture_shape_mode == 3)
					? std::max<PF_FpLong>(0.15, local.notch_scale)
					: local.notch_scale;
				const PF_FpLong nb = (local.notch_angle / kPi) * (ba_rad * 0.5);
				// Polygon mask provides its own narrow AA; no outer
				// circular fade unless the user explicitly raises the
				// Softness slider (extra bokeh-style bleed).
				m = GetPolygonalAperture(cp, local.aperture_blades,
					ClampValue<PF_FpLong>(local.blade_curve, -1.0, 1.0),
					ns, nb);
				if (local.softness > 0.001) {
					m *= 1.0 - SmoothStep(edge_start, 1.0, cd);
				}
			} else {
				if (cd >= 1.0) return 0.0;
				m = 1.0 - SmoothStep(edge_start, 1.0, cd);
				if (local.aperture_shape_mode == 4) {
					m *= SampleApertureTextureMask(r->aperture_tex_world, cp, field_u, field_v, local);
				}
			}
			// Built-in aperture-map library shape (same bake as the gather).
			if (local.aperture_map_index > 0 && zlux_apmap::Active(r->apmap)) {
				m *= zlux_apmap::Sample(r->apmap, cp.x, cp.y);
			}
			if (r->iris_mod_world && local.aperture_shape_mode != 4) {
				m *= SampleApertureTextureMask(r->iris_mod_world, cp, field_u, field_v, local);
			}
			if (local.catadioptric > 0.1) {
				m *= GetCatadioptricMask(cp, local.catadioptric);
			}
			if (std::abs(local.spherical_aberration_amount) > 0.001) {
				m *= ClampValue<PF_FpLong>(
					ComputeSphericalProfile(ClampValue<PF_FpLong>(cd, 0.0, 1.0), 1.0, local),
					0.0, 2.8);
			}
			if (std::abs(local.vignetting) > 0.001) {
				m *= GetCatsEyeMask(p, field_u, field_v, local.vignetting, local.vignetting_scale);
			}
			m *= GetMatteBoxApertureMask(cp, local);
			return Clamp01(m);
		};

		// Opponent-axis CA preview (same matrix as the gather).
		const PF_FpLong r_s = 1.0 + ClampValue<PF_FpLong>(( local.ca_rc       - 0.5 * local.ca_gm - 0.5 * local.ca_by) * 0.4, -0.45, 0.45);
		const PF_FpLong g_s = 1.0 + ClampValue<PF_FpLong>((-0.5 * local.ca_rc + local.ca_gm       - 0.5 * local.ca_by) * 0.4, -0.45, 0.45);
		const PF_FpLong b_s = 1.0 + ClampValue<PF_FpLong>((-0.5 * local.ca_rc - 0.5 * local.ca_gm + local.ca_by      ) * 0.4, -0.45, 0.45);

		PF_FpLong rm = iris_mask(r_s);
		PF_FpLong gm = iris_mask(g_s);
		PF_FpLong bm = iris_mask(b_s);

		final_color = {rm, gm, bm};
		}
		iris_done:;
	} else if (local.display_mode == 6) {
		// Selected Highlights: the highlight mask used for bokeh shaping --
		// white where a pixel is treated as a specular highlight. Tune the
		// Lower/Upper Threshold + Softness against this view.
		const PF_FpLong hm = ComputeHighlightMask(original, local);
		final_color = {hm, hm, hm};
	}

	if (local.display_mode != 1) {
		const PF_FpLong dx = u - local.auto_focus_point.x;
		const PF_FpLong dy = v - local.auto_focus_point.y;
		if (std::sqrt(dx * dx + dy * dy) < 0.006) {
			final_color = {1.0, 0.15, 0.15};
		}
	}

	*outP = PixFromColor<PIX>(final_color, out_alpha);
	return PF_Err_NONE;
}

static PF_Err RenderPixel8(void* refcon, A_long x, A_long y, PF_Pixel8* inP, PF_Pixel8* outP)
{
	return RenderPixelImpl<PF_Pixel8>(refcon, x, y, inP, outP);
}

static PF_Err RenderPixel16(void* refcon, A_long x, A_long y, PF_Pixel16* inP, PF_Pixel16* outP)
{
	return RenderPixelImpl<PF_Pixel16>(refcon, x, y, inP, outP);
}

static PF_Err RenderPixelFloat(void* refcon, A_long x, A_long y, PF_PixelFloat* inP, PF_PixelFloat* outP)
{
	return RenderPixelImpl<PF_PixelFloat>(refcon, x, y, inP, outP);
}

// Drop-in parallel replacement for AE's iterate suite over the final per-pixel
// pass. RenderPixelImpl is fully output-pixel-independent and touches only the
// already-checked-out source/output worlds plus the precomputed read-only
// caches in the refcon -- it never calls an AE suite -- so it is safe to invoke
// from worker threads. AE only hands a SmartRender effect a single full-frame
// call (this filter is 1:1 with no buffer expansion, so the frame is never
// tiled), which meant the entire compositing + near-gather pass previously ran
// on one core regardless of the threaded-render flag. Splitting the scanlines
// across all cores here is the dominant speedup for this stage.
template <typename PIX>
static void ParallelApply(PF_EffectWorld* src, PF_EffectWorld* out, RenderRefcon* rc)
{
	const A_long h = out->height;
	const A_long w = out->width;
	ParallelRows(h, 8, [&](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; ++y) {
			for (A_long x = 0; x < w; ++x) {
				PIX* inP  = PixelPtr<PIX>(src, x, y);
				PIX* outP = PixelPtr<PIX>(out, x, y);
				RenderPixelImpl<PIX>(rc, x, y, inP, outP);
			}
		}
	});
}

// Stage profiling for the standalone repro driver (debug/dof_real.cpp defines
// ZLUX_PROFILE before including this TU). Compiles to nothing in plugin builds.
#ifdef ZLUX_PROFILE
struct ZluxStageTimer {
	std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
	void lap(const char* name) {
		const auto n = std::chrono::steady_clock::now();
		std::printf("  [prof] %-18s %7.1f ms\n", name,
			std::chrono::duration<double, std::milli>(n - t).count());
		t = n;
	}
};
#define ZLUX_PROF_DECL ZluxStageTimer zlux_prof_
#define ZLUX_PROF(name) zlux_prof_.lap(name)
#else
#define ZLUX_PROF_DECL ((void)0)
#define ZLUX_PROF(name) ((void)0)
#endif

// Fill the Depth Levels histogram global from the depth layer. Sampled on a
// strided grid (depth maps are smooth, so a coarse sample reproduces the shape
// for free) in the same NORMALIZED d-space the Black/White/Gamma controls use.
template <typename PIX>
static void ComputeDepthHistogramTyped(const PF_EffectWorld* dw, int ch, const DOFSettings& s)
{
	const A_long w = dw->width, h = dw->height;
	if (w < 1 || h < 1) { zlux_depthhist::g_valid = false; return; }
	A_long bins[zlux_depthhist::kBins] = { 0 };
	const A_long stepx = std::max<A_long>(1, w / 512);
	const A_long stepy = std::max<A_long>(1, h / 512);
	for (A_long y = 0; y < h; y += stepy) {
		const PIX* row = PixelPtr<PIX>(const_cast<PF_EffectWorld*>(dw), 0, y);
		for (A_long x = 0; x < w; x += stepx) {
			const PF_FpLong raw = ReadDepthChannelRawPix<PIX>(&row[x], ch);
			const PF_FpLong dn = NormalizeRawDepth(raw, s);
			int b = static_cast<int>(dn * zlux_depthhist::kBins);
			if (b < 0) b = 0; else if (b >= zlux_depthhist::kBins) b = zlux_depthhist::kBins - 1;
			++bins[b];
		}
	}
	A_long mx = 1;
	for (int i = 0; i < zlux_depthhist::kBins; ++i) if (bins[i] > mx) mx = bins[i];
	for (int i = 0; i < zlux_depthhist::kBins; ++i) zlux_depthhist::g_bins[i] = bins[i];
	zlux_depthhist::g_max = mx;
	zlux_depthhist::g_valid = true;
}

static void ComputeDepthHistogram(const PF_EffectWorld* dw, int ch, const DOFSettings& s)
{
	if (PF_WORLD_IS_DEEP(const_cast<PF_EffectWorld*>(dw)))
		ComputeDepthHistogramTyped<PF_Pixel16>(dw, ch, s);
	else if (WorldIsFloat(dw))
		ComputeDepthHistogramTyped<PF_PixelFloat>(dw, ch, s);
	else
		ComputeDepthHistogramTyped<PF_Pixel8>(dw, ch, s);
}

PF_Err RenderCore(PF_InData* in_data, PF_OutData* out_data, PF_EffectWorld* src_world, PF_EffectWorld* output,
                  PF_EffectWorld* depth_world, PF_EffectWorld* aperture_tex_world,
                  PF_EffectWorld* iris_mod_world,
                  const DOFSettings& s, PF_Boolean is_deep, PF_Boolean is_float = FALSE)
{
	PF_Err err = PF_Err_NONE;
	ZLUX_PROF_DECL;
	const A_long out_w = output->width;
	const A_long out_h = output->height;
	const PF_FpLong inv_w = 1.0 / std::max<PF_FpLong>(1.0, static_cast<PF_FpLong>(out_w));
	const PF_FpLong inv_h = 1.0 / std::max<PF_FpLong>(1.0, static_cast<PF_FpLong>(out_h));

	DOFSettings local = s;

	// Render Mode governor (see DOFSettings::render_mode). Fast trades sample
	// density for ~2.5-3x faster scrubbing; Extreme doubles the budget for
	// hero shots with huge Blur Amounts (its other effects -- relaxed tap cap
	// and full-res Far gather -- are applied at their respective sites).
	if (local.render_mode == 1) {
		local.sample_count = ClampValue<A_long>(local.sample_count * 2 / 5, 16, 192);
	} else if (local.render_mode == 3) {
		local.sample_count = ClampValue<A_long>(local.sample_count * 2, 64, kMaxVogelSamples);
	}

	// Precompute the frame-constant Bokeh Gamma weight LUT so the gather can
	// replace its per-tap pow(1 + luma, gamma) with a cheap lerp (luma is in
	// [0,1], gamma is constant across the frame). Only built when the feature
	// is on; otherwise the gather's has_bokeh_gamma branch never reads it.
	if (local.bokeh_gamma > 0.001) {
		for (int i = 0; i <= 256; ++i) {
			const PF_FpLong luma = static_cast<PF_FpLong>(i) / 256.0;
			local.bokeh_gamma_lut[i] = std::pow(1.0 + luma, local.bokeh_gamma);
		}
	}

	// Load the selected built-in aperture-map shape once, up front (single-
	// threaded), so it is available to BOTH the gather bake and the Iris /
	// Iris-Array preview display modes regardless of the active mode.
	// One snapshot for the whole frame, taken up front and held by this
	// shared_ptr until RenderCore returns. Everything downstream -- the Vogel
	// bake, the per-pixel gather on every worker thread, the Iris display modes
	// -- reads through this borrowed pointer, so a concurrent render of another
	// instance loading a DIFFERENT map cannot disturb it.
	const zlux_apmap::ApMapRef apmap_ref = zlux_apmap::LoadApMap(local.aperture_map_index);
	const zlux_apmap::ApMap* const apmap = apmap_ref.get();

	const A_long tiles_x = (out_w + kCocTileSize - 1) / kCocTileSize;
	const A_long tiles_y = (out_h + kCocTileSize - 1) / kCocTileSize;
	const size_t pixel_count = static_cast<size_t>(out_w * out_h);
	// Display modes (1-based popup): 1 Rendered, 2 Depth Map, 3 Depth Stabilized,
	// 4 CoC Heat Map, 5 Focus Peaking, 6 Selected Highlights, 7 Iris, 8 Iris Array.
	// Depth caches (raw + stabilized) are needed by the render and by every depth/
	// CoC-based view (modes 1-5); the iris/highlight previews (6-8) read only the
	// source colour.
	const bool needs_depth = (local.display_mode <= 5);
	// Consumers of the final signed-CoC field: the render itself plus CoC Heat Map
	// (4) and Focus Peaking (5). They need the full CoC pipeline (auto-range,
	// smoothing, protect, boundary snap, field curvature) but none of the gather
	// machinery (pyramid / LUTs / half-res far), which stays render-only.
	const bool needs_coc = (local.display_mode == 1 || local.display_mode == 4 || local.display_mode == 5);
	// Only the actual Rendered view (1) builds the bokeh machinery (pyramid +
	// Vogel LUTs + half-res far layers).
	const bool needs_bokeh = (local.display_mode == 1);

	std::vector<CoCTileData> coc_tiles(static_cast<size_t>(tiles_x * tiles_y));
	std::vector<PF_FpLong> raw_depth_cache(needs_depth ? pixel_count : 1);
	std::vector<float> depth_cache(needs_depth ? pixel_count : 1);
	// float (not double): the signed CoC is read at scattered positions once per
	// Vogel tap in the gather -- the hottest random memory access in the frame.
	// Halving the element size keeps twice as much of the cache resident; the
	// CoC magnitude (~±0.25) has precision to spare in float.
	std::vector<float> signed_coc_cache(needs_coc ? pixel_count : 1);
	// Per-pixel foreground-protection weight. Non-zero only for pixels the
	// user has marked via the Foreground Protect slider as "thin near detail
	// that should be spared from near-blur overlay". Compositing uses it to
	// mask out incoming near alpha so neighbouring near splashes cannot
	// repaint over preserved wire/antenna/branch pixels.
	std::vector<PF_FpLong> protect_mask_cache(needs_coc ? pixel_count : 1, 0.0);
	std::vector<CoCTileData> coc_tiles_dilated;
	std::vector<unsigned char> tile_skip;
	// Distance-to-nearest-CoC-discontinuity field (output px). Built only for the
	// gather path; consumed by GatherPass's silhouette-band test (see below).
	std::vector<float> coc_disc_dist;
	// Finer ladder than the old {16,64,256,512,1024}: PickVogelLUT and the
	// footprint cap compute an exact desired tap count, but the gather then
	// ceils to the next LUT size -- with the coarse ladder a Fast-mode request
	// of 20 taps ran a 64-tap LUT (3x waste) and the render-mode budget cut
	// never actually engaged at small blur radii. The ceil rule is unchanged,
	// so every pixel still gets AT LEAST the tap count the quality model asked
	// for -- only the overshoot shrinks. Heap-allocated: 13 LUTs x 72KB on the
	// stack would risk overflowing AE's render-thread stack.
	constexpr A_long kNumVogelLUTs = 15;
	static const A_long kVogelLUTSizes[kNumVogelLUTs] =
		{16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
	std::vector<VogelLUT> vogel_luts_store;

	if (needs_depth) {
		for (A_long i = 0; i < tiles_x * tiles_y; ++i) {
			coc_tiles[static_cast<size_t>(i)] = { 10.0, -10.0, 10.0, -10.0 };
		}

		if (local.no_depth) {
			local.focal_distance = 0.0;
			const PF_FpLong uniform_coc = ClampValue<PF_FpLong>(local.blur_strength, 0.0, kCocClamp);
			for (size_t i = 0; i < pixel_count; ++i) {
				raw_depth_cache[i] = 1.0;
				depth_cache[i] = 1.0f;
				if (needs_bokeh) signed_coc_cache[i] = static_cast<float>(uniform_coc);
			}
			const CoCTileData uniform_tile = { uniform_coc, uniform_coc, 1.0, 1.0 };
			for (A_long i = 0; i < tiles_x * tiles_y; ++i) {
				coc_tiles[static_cast<size_t>(i)] = uniform_tile;
			}
		} else {
			// ── Depth auto-range ────────────────────────────────────────────
			// Handles non-normalized depth inputs -- most importantly linear-Z
			// EXR passes whose values are not in [0,1] (the legacy path clamped
			// everything past 1.0 to "far", discarding the whole background
			// gradient). Must run before the auto-focus sample and the depth
			// build, since every depth read normalizes through these fields.
			// Conventional [0,1] maps leave auto-range off -> identical to the
			// legacy behavior.
			local.depth_autorange = FALSE;
			local.depth_range_min = 0.0;
			local.depth_range_inv_span = 1.0;
			{
				PF_FpLong rmin = 0.0, rmax = 1.0;
				bool auto_r;
				if (PF_WORLD_IS_DEEP(const_cast<PF_EffectWorld*>(depth_world)))
					auto_r = ComputeDepthAutoRange<PF_Pixel16>(depth_world, local.depth_channel, rmin, rmax);
				else if (WorldIsFloat(depth_world))
					auto_r = ComputeDepthAutoRange<PF_PixelFloat>(depth_world, local.depth_channel, rmin, rmax);
				else
					auto_r = ComputeDepthAutoRange<PF_Pixel8>(depth_world, local.depth_channel, rmin, rmax);
				if (auto_r) {
					local.depth_autorange = TRUE;
					local.depth_range_min = rmin;
					local.depth_range_inv_span = 1.0 / std::max<PF_FpLong>(kEps, rmax - rmin);
				}
			}

			ZLUX_PROF("autorange");
			PF_FpLong focal_distance = local.focal_distance;
			const bool focus_pt_moved = (std::abs(local.auto_focus_point.x - 0.5) > 0.015 ||
			                             std::abs(local.auto_focus_point.y - 0.5) > 0.015);
			if (focus_pt_moved) {
				if (is_deep) {
					focal_distance = SampleWeightedFocusDepth<PF_Pixel16>(depth_world, local, local.auto_focus_point.x, local.auto_focus_point.y, inv_w, inv_h);
				} else {
					focal_distance = SampleWeightedFocusDepth<PF_Pixel8>(depth_world, local, local.auto_focus_point.x, local.auto_focus_point.y, inv_w, inv_h);
				}
			}
			local.focal_distance = focal_distance;

			// Feed the Depth Levels custom-UI histogram (cheap strided pass over
			// the depth layer, in the same normalized d-space the handles use).
			ComputeDepthHistogram(depth_world, local.depth_channel, local);

			const A_long dw = depth_world->width;
			const A_long dh = depth_world->height;
			const int dch = local.depth_channel;
			const bool depth_is_deep = PF_WORLD_IS_DEEP(const_cast<PF_EffectWorld*>(depth_world));
			const bool depth_is_float = !depth_is_deep && WorldIsFloat(depth_world);
			const bool src_is_deep = PF_WORLD_IS_DEEP(const_cast<PF_EffectWorld*>(src_world));
			const bool src_is_float = !src_is_deep && WorldIsFloat(src_world);
			const A_long sw = src_world->width;
			const A_long sh = src_world->height;

			// Cross-shaped bilateral depth reader -- suppresses depth-map noise
			// and quantization steps that otherwise show up as CoC flicker.
			auto read_depth_nn = [&](A_long px, A_long py) -> PF_FpLong {
				px = ClampValue<A_long>(px, 0, dw - 1);
				py = ClampValue<A_long>(py, 0, dh - 1);
				PF_FpLong d;
				if      (depth_is_deep)  d = ReadDepthNN<PF_Pixel16>(depth_world, px, py, dch);
				else if (depth_is_float) d = ReadDepthNN<PF_PixelFloat>(depth_world, px, py, dch);
				else                     d = ReadDepthNN<PF_Pixel8>(depth_world, px, py, dch);
				// Normalize raw -> [0,1]: auto-range for non-[0,1] inputs (linear-Z
				// EXR), legacy clamp otherwise. Also folds in the NaN/Inf guard for
				// 32bpc AOV garbage at subpixel-coverage edges.
				return NormalizeRawDepth(d, local);
			};

			// Luma reader on the source layer at output-resolution coords.
			// Used as the *color edge-stop* for the joint bilateral depth
			// filter below: preserves depth discontinuities that align with
			// real colour silhouettes (typical object boundary) and smooths
			// through quantization noise that does not correlate with the
			// actual image edges. This is what kills flying-pixel halos on
			// thin foreground geometry.
			auto read_luma_at = [&](A_long ox, A_long oy) -> PF_FpLong {
				ox = ClampValue<A_long>(ox, 0, sw - 1);
				oy = ClampValue<A_long>(oy, 0, sh - 1);
				PF_FpLong r = 0.0, g = 0.0, b = 0.0;
				if (src_is_deep) {
					const PF_Pixel16* p = PixelPtr<PF_Pixel16>(src_world, ox, oy);
					r = p->red  * (1.0 / 32768.0);
					g = p->green* (1.0 / 32768.0);
					b = p->blue * (1.0 / 32768.0);
				} else if (src_is_float) {
					const PF_PixelFloat* p = PixelPtr<PF_PixelFloat>(src_world, ox, oy);
					r = p->red; g = p->green; b = p->blue;
				} else {
					const PF_Pixel8* p = PixelPtr<PF_Pixel8>(src_world, ox, oy);
					r = p->red  * (1.0 / 255.0);
					g = p->green* (1.0 / 255.0);
					b = p->blue * (1.0 / 255.0);
				}
				const PF_FpLong y_lum = r * 0.2126 + g * 0.7152 + b * 0.0722;
				return std::isfinite(y_lum) ? Clamp01(y_lum) : 0.0;
			};

			// Colour-domain bilateral weight. Coarser falloff than the depth
			// weight because natural luma variation is wider than quantized
			// depth jitter. 30 here is tuned to kill haloing without smearing
			// genuine depth edges that lack colour contrast.
			auto color_bilateral_w = [](PF_FpLong diff) -> PF_FpLong {
				const PF_FpLong xv = std::abs(diff) * 30.0;
				return 1.0 / (1.0 + xv * xv);
			};

			// Bilinear depth read at output coords. AI depth passes are
			// routinely lower-resolution than the comp; the nearest-texel
			// mapping upscaled them into hard blocks and the CoC inherited the
			// staircase -- visible as jagged silhouette edges once the subject
			// is strongly blurred. Bilinear interpolation of the (already
			// normalized) taps turns the steps into smooth ramps; the
			// occlusion-sliver rescue downstream handles the interpolated band
			// at true depth jumps. The 1:1 case keeps the exact nearest read
			// so full-res depth maps stay bit-identical.
			const bool depth_scaled = (dw != out_w || dh != out_h);
			auto read_depth_center = [&](A_long ox, A_long oy, A_long px, A_long py) -> PF_FpLong {
				if (!depth_scaled) return read_depth_nn(px, py);
				const PF_FpLong fpx = (static_cast<PF_FpLong>(ox) + 0.5) * inv_w * dw - 0.5;
				const PF_FpLong fpy = (static_cast<PF_FpLong>(oy) + 0.5) * inv_h * dh - 0.5;
				const A_long bx = static_cast<A_long>(std::floor(fpx));
				const A_long by = static_cast<A_long>(std::floor(fpy));
				const PF_FpLong tx = fpx - std::floor(fpx);
				const PF_FpLong ty = fpy - std::floor(fpy);
				const PF_FpLong d00 = read_depth_nn(bx,     by);
				const PF_FpLong d10 = read_depth_nn(bx + 1, by);
				const PF_FpLong d01 = read_depth_nn(bx,     by + 1);
				const PF_FpLong d11 = read_depth_nn(bx + 1, by + 1);
				return Mix(Mix(d00, d10, tx), Mix(d01, d11, tx), ty);
			};

			// Joint bilateral: edge-stop by both depth gradient *and* source
			// colour gradient. Cleaner than the old depth-only bilateral at
			// object silhouettes where the depth AOV is jittery but the RGB
			// has a clean edge. We also hand back the peak local colour
			// gradient so the foreground-protect path can tell which near
			// pixels sit on a sharp high-contrast detail (wires, hair,
			// antennas) and need to be rescued from the near gather.
			struct DepthSample { PF_FpLong depth; PF_FpLong contrast; };
			auto read_depth_stable = [&](A_long ox, A_long oy, A_long px, A_long py, PF_FpLong c) -> DepthSample {
				const PF_FpLong l  = read_depth_nn(px - 1, py);
				const PF_FpLong rr = read_depth_nn(px + 1, py);
				const PF_FpLong up = read_depth_nn(px,     py - 1);
				const PF_FpLong dn = read_depth_nn(px,     py + 1);
				const PF_FpLong yC = read_luma_at(ox,     oy);
				const PF_FpLong yL = read_luma_at(ox - 1, oy);
				const PF_FpLong yR = read_luma_at(ox + 1, oy);
				const PF_FpLong yU = read_luma_at(ox,     oy - 1);
				const PF_FpLong yD = read_luma_at(ox,     oy + 1);
				const PF_FpLong wl  = FastBilateralWeight(l  - c) * color_bilateral_w(yL - yC);
				const PF_FpLong wr_ = FastBilateralWeight(rr - c) * color_bilateral_w(yR - yC);
				const PF_FpLong wu  = FastBilateralWeight(up - c) * color_bilateral_w(yU - yC);
				const PF_FpLong wd  = FastBilateralWeight(dn - c) * color_bilateral_w(yD - yC);
				const PF_FpLong sum = c * 2.5 + l * wl + rr * wr_ + up * wu + dn * wd;
				const PF_FpLong tot = 2.5 + wl + wr_ + wu + wd;
				const PF_FpLong contrast = std::max(
					std::max(std::abs(yL - yC), std::abs(yR - yC)),
					std::max(std::abs(yU - yC), std::abs(yD - yC)));
				return { sum / std::max<PF_FpLong>(kEps, tot), contrast };
			};

			const bool fg_protect_active = (local.foreground_protect > 0.001) && needs_coc;

			// Parallelize the per-pixel depth/CoC build across cores. The only
			// shared writes are the coc_tiles min/max reductions; chunking on
			// whole tile-rows (each thread owns a contiguous band of complete
			// 16px tiles) means no two threads ever touch the same tile, so the
			// result is bit-for-bit identical to the serial loop.
			ParallelRows(tiles_y, 1, [&](A_long tr0, A_long tr1) {
			const A_long y_lo = tr0 * kCocTileSize;
			const A_long y_hi = std::min<A_long>(out_h, tr1 * kCocTileSize);
			for (A_long y = y_lo; y < y_hi; ++y) {
				const A_long dy = ClampValue<A_long>(static_cast<A_long>((static_cast<PF_FpLong>(y) + 0.5) * inv_h * dh), 0, dh - 1);
				for (A_long x = 0; x < out_w; ++x) {
					const size_t pidx = static_cast<size_t>(y * out_w + x);
					const A_long dx = ClampValue<A_long>(static_cast<A_long>((static_cast<PF_FpLong>(x) + 0.5) * inv_w * dw), 0, dw - 1);
					const PF_FpLong d_center = read_depth_center(x, y, dx, dy);
					const PF_FpLong raw_depth = RemapDepth(d_center, local);
					const DepthSample ds = read_depth_stable(x, y, dx, dy, d_center);

					const PF_FpLong depth = RemapDepth(ds.depth, local);
					raw_depth_cache[pidx] = raw_depth;
					depth_cache[pidx] = static_cast<float>(depth);
					if (needs_coc) {
						PF_FpLong signed_coc = ComputeSignedCoC(depth, local);

						// Field curvature: depth-independent edge blur. Push
						// the CoC AWAY from zero so near content stays near
						// and far stays far while focused edges defocus --
						// exactly what a curved focal surface does. Applied
						// before the tile reductions so dilation, gathers and
						// the sliver logic all see it natively.
						if (local.field_curvature > 0.001) {
							const PF_FpLong fcoc = FieldCurvatureCoc(
								(static_cast<PF_FpLong>(x) + 0.5) * inv_w,
								(static_cast<PF_FpLong>(y) + 0.5) * inv_h,
								local.field_curvature, local.field_sweet);
							signed_coc += (signed_coc >= 0.0) ? fcoc : -fcoc;
						}

						// Foreground detail rescue. AI-generated depth maps
						// routinely misclassify thin sharp structures (wires,
						// fences, antennas, branches) as near-field because
						// they get a tiny-but-nonzero disparity. The near
						// gather then erases them completely. When the user
						// dials in foreground_protect > 0 we detect these
						// pixels by their peak local luma gradient: a near
						// pixel sitting on a strong colour edge is far more
						// likely a flying-pixel thin detail than an actual
						// chunky foreground object -- chunky objects are
						// surrounded by similar near pixels and show *lower*
						// local contrast. Attenuating only near CoC leaves
						// genuine foreground blur (blocky objects, skin,
						// broad motion) untouched.
						//
						// Two-sided protect: in addition to muting this
						// pixel's own near CoC we record the rescue weight
						// so the compositor can also deny *incoming* near
						// splashes from neighbouring pixels that would
						// otherwise paint over the preserved wire. Without
						// this second half the near layer paints on top of
						// the sharp overlay and the wire disappears again.
						// Gated on signed_coc < 0 so bright background
						// highlights (stars etc.) are not accidentally
						// preserved.
						PF_FpLong rescue = 0.0;
						if (fg_protect_active && signed_coc < 0.0) {
							const PF_FpLong contrast_weight = SmoothStep(0.06, 0.28, ds.contrast);
							rescue = Clamp01(local.foreground_protect * contrast_weight);
							// The CoC damping that used to happen inline here
							// moved to the thinness-filtered post-pass below:
							// contrast alone also fires on the whole silhouette
							// of a big defocused subject, and "protecting" a
							// silhouette leaves a sharp rim + a bright base-
							// layer halo. The post-pass keeps the rescue only
							// for genuinely thin structures.
						}

						signed_coc_cache[pidx] = static_cast<float>(signed_coc);
						protect_mask_cache[pidx] = rescue;

						const A_long tx = x / kCocTileSize;
						const A_long ty = y / kCocTileSize;
						CoCTileData& tile = coc_tiles[static_cast<size_t>(ty * tiles_x + tx)];
						tile.min_coc = std::min(tile.min_coc, signed_coc);
						tile.max_coc = std::max(tile.max_coc, signed_coc);
						tile.min_depth = std::min(tile.min_depth, depth);
						tile.max_depth = std::max(tile.max_depth, depth);
					}
				}
			}
			});


			// ── Guided-filter depth upsample (colour-guided, He et al.) ──────
			// The depth layer AE feeds us is frequently lower-resolution or
			// compression-blocked (mp4/H.264 macroblocks, nearest-upscaled AOVs).
			// The bilinear read + joint-bilateral above DENOISE it but cannot
			// MOVE a hard staircase onto the true object boundary, so a strongly
			// defocused silhouette stair-steps and leaks. A guided filter fixes
			// this at the root: within each window it linearly fits
			// depth ≈ a·luma + b against the sharp full-res source luma, then
			// averages the (a,b) fits. Where depth correlates with a colour edge
			// (a real silhouette) the fit snaps the depth transition crisply onto
			// that edge; where colour is flat (cov→0 ⇒ a→0) the output is just the
			// local mean, so flat regions stay flat and smooth depth GRADIENTS are
			// preserved (a box-mean of a ramp is the same ramp) — it never
			// flattens the FL depth gradient. This is why it succeeds where the
			// v2.18.0 cross-bilateral BLUR failed: a blur halts at the colour edge
			// and leaves the block intact, whereas the linear fit relocates the
			// edge. Only the CoC/render path needs it; runs before the CoC is
			// re-derived so every downstream stage (rescue damping, bilateral
			// refine, depth smoothing, occlusion snap, the gather) sees clean depth.
#if 0 // ZLUX: depth edge-processing (guided-filter depth snap) removed per user request
			if (needs_coc) {
				std::vector<float> luma_buf(pixel_count);
				ParallelRows(out_h, 16, [&](A_long y0, A_long y1) {
					for (A_long y = y0; y < y1; ++y)
						for (A_long x = 0; x < out_w; ++x)
							luma_buf[static_cast<size_t>(y) * out_w + x] =
								static_cast<float>(read_luma_at(x, y));
				});

				// Separable box-mean (radius r, edge-clamped) via an incremental
				// sliding window: O(1) per pixel regardless of r. scratch holds the
				// horizontal partial, so the horizontal pass fully consumes src
				// before the vertical pass writes dst.
				auto box_mean = [&](const std::vector<float>& src,
				                    std::vector<float>& dst,
				                    std::vector<float>& scratch, A_long r) {
					ParallelRows(out_h, 8, [&](A_long y0, A_long y1) {
						for (A_long y = y0; y < y1; ++y) {
							const float* in  = src.data()     + static_cast<size_t>(y) * out_w;
							float*       out = scratch.data() + static_cast<size_t>(y) * out_w;
							double acc = 0.0;
							const A_long hi0 = std::min<A_long>(r, out_w - 1);
							for (A_long i = 0; i <= hi0; ++i) acc += in[i];
							for (A_long x = 0; x < out_w; ++x) {
								const A_long lo = std::max<A_long>(0, x - r);
								const A_long hi = std::min<A_long>(out_w - 1, x + r);
								out[x] = static_cast<float>(acc / static_cast<double>(hi - lo + 1));
								const A_long add_idx = x + r + 1;
								const A_long rem_idx = x - r;
								if (add_idx < out_w) acc += in[add_idx];
								if (rem_idx >= 0)    acc -= in[rem_idx];
							}
						}
					});
					ParallelRows(out_w, 8, [&](A_long x0, A_long x1) {
						for (A_long x = x0; x < x1; ++x) {
							double acc = 0.0;
							const A_long hi0 = std::min<A_long>(r, out_h - 1);
							for (A_long i = 0; i <= hi0; ++i)
								acc += scratch[static_cast<size_t>(i) * out_w + x];
							for (A_long y = 0; y < out_h; ++y) {
								const A_long lo = std::max<A_long>(0, y - r);
								const A_long hi = std::min<A_long>(out_h - 1, y + r);
								dst[static_cast<size_t>(y) * out_w + x] =
									static_cast<float>(acc / static_cast<double>(hi - lo + 1));
								const A_long add_idx = y + r + 1;
								const A_long rem_idx = y - r;
								if (add_idx < out_h) acc += scratch[static_cast<size_t>(add_idx) * out_w + x];
								if (rem_idx >= 0)    acc -= scratch[static_cast<size_t>(rem_idx) * out_w + x];
							}
						}
					});
				};

				const size_t n = pixel_count;
				std::vector<float> meanI(n), meanP(n), corrIp(n), corrII(n);
				std::vector<float> scratch(n), coefA(n), coefB(n), corrPP(n);
				// Window radius: wide enough to bridge typical macroblock / low-res
				// depth steps without smearing fine real structure. eps governs how
				// hard the fit follows colour. The previous 1.0e-3 was far below the
				// He et al. range for [0,1] luma (≈0.01–0.04) and made the fit chase
				// colour even where depth is flat -- so a texture-rich but uniformly
				// FAR wall (windows, brick) had its colour pattern injected into the
				// depth field, giving a lumpy CoC and a blur that preserved building
				// structure instead of dissolving it. Raised to 0.012, AND the fit is
				// now gated by local DEPTH variance (see vlo/vhi below): the colour-
				// snapped result is used only where depth genuinely steps (real
				// silhouettes); flat-depth regions fall back to the plain box-mean so
				// they stay smooth. This kills the "structured far blur" artefact
				// without losing the silhouette de-staircasing v2.18 added.
				const A_long  gr   = 12;
				const double  geps = 1.2e-2;
				// Depth-variance gate. Compression noise / macroblocks give tiny
				// local depth variance; a true near↔far boundary in a [0,1] depth
				// map gives a large one. Blend from box-mean (flat) to guided (edge).
				const PF_FpLong vlo = 0.004;
				const PF_FpLong vhi = 0.020;

				// coefA / coefB / corrPP double as scratch for the I·p, I·I and P·P
				// products before they are box-averaged into the correlation means.
				ParallelRows(out_h, 16, [&](A_long y0, A_long y1) {
					for (A_long y = y0; y < y1; ++y)
						for (A_long x = 0; x < out_w; ++x) {
							const size_t i = static_cast<size_t>(y) * out_w + x;
							const float I = luma_buf[i], P = depth_cache[i];
							coefA[i]  = I * P;
							coefB[i]  = I * I;
							corrPP[i] = P * P;
						}
				});
				box_mean(luma_buf,    meanI,  scratch, gr);
				box_mean(depth_cache, meanP,  scratch, gr);   // meanP = E[P] (kept)
				box_mean(coefA,       corrIp, scratch, gr);
				box_mean(coefB,       corrII, scratch, gr);
				box_mean(corrPP,      corrPP, scratch, gr);    // in-place → E[P²]

				ParallelRows(out_h, 16, [&](A_long y0, A_long y1) {
					for (A_long y = y0; y < y1; ++y)
						for (A_long x = 0; x < out_w; ++x) {
							const size_t i = static_cast<size_t>(y) * out_w + x;
							const double mI = meanI[i], mP = meanP[i];
							const double varI  = static_cast<double>(corrII[i]) - mI * mI;
							const double covIp = static_cast<double>(corrIp[i]) - mI * mP;
							const double a = covIp / (varI + geps);
							coefA[i] = static_cast<float>(a);
							coefB[i] = static_cast<float>(mP - a * mI);
						}
				});
				box_mean(coefA, meanI,  scratch, gr);   // mean_a → meanI
				box_mean(coefB, corrIp, scratch, gr);   // mean_b → corrIp (meanP kept)

				ParallelRows(out_h, 16, [&](A_long y0, A_long y1) {
					for (A_long y = y0; y < y1; ++y)
						for (A_long x = 0; x < out_w; ++x) {
							const size_t i = static_cast<size_t>(y) * out_w + x;
							// Guided (colour-snapped) estimate.
							const double q = static_cast<double>(meanI[i]) * luma_buf[i]
							               + static_cast<double>(corrIp[i]);
							// Local depth variance → edge confidence.
							const PF_FpLong mP = meanP[i];
							const PF_FpLong varP = std::max<PF_FpLong>(0.0,
								static_cast<PF_FpLong>(corrPP[i]) - mP * mP);
							const PF_FpLong w = SmoothStep(vlo, vhi, varP);
							// Flat depth → smooth box-mean; real edge → snapped fit.
							const double out_d = mP + w * (q - mP);
							depth_cache[i] = static_cast<float>(Clamp01(out_d));
						}
				});

				// Re-derive the signed CoC + tile min/max from the cleaned depth
				// (undamped; the foreground-protect rescue weights set above are
				// preserved and applied by the thinness pass below). Mirrors the
				// CoC half of the main build loop exactly so the gather sees a
				// consistent depth/CoC/tile triple.
				for (A_long i = 0; i < tiles_x * tiles_y; ++i)
					coc_tiles[static_cast<size_t>(i)] = { 10.0, -10.0, 10.0, -10.0 };
				ParallelRows(tiles_y, 1, [&](A_long tr0, A_long tr1) {
					const A_long y_lo = tr0 * kCocTileSize;
					const A_long y_hi = std::min<A_long>(out_h, tr1 * kCocTileSize);
					for (A_long y = y_lo; y < y_hi; ++y) {
						for (A_long x = 0; x < out_w; ++x) {
							const size_t pidx = static_cast<size_t>(y) * out_w + x;
							const PF_FpLong depth = depth_cache[pidx];
							PF_FpLong signed_coc = ComputeSignedCoC(depth, local);
							if (local.field_curvature > 0.001) {
								const PF_FpLong fcoc = FieldCurvatureCoc(
									(static_cast<PF_FpLong>(x) + 0.5) * inv_w,
									(static_cast<PF_FpLong>(y) + 0.5) * inv_h,
									local.field_curvature, local.field_sweet);
								signed_coc += (signed_coc >= 0.0) ? fcoc : -fcoc;
							}
							signed_coc_cache[pidx] = static_cast<float>(signed_coc);
							const A_long tx = x / kCocTileSize;
							const A_long ty = y / kCocTileSize;
							CoCTileData& tile = coc_tiles[static_cast<size_t>(ty * tiles_x + tx)];
							tile.min_coc   = std::min(tile.min_coc,   signed_coc);
							tile.max_coc   = std::max(tile.max_coc,   signed_coc);
							tile.min_depth = std::min(tile.min_depth, depth);
							tile.max_depth = std::max(tile.max_depth, depth);
						}
					}
				});
			}
#endif // ZLUX: end guided-filter depth snap removal


			// ── Foreground-protect thinness filter ──────────────────────────
			// The contrast heuristic above flags EVERY high-contrast near
			// pixel, including the entire silhouette of a big defocused
			// subject against a bright background. Protecting a silhouette is
			// always wrong: focus_mask = max(..., protect) leaves a sharp rim
			// on a blurred subject, and near_alpha *= (1 - protect) punches
			// the (often bright) base layer through the near splash -- the
			// halo along the shoulder the user flagged. Genuine rescue
			// targets are THIN structures (wires, hair, antennas): near
			// pixels with non-near content on BOTH opposite sides within a
			// few pixels. Probe the freshly built signed CoC along 4 axes at
			// ±2/±4 px and keep the rescue only where some axis is thin.
			// Runs as two passes so every thinness probe reads undamped CoC
			// regardless of thread scheduling; the second pass applies the
			// CoC damping the build loop used to do inline.
			if (fg_protect_active && needs_coc) {
				ParallelRows(out_h, 32, [&](A_long py0, A_long py1) {
					for (A_long y = py0; y < py1; ++y) {
						for (A_long x = 0; x < out_w; ++x) {
							const size_t pidx = static_cast<size_t>(y) * out_w + x;
							const PF_FpLong rescue = protect_mask_cache[pidx];
							if (rescue <= 0.001) continue;
							static const A_long axes[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
							PF_FpLong thin = 0.0;
							// Small offsets only (±2/±3): genuine rescue
							// targets (wires, hair) are a couple of pixels
							// wide. Larger windows started matching smeared
							// silhouette bands and let ghost-sharp streaks
							// through at high Edge Protect values. Tight
							// threshold: both sides must be CLEARLY out of
							// the near field.
							for (int a = 0; a < 4 && thin < 0.999; ++a) {
								for (A_long o = 2; o <= 3; ++o) {
									const A_long xa = ClampValue<A_long>(x - axes[a][0] * o, 0, out_w - 1);
									const A_long ya = ClampValue<A_long>(y - axes[a][1] * o, 0, out_h - 1);
									const A_long xb = ClampValue<A_long>(x + axes[a][0] * o, 0, out_w - 1);
									const A_long yb = ClampValue<A_long>(y + axes[a][1] * o, 0, out_h - 1);
									const PF_FpLong ca = signed_coc_cache[static_cast<size_t>(ya) * out_w + xa];
									const PF_FpLong cb = signed_coc_cache[static_cast<size_t>(yb) * out_w + xb];
									// Both sides clear of the near field => thin along this axis.
									const PF_FpLong t = SmoothStep(-0.012, -0.002, std::min(ca, cb));
									if (t > thin) thin = t;
								}
							}
							// Centre-solidity gate: a genuinely misclassified
							// wire carries a STRONG bogus near CoC; weakly-near
							// pixels are AA fringe / rim-light on a silhouette
							// ramp. The tangent-axis probes along an edge read
							// other ramp pixels as "clear" and slipped bright
							// rim specks through at high Edge Protect -- this
							// gate closes that loophole.
							thin *= SmoothStep(0.008, 0.03,
								-static_cast<PF_FpLong>(signed_coc_cache[pidx]));
							protect_mask_cache[pidx] = rescue * thin;
						}
					}
				});
				ParallelRows(out_h, 32, [&](A_long py0, A_long py1) {
					for (A_long y = py0; y < py1; ++y) {
						for (A_long x = 0; x < out_w; ++x) {
							const size_t pidx = static_cast<size_t>(y) * out_w + x;
							const PF_FpLong rescue = protect_mask_cache[pidx];
							if (rescue > 0.001) {
								signed_coc_cache[pidx] =
									static_cast<float>(signed_coc_cache[pidx] * (1.0 - rescue));
							}
						}
					}
				});
			}
		}
	}

	ZLUX_PROF("depth+coc build");

	// ── #4 Bilateral CoC refinement ─────────────────────────────────────
	//
	// AI / disparity-derived depth maps quantize to discrete gradient steps,
	// so the CoC derived from them shows a matching staircase -- visible as
	// concentric rings of constant blur around subjects, especially when
	// the DOF slope is steep. A 3×3 bilateral smoothing pass on the signed
	// CoC with depth as the guidance keeps genuine depth edges crisp (the
	// sigma_d gate attenuates any neighbour more than ~2% of unit depth
	// away) while softening the staircase into smooth transitions. Pixels
	// that foreground-protect marked as flying thin details are bypassed so
	// their -> 0 damping survives; otherwise bilateral averaging would pull
	// them back toward the blurred foreground. Cost: 9 reads per pixel,
	// single cheap exp per neighbour, bounded by the depth cache build
	// cost (well under 1% of total frame time in practice).
	if (needs_coc && !local.no_depth) {
#if 0 // ZLUX: bilateral CoC refine (depth edge smoothing) removed per user request
		std::vector<float> tmp_coc(signed_coc_cache.size());
		const PF_FpLong sigma_d = 0.02;
		const PF_FpLong inv_2ssd = 1.0 / (2.0 * sigma_d * sigma_d);
		// Per-pixel 3x3 bilateral; each output writes only its own tmp_coc[pidx]
		// and reads from the (now read-only) depth/coc caches, so it is fully
		// row-independent and parallelizes without any reduction.
		ParallelRows(out_h, 32, [&](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; ++y) {
			for (A_long x = 0; x < out_w; ++x) {
				const size_t pidx = static_cast<size_t>(y * out_w + x);
				if (protect_mask_cache[pidx] > 0.01) {
					tmp_coc[pidx] = signed_coc_cache[pidx];
					continue;
				}
				const PF_FpLong center_d = depth_cache[pidx];
				PF_FpLong sum_c = 0.0, sum_w = 0.0;
				for (A_long dyk = -1; dyk <= 1; ++dyk) {
					const A_long ny = ClampValue<A_long>(y + dyk, 0, out_h - 1);
					for (A_long dxk = -1; dxk <= 1; ++dxk) {
						const A_long nx = ClampValue<A_long>(x + dxk, 0, out_w - 1);
						const size_t nidx = static_cast<size_t>(ny * out_w + nx);
						const PF_FpLong d_diff = depth_cache[nidx] - center_d;
						const PF_FpLong w = std::exp(-d_diff * d_diff * inv_2ssd);
						sum_c += signed_coc_cache[nidx] * w;
						sum_w += w;
					}
				}
				tmp_coc[pidx] = static_cast<float>((sum_w > kEps) ? (sum_c / sum_w) : signed_coc_cache[pidx]);
			}
		}
		});
		signed_coc_cache.swap(tmp_coc);
		ZLUX_PROF("bilateral refine");
#endif // ZLUX: end bilateral CoC refine removal

		// ── Depth Smoothing ──────────────────────────────────────────────
		// User-driven separable smoothing on the signed CoC -- the DOF PRO
		// "depth aliasing" tool. Erases the staircase/banding an integer or
		// AI depth pass leaves in the CoC, for buttery focus transitions.
		//
		// Depth-GUIDED, not a blind box blur: each neighbour is weighted by
		// its depth similarity, so smoothing never averages across a real
		// object boundary. The old box blur smeared the near field across
		// silhouettes; at high radii the zero-CoC contour migrated tens of
		// pixels away from the true edge and rendered as a sharp ghost ring
		// floating around the subject (the dark arc above the head in the
		// user's max-settings screenshot). Foreground-protected thin-detail
		// pixels are skipped so their rescue survives. Runs only when the
		// slider is non-zero.
		if (local.depth_smoothing > 0.001) {
			const A_long r = std::max<A_long>(1, static_cast<A_long>(std::lround(local.depth_smoothing * 24.0)));
			const PF_FpLong sigma_s = 0.03; // depth-similarity gate (unit depth)
			const PF_FpLong inv_2ss = 1.0 / (2.0 * sigma_s * sigma_s);
			std::vector<float> tmpb(signed_coc_cache.size());
			ParallelRows(out_h, 16, [&](A_long y0, A_long y1) {
				for (A_long y = y0; y < y1; ++y) {
					const size_t row = static_cast<size_t>(y) * out_w;
					for (A_long x = 0; x < out_w; ++x) {
						const size_t idx = row + static_cast<size_t>(x);
						if (protect_mask_cache[idx] > 0.01) {
							tmpb[idx] = signed_coc_cache[idx];
							continue;
						}
						const PF_FpLong d0 = depth_cache[idx];
						PF_FpLong sum = 0.0, wsum = 0.0;
						for (A_long k = -r; k <= r; ++k) {
							const size_t nidx = row + static_cast<size_t>(ClampValue<A_long>(x + k, 0, out_w - 1));
							const PF_FpLong dd = depth_cache[nidx] - d0;
							const PF_FpLong w = std::exp(-dd * dd * inv_2ss);
							sum += signed_coc_cache[nidx] * w;
							wsum += w;
						}
						tmpb[idx] = static_cast<float>(sum / std::max<PF_FpLong>(wsum, kEps));
					}
				}
			});
			ParallelRows(out_w, 16, [&](A_long x0, A_long x1) {
				for (A_long x = x0; x < x1; ++x) {
					for (A_long y = 0; y < out_h; ++y) {
						const size_t idx = static_cast<size_t>(y) * out_w + x;
						if (protect_mask_cache[idx] > 0.01) continue;
						const PF_FpLong d0 = depth_cache[idx];
						PF_FpLong sum = 0.0, wsum = 0.0;
						for (A_long k = -r; k <= r; ++k) {
							const size_t nidx = static_cast<size_t>(ClampValue<A_long>(y + k, 0, out_h - 1)) * out_w + static_cast<size_t>(x);
							const PF_FpLong dd = depth_cache[nidx] - d0;
							const PF_FpLong w = std::exp(-dd * dd * inv_2ss);
							sum += tmpb[nidx] * w;
							wsum += w;
						}
						signed_coc_cache[idx] = static_cast<float>(sum / std::max<PF_FpLong>(wsum, kEps));
					}
				}
			});
		}

		// ── Depth-field filters on the GPU ─────────────────────────────────
		// Boundary snap, despeckle and declutter run back to back over the same
		// CoC field, so one call does all three with a single upload/download
		// instead of three round trips. Each stage keeps its CPU implementation
		// below; ZLUX_CPUSNAP / ZLUX_CPUSPECK / ZLUX_CPUDECL force any of them
		// back so they can be A/B'd individually against the reference.
		bool zlux_snap_on_gpu = false, zlux_speck_on_gpu = false,
		     zlux_depth_filters_on_gpu = false;
#ifdef ZLUX_CUDA
		if (!local.no_depth && zlux_gpu::Enabled()) {
			static const bool kCpuSnap  = (std::getenv("ZLUX_CPUSNAP")  != nullptr);
			static const bool kCpuSpeck = (std::getenv("ZLUX_CPUSPECK") != nullptr);
			static const bool kCpuDecl  = (std::getenv("ZLUX_CPUDECL")  != nullptr);
			if (!(kCpuSnap && kCpuSpeck && kCpuDecl)) {
				// Lock BEFORE asking for the context: CtxLocked() may create it,
				// and the creation itself has to be serialised across MFR threads.
				std::lock_guard<std::mutex> lk(zlux_gpu::DeviceMutex());
				if (ZluxGpuContext* dctx = zlux_gpu::CtxLocked()) {
					std::vector<float> protect_f(static_cast<size_t>(pixel_count));
					for (size_t i = 0; i < protect_f.size(); ++i)
						protect_f[i] = static_cast<float>(protect_mask_cache[i]);
					std::vector<float> filtered(static_cast<size_t>(pixel_count));
					if (zluxGpuDepthFilters(dctx, signed_coc_cache.data(), protect_f.data(),
					                        out_w, out_h,
					                        kCpuSnap  ? 0 : 1,
					                        kCpuSpeck ? 0 : 1,
					                        kCpuDecl  ? 0 : 1,
					                        filtered.data()) == 0) {
						signed_coc_cache.swap(filtered);
						zlux_snap_on_gpu           = !kCpuSnap;
						zlux_speck_on_gpu          = !kCpuSpeck;
						zlux_depth_filters_on_gpu  = !kCpuDecl;
					}
				}
			}
		}
#endif

		// ── Occlusion-boundary snap (real-lens edge model) ──────────────
		if (!zlux_snap_on_gpu)
		// A real foreground object has a hard geometric edge: its defocus
		// disc spreads OUTWARD at full strength starting exactly at the
		// silhouette, and nothing "fades" across the boundary. AI depth
		// maps (and the bilinear depth upscale) instead ramp the depth over
		// several pixels, so the near CoC decays to zero along the contact
		// band with the background. Because every gather tap is gated by
		// its own CoC reach, that faded band under-splashes -- its coverage
		// alpha dips and the bright background reconstruction leaks through
		// as a glowing halo hugging the silhouette. Restore the physical
		// behaviour by snapping confirmed contact-band pixels (the sliver
		// detector requires BOTH a clearly-near and a clearly-far field
		// within ±6 px, so focused subjects' edges never qualify) to the
		// most-near CoC found within ±4 px: the band re-joins the
		// foreground at full blur strength, exactly like a hard-edged
		// object in front of a real lens. Runs on a copy so probe reads
		// are deterministic regardless of thread scheduling.
		{
			// Smooth morphological erosion (sliding-window minimum, radius 6,
			// separable O(n) via a monotonic deque). The previous sparse
			// 4-axis probe produced 4px facets / chunky steps along wide AI-
			// depth ramps; a true windowed minimum shifts by one pixel per
			// pixel and stays visually smooth.
			const A_long erode_r = 6;
			std::vector<float> tmp_e(signed_coc_cache.size());
			std::vector<float> eroded(signed_coc_cache.size());
			auto slide_min = [erode_r](const float* in, float* out, A_long n, A_long stride,
			                           std::vector<A_long>& dq) {
				A_long head = 0, tail = 0;
				for (A_long i = 0; i < n + erode_r; ++i) {
					if (i < n) {
						const float v = in[static_cast<size_t>(i) * stride];
						while (tail > head && in[static_cast<size_t>(dq[tail - 1]) * stride] >= v) --tail;
						dq[tail++] = i;
					}
					const A_long o = i - erode_r;
					if (o >= 0) {
						while (dq[head] < o - erode_r) ++head;
						out[static_cast<size_t>(o) * stride] = in[static_cast<size_t>(dq[head]) * stride];
					}
				}
			};
			ParallelRows(out_h, 16, [&](A_long y0, A_long y1) {
				std::vector<A_long> dq(static_cast<size_t>(out_w));
				for (A_long y = y0; y < y1; ++y) {
					const size_t row = static_cast<size_t>(y) * out_w;
					slide_min(&signed_coc_cache[row], &tmp_e[row], out_w, 1, dq);
				}
			});
			ParallelRows(out_w, 16, [&](A_long x0, A_long x1) {
				std::vector<A_long> dq(static_cast<size_t>(out_h));
				for (A_long x = x0; x < x1; ++x) {
					slide_min(&tmp_e[static_cast<size_t>(x)], &eroded[static_cast<size_t>(x)],
					          out_h, out_w, dq);
				}
			});

			std::vector<float> snapped(signed_coc_cache.size());
			ParallelRows(out_h, 32, [&](A_long y0, A_long y1) {
				for (A_long y = y0; y < y1; ++y) {
					for (A_long x = 0; x < out_w; ++x) {
						const size_t pidx = static_cast<size_t>(y) * out_w + x;
						const PF_FpLong c = signed_coc_cache[pidx];
						const PF_FpLong sliver = DetectCocSliver(
							signed_coc_cache.data(), out_w, out_h, x, y, c);
						snapped[pidx] = (sliver > 0.01)
							? static_cast<float>(Mix(c, eroded[pidx], sliver))
							: static_cast<float>(c);
					}
				}
			});
			signed_coc_cache.swap(snapped);
		}
		ZLUX_PROF("boundary snap");

		// ── Isolated-focus-speck suppression ────────────────────────────────
		if (!zlux_speck_on_gpu)
		// Real depth sources are noisy exactly where it hurts most: a renderer
		// writes EITHER the glass surface's Z OR the background's Z per pixel
		// on semi-transparent / reflective surfaces, AI depth passes jitter,
		// and AA'd silhouettes interpolate through the focal distance. Any
		// such pixel whose CoC lands near zero INSIDE an otherwise strongly
		// blurred field claims "in focus", keeps the razor-sharp (and often
		// grainy) source texel, and renders as scattered sharp garbage specks
		// floating in smooth bokeh -- the "broken pixelated patches" look.
		// DetectCocSliver only rescues the two-sided case (clearly-near AND
		// clearly-far in the window); this pass handles the one-sided case:
		// a focus claim surrounded by blur in EVERY direction is depth noise,
		// not a focused object, and is re-joined to the surrounding blur.
		//
		// Discrimination: probe 4 axes at ±3/±6 px. A side counts as blurred
		// only if BOTH its probes are blurred (min), an axis only if BOTH
		// sides are (min), and the pixel only if ALL axes are (min) -- so any
		// genuinely focused structure wider than ~6 px, or any contour the
		// focus plane sweeps through smoothly (its iso-focus axis stays
		// focused), is left untouched. Replacement is the mean of the blurred
		// probes, so the speck adopts the local field instead of a constant.
		// Foreground-protected pixels are exempt (their sharpness is the
		// user's explicit request).
		{
			std::vector<float> despeck(signed_coc_cache.size());
			ParallelRows(out_h, 32, [&](A_long y0, A_long y1) {
				for (A_long y = y0; y < y1; ++y) {
					for (A_long x = 0; x < out_w; ++x) {
						const size_t pidx = static_cast<size_t>(y) * out_w + x;
						const PF_FpLong c = signed_coc_cache[pidx];
						const PF_FpLong ac = std::abs(c);
						if (ac >= 0.02 || protect_mask_cache[pidx] > 0.01) {
							despeck[pidx] = static_cast<float>(c);
							continue;
						}
						static const A_long axes[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
						PF_FpLong surround = 10.0;
						PF_FpLong repl_sum = 0.0;
						A_long repl_n = 0;
						for (int a = 0; a < 4; ++a) {
							PF_FpLong side_mag[2];
							for (int sgn = 0; sgn < 2; ++sgn) {
								const A_long dir = sgn ? 1 : -1;
								PF_FpLong m = 10.0;
								for (A_long o = 3; o <= 6; o += 3) {
									const A_long px = ClampValue<A_long>(x + axes[a][0] * dir * o, 0, out_w - 1);
									const A_long py = ClampValue<A_long>(y + axes[a][1] * dir * o, 0, out_h - 1);
									const PF_FpLong cv = signed_coc_cache[static_cast<size_t>(py) * out_w + px];
									const PF_FpLong av = std::abs(cv);
									if (av < m) m = av;
									if (av >= 0.012) { repl_sum += cv; ++repl_n; }
								}
								side_mag[sgn] = m;
							}
							const PF_FpLong axis_blocked = std::min(side_mag[0], side_mag[1]);
							if (axis_blocked < surround) surround = axis_blocked;
						}
						// Suppression ramps in once every direction is clearly
						// blurred and the centre clearly claims focus.
						const PF_FpLong t = SmoothStep(0.015, 0.035, surround)
						                  * (1.0 - SmoothStep(0.008, 0.02, ac));
						if (t > 0.01 && repl_n > 0) {
							const PF_FpLong repl = repl_sum / static_cast<PF_FpLong>(repl_n);
							despeck[pidx] = static_cast<float>(Mix(c, repl, t));
						} else {
							despeck[pidx] = static_cast<float>(c);
						}
					}
				}
			});
			signed_coc_cache.swap(despeck);
		}
		ZLUX_PROF("despeckle");

		// ── Thin-clutter declutter (wires / antennas) ───────────────────────
		// A real lens blurs a thin background occluder far from focus (a wire,
		// a TV antenna) until it vanishes -- a 1-3px wire covers a negligible
		// fraction of a 60px blur disc. Our gather only does that if the wire's
		// OWN CoC is large; AI / game depth maps give such structures a tiny CoC
		// (tagged near, or landing between focus and background), so the gather
		// there spans only a few px and the wire survives as a crisp sliver.
		//
		// Discriminator (purely geometric -- depth can't tell a mis-tagged wire
		// from a real subject): a wire is a THIN, UNDER-BLURRED MINORITY inside
		// an otherwise strongly-defocused neighbourhood. Two separable box
		// statistics over a 13x13 window decide it:
		//   frac  = fraction of the window that is blurred  (is it a minority?)
		//   relU  = 1 - |coc| / |mean blurred coc|          (am I much sharper
		//                                                     than my surrounds?)
		// Promote only when BOTH are high. This is what keeps it surgical:
		//   * a thin wire (focus-claiming OR small-CoC) -> frac≈1, relU≈1 -> goes
		//   * a SOLID near/focused region (wall, weapon body) -> its window is
		//     mostly itself, frac is low -> untouched (this is what the previous
		//     morphological-closing build got wrong: it filled every CoC valley,
		//     blurring the whole near layer and blocking the bokeh)
		//   * a region EDGE (≈50% blur) -> frac≈0.5 < gate -> untouched
		//   * an already-blurred pixel -> relU≈0 -> untouched
		// Near foreground (negative CoC) and Edge-Protect'd pixels are skipped.
		if (!local.no_depth && !zlux_depth_filters_on_gpu) {
			const size_t Nc = signed_coc_cache.size();
			const A_long boxR = 6;                      // 13x13 window
			const PF_FpLong box_area = static_cast<PF_FpLong>((2 * boxR + 1) * (2 * boxR + 1));
			std::vector<float> isb(Nc), cb(Nc);         // "is blurred", coc*isblurred
			for (size_t i = 0; i < Nc; ++i) {
				const float c = signed_coc_cache[i];
				const float b = (std::abs(c) >= 0.012f) ? 1.0f : 0.0f;
				isb[i] = b;
				cb[i] = c * b;
			}
			auto boxsum = [&](std::vector<float>& a) {
				std::vector<float> tmp(Nc);
				ParallelRows(out_h, 16, [&](A_long y0, A_long y1) {
					for (A_long y = y0; y < y1; ++y) {
						const size_t row = static_cast<size_t>(y) * out_w;
						for (A_long x = 0; x < out_w; ++x) {
							float s = 0.0f;
							for (A_long k = -boxR; k <= boxR; ++k)
								s += a[row + static_cast<size_t>(ClampValue<A_long>(x + k, 0, out_w - 1))];
							tmp[row + static_cast<size_t>(x)] = s;
						}
					}
				});
				ParallelRows(out_w, 16, [&](A_long x0, A_long x1) {
					for (A_long x = x0; x < x1; ++x) {
						for (A_long y = 0; y < out_h; ++y) {
							float s = 0.0f;
							for (A_long k = -boxR; k <= boxR; ++k)
								s += tmp[static_cast<size_t>(ClampValue<A_long>(y + k, 0, out_h - 1)) * out_w + static_cast<size_t>(x)];
							a[static_cast<size_t>(y) * out_w + static_cast<size_t>(x)] = s;
						}
					}
				});
			};
			boxsum(isb);   // isb[i] = count of blurred pixels in the window
			boxsum(cb);    // cb[i]  = sum of their signed CoC
			std::vector<float> declut(Nc);
			ParallelRows(out_h, 32, [&](A_long y0, A_long y1) {
				for (A_long y = y0; y < y1; ++y) {
					for (A_long x = 0; x < out_w; ++x) {
						const size_t pidx = static_cast<size_t>(y) * out_w + x;
						const PF_FpLong c = signed_coc_cache[pidx];
						if (c < 0.0 || protect_mask_cache[pidx] > 0.01) {
							declut[pidx] = static_cast<float>(c);
							continue;
						}
						const PF_FpLong nblur = isb[pidx];
						if (nblur < 1.0) { declut[pidx] = static_cast<float>(c); continue; }
						const PF_FpLong frac = nblur / box_area;
						const PF_FpLong mean_blur = cb[pidx] / nblur;       // far surround
						const PF_FpLong amean = std::abs(mean_blur);
						const PF_FpLong relU = (amean > 1e-4)
							? Clamp01(1.0 - std::abs(c) / amean) : 0.0;
						// Minority (frac) AND under-blurred-vs-surrounds (relU).
						// frac is the safety gate (a solid near/subject region is
						// not a minority, so it is never touched no matter how
						// aggressive relU is); relU is kept generous so a wire at
						// moderate CoC still dissolves fully toward the sky.
						const PF_FpLong t = SmoothStep(0.60, 0.85, frac) *
						                    SmoothStep(0.30, 0.65, relU);
						declut[pidx] = (t > 0.01)
							? static_cast<float>(Mix(c, mean_blur, t))
							: static_cast<float>(c);
					}
				}
			});
			signed_coc_cache.swap(declut);
		}
		ZLUX_PROF("thin declutter");
	}

	if (needs_bokeh) {
		if (!local.no_depth) {
			// Adaptive separable CoC-max dilation. The dilation radius must
			// cover the largest CoC found anywhere in the frame so that a
			// foreground pixel in tile T can still splash into a neighbour
			// that is up to max_CoC_px away. Without this, big near-field
			// blur halts at tile boundaries and produces hard ghost edges.
			PF_FpLong global_max_near = 0.0, global_max_far = 0.0;
			for (const auto& t : coc_tiles) {
				if (t.min_coc < 10.0) {
					global_max_near = std::max<PF_FpLong>(global_max_near, -t.min_coc);
					global_max_far  = std::max<PF_FpLong>(global_max_far,   t.max_coc);
				}
			}
			const PF_FpLong min_dim_px =
				static_cast<PF_FpLong>(std::min(out_w, out_h));
			const PF_FpLong max_reach_px =
				std::max(global_max_near, global_max_far) * min_dim_px * 0.15;
			const A_long dilate_r = ClampValue<A_long>(
				static_cast<A_long>(std::ceil(max_reach_px / static_cast<PF_FpLong>(kCocTileSize))) + 1,
				1, std::max(tiles_x, tiles_y));

			std::vector<CoCTileData> tmp(
				static_cast<size_t>(tiles_x * tiles_y),
				CoCTileData{ 10.0, -10.0, 10.0, -10.0 });
			coc_tiles_dilated.assign(
				static_cast<size_t>(tiles_x * tiles_y),
				CoCTileData{ 10.0, -10.0, 10.0, -10.0 });

			// Horizontal max-pass
			for (A_long ty = 0; ty < tiles_y; ++ty) {
				for (A_long tx = 0; tx < tiles_x; ++tx) {
					CoCTileData d{ 10.0, -10.0, 10.0, -10.0 };
					const A_long x0 = std::max<A_long>(0, tx - dilate_r);
					const A_long x1 = std::min<A_long>(tiles_x - 1, tx + dilate_r);
					for (A_long nx = x0; nx <= x1; ++nx) {
						const CoCTileData& t = coc_tiles[static_cast<size_t>(ty * tiles_x + nx)];
						d.min_coc = std::min(d.min_coc, t.min_coc);
						d.max_coc = std::max(d.max_coc, t.max_coc);
						d.min_depth = std::min(d.min_depth, t.min_depth);
						d.max_depth = std::max(d.max_depth, t.max_depth);
					}
					tmp[static_cast<size_t>(ty * tiles_x + tx)] = d;
				}
			}

			// Vertical max-pass
			for (A_long ty = 0; ty < tiles_y; ++ty) {
				for (A_long tx = 0; tx < tiles_x; ++tx) {
					CoCTileData d{ 10.0, -10.0, 10.0, -10.0 };
					const A_long y0 = std::max<A_long>(0, ty - dilate_r);
					const A_long y1 = std::min<A_long>(tiles_y - 1, ty + dilate_r);
					for (A_long ny = y0; ny <= y1; ++ny) {
						const CoCTileData& t = tmp[static_cast<size_t>(ny * tiles_x + tx)];
						d.min_coc = std::min(d.min_coc, t.min_coc);
						d.max_coc = std::max(d.max_coc, t.max_coc);
						d.min_depth = std::min(d.min_depth, t.min_depth);
						d.max_depth = std::max(d.max_depth, t.max_depth);
					}
					coc_tiles_dilated[static_cast<size_t>(ty * tiles_x + tx)] = d;
				}
			}
			// ── Provably-sharp tile early-out map ───────────────────────
			// A tile is skippable when nothing the per-pixel pass could do
			// there can deviate from the plain source pixel:
			//   * |raw CoC| <= 0.001 within ±2 tiles (32px) -- covers the
			//     pixel itself plus everything the bilateral refine, Depth
			//     Smoothing (<=24px) and boundary snap can mix in, and keeps
			//     focus_mask pinned at exactly 1 / sliver at exactly 0;
			//   * dilated near reach <= 0.001 -- no near splash can arrive
			//     (the dilation radius is sized to the frame's max CoC);
			//   * far CoC < 0.012 within ±4 tiles (64px) -- ProbeFarReachWide
			//     (±48px ladder) cannot trigger the far bleed-over.
			// Under these bounds the composite reduces algebraically to
			// src_lin (Mix(a,b,1)==b-exact form a+(b-a)*1 with a==b, and
			// Mix(a,_,0)==a), so the skip is bit-identical, not approximate.
			{
				const A_long tn = tiles_x * tiles_y;
				tile_skip.assign(static_cast<size_t>(tn), 0);
				std::vector<float> absmax(static_cast<size_t>(tn));
				std::vector<float> farmax(static_cast<size_t>(tn));
				std::vector<float> tmpf(static_cast<size_t>(tn));
				for (A_long i = 0; i < tn; ++i) {
					const CoCTileData& t = coc_tiles[static_cast<size_t>(i)];
					const bool valid = (t.min_coc < 10.0);
					absmax[static_cast<size_t>(i)] = valid
						? static_cast<float>(std::max(std::abs(t.min_coc), std::abs(t.max_coc)))
						: 0.0f;
					farmax[static_cast<size_t>(i)] = valid
						? static_cast<float>(std::max<PF_FpLong>(0.0, t.max_coc))
						: 0.0f;
				}
				auto maxfilt = [&](std::vector<float>& a, A_long radius) {
					for (A_long ty = 0; ty < tiles_y; ++ty) {
						for (A_long tx = 0; tx < tiles_x; ++tx) {
							float m = 0.0f;
							const A_long x0 = std::max<A_long>(0, tx - radius);
							const A_long x1 = std::min<A_long>(tiles_x - 1, tx + radius);
							for (A_long nx = x0; nx <= x1; ++nx) {
								m = std::max(m, a[static_cast<size_t>(ty * tiles_x + nx)]);
							}
							tmpf[static_cast<size_t>(ty * tiles_x + tx)] = m;
						}
					}
					for (A_long tx = 0; tx < tiles_x; ++tx) {
						for (A_long ty = 0; ty < tiles_y; ++ty) {
							float m = 0.0f;
							const A_long y0 = std::max<A_long>(0, ty - radius);
							const A_long y1 = std::min<A_long>(tiles_y - 1, ty + radius);
							for (A_long ny = y0; ny <= y1; ++ny) {
								m = std::max(m, tmpf[static_cast<size_t>(ny * tiles_x + tx)]);
							}
							a[static_cast<size_t>(ty * tiles_x + tx)] = m;
						}
					}
				};
				maxfilt(absmax, 2);
				maxfilt(farmax, 4);
				for (A_long i = 0; i < tn; ++i) {
					tile_skip[static_cast<size_t>(i)] =
						(absmax[static_cast<size_t>(i)] <= 0.001f &&
						 coc_tiles_dilated[static_cast<size_t>(i)].min_coc >= -0.001 &&
						 farmax[static_cast<size_t>(i)] < 0.012f) ? 1 : 0;
				}
			}
		} else {
			coc_tiles_dilated = coc_tiles;
		}

		vogel_luts_store.resize(static_cast<size_t>(kNumVogelLUTs));
		for (A_long li = 0; li < kNumVogelLUTs; ++li) {
			BuildVogelLUT(vogel_luts_store[static_cast<size_t>(li)], kVogelLUTSizes[li], local.bokeh_rotation_rad);
			FinalizeVogelLUT(vogel_luts_store[static_cast<size_t>(li)], local, apmap);
		}
	} else {
		coc_tiles_dilated.resize(static_cast<size_t>(tiles_x * tiles_y));
	}
	ZLUX_PROF("dilate+LUTs");

	// ── Distance-to-CoC-discontinuity field (silhouette-band accelerator) ────
	// GatherPass forces full-res colour taps in a band around every depth/CoC
	// silhouette (Frischluft parity). Detecting that band used to cost a 16-tap
	// scatter scan of the CoC cache PER PIXEL inside the gather -- a cache-miss
	// storm. Precompute it here instead: seeds are pixels straddling a CoC jump
	// (> kDiscThresh between 4-neighbours), then a two-pass chamfer distance
	// transform fills the (approximate Euclidean) distance to the nearest seed.
	// The gather compares this one read against the per-pixel reach. signed_coc_
	// cache is final at this point (tile dilation does not modify it).
	// Skipped when the CUDA path is active: zluxGpuBuildDiscDist regenerates this
	// field on the device from the CoC texture that is uploaded anyway, so the
	// sequential chamfer sweeps below (13.7 ms, unthreadable) are pure waste.
	// ZLUX_CPUDISC=1 forces the legacy CPU chamfer so the two implementations can
	// be A/B'd back to back under identical conditions.
	static const bool kForceCpuDisc = (std::getenv("ZLUX_CPUDISC") != nullptr);
	const bool disc_on_gpu = needs_bokeh && zlux_gpu::Enabled() && !kForceCpuDisc;
	if (needs_bokeh && !disc_on_gpu) {
		constexpr float kDiscThresh = 0.03f;
		const float kBig = static_cast<float>(out_w + out_h); // > any in-frame distance
		coc_disc_dist.assign(pixel_count, kBig);
		// Seed detection parallelizes cleanly (read-only on signed_coc_cache).
		ParallelRows(out_h, 16, [&](A_long y0, A_long y1) {
			for (A_long y = y0; y < y1; ++y) {
				for (A_long x = 0; x < out_w; ++x) {
					const size_t i = static_cast<size_t>(y) * out_w + x;
					const float c = signed_coc_cache[i];
					float md = 0.0f;
					if (x > 0)          md = std::max(md, std::abs(c - signed_coc_cache[i - 1]));
					if (x < out_w - 1)  md = std::max(md, std::abs(c - signed_coc_cache[i + 1]));
					if (y > 0)          md = std::max(md, std::abs(c - signed_coc_cache[i - out_w]));
					if (y < out_h - 1)  md = std::max(md, std::abs(c - signed_coc_cache[i + out_w]));
					if (md > kDiscThresh) coc_disc_dist[i] = 0.0f;
				}
			}
		});
		// Chamfer transform (orthogonal weight 1, diagonal ~sqrt(2)). Two raster
		// sweeps; inherently sequential but O(n) and trivial next to the gather.
		constexpr float d1 = 1.0f, d2 = 1.41421356f;
		auto relax = [](float& d, float other, float w) { const float nd = other + w; if (nd < d) d = nd; };
		for (A_long y = 0; y < out_h; ++y) {
			for (A_long x = 0; x < out_w; ++x) {
				const size_t i = static_cast<size_t>(y) * out_w + x;
				float d = coc_disc_dist[i];
				if (x > 0)                  relax(d, coc_disc_dist[i - 1], d1);
				if (y > 0)                  relax(d, coc_disc_dist[i - out_w], d1);
				if (x > 0 && y > 0)         relax(d, coc_disc_dist[i - out_w - 1], d2);
				if (x < out_w - 1 && y > 0) relax(d, coc_disc_dist[i - out_w + 1], d2);
				coc_disc_dist[i] = d;
			}
		}
		for (A_long y = out_h - 1; y >= 0; --y) {
			for (A_long x = out_w - 1; x >= 0; --x) {
				const size_t i = static_cast<size_t>(y) * out_w + x;
				float d = coc_disc_dist[i];
				if (x < out_w - 1)                  relax(d, coc_disc_dist[i + 1], d1);
				if (y < out_h - 1)                  relax(d, coc_disc_dist[i + out_w], d1);
				if (x < out_w - 1 && y < out_h - 1) relax(d, coc_disc_dist[i + out_w + 1], d2);
				if (x > 0 && y < out_h - 1)         relax(d, coc_disc_dist[i + out_w - 1], d2);
				coc_disc_dist[i] = d;
			}
		}
	}
	const float* coc_disc_dist_ptr = coc_disc_dist.empty() ? nullptr : coc_disc_dist.data();
	ZLUX_PROF("disc dist");

	// Build the linear-light mip pyramid of the source once per frame.
	// Gather sampling reads exclusively from the pyramid so we pay gamma
	// decode + int-to-float once per source pixel (here) instead of once
	// per Vogel tap. Only built when the bokeh path is active.
	SourcePyramid pyramid;
	if (needs_bokeh) {
		// Pass the full-res signed-CoC field so the downsample is EDGE-AWARE:
		// coarse mips near a silhouette keep the blurred side's colour instead of
		// a FG+BG average, which is what removes the dark edge halo at every blur
		// size (not just the small full-res band). needs_bokeh implies needs_coc,
		// so signed_coc_cache is populated here.
		BuildSourcePyramid(src_world, pyramid, local.highlight_recovery,
		                   signed_coc_cache.data(), out_w, out_h);
	}
	ZLUX_PROF("pyramid");

	// ── CUDA gather ──────────────────────────────────────────────────────────
	//
	// The gather measured at 94% of frame time and is bound by random access
	// (mip fetches + scattered CoC reads), which is precisely the workload a GPU
	// absorbs: hardware trilinear replaces the 8-load software filter, and
	// occupancy hides the latency an out-of-order window cannot.
	//
	// The three gathers RenderPixelImpl can issue per pixel -- far, near, and the
	// far bleed-over probe -- all run in ONE launch. Their radii are materialised
	// here into per-pixel arrays using the exact same expressions the CPU path
	// uses, so the composite downstream is untouched: it keeps its own gating and
	// merely reads a precomputed PassOutput instead of calling GatherPass.
	//
	// Anything the kernel does not implement (custom aperture texture, iris
	// modulator, astigmatism) is rejected by zluxGpuCanRender and silently falls
	// back to the CPU gather, so enabling a GPU never changes plugin output.
	// Timed so the panel badge can report what actually ran and how long the
	// gather took -- see zlux_gpu::RecordPath.
	const auto gather_t0 = std::chrono::steady_clock::now();
	// Owned by the CUDA context's pinned staging buffers, valid until the next
	// gather on the same context (which the device mutex serialises).
	// Borrowed from the CUDA context's pinned result pool; handed back at the
	// end of RenderCore by the guard below.
	const float4_gpu *gpu_far = nullptr, *gpu_near = nullptr,
	                 *gpu_bleed = nullptr, *gpu_matte = nullptr;
	int  gpu_slot = -1;
	bool gpu_active = false;
#ifdef ZLUX_CUDA
	// Returns the pooled set no matter how RenderCore exits; leaking one would
	// shrink the pool until every later frame fell back to the CPU.
	struct GpuResultGuard {
		int* slot;
		~GpuResultGuard() {
			// CtxIfAlive, not CtxLocked: releasing a slot must never resurrect a
			// context that Shutdown() just tore down (and we hold no device lock
			// here, so creating one would race).
			if (*slot >= 0) { zluxGpuReleaseResults(zlux_gpu::CtxIfAlive(), *slot); *slot = -1; }
		}
	} gpu_result_guard{&gpu_slot};
#endif
#ifdef ZLUX_CUDA
	if (needs_bokeh && pyramid.num_levels > 0 && zlux_gpu::Enabled()) {
		ZluxGatherParams gp{};
		gp.width = out_w; gp.height = out_h;
		gp.inv_w = static_cast<float>(inv_w); gp.inv_h = static_cast<float>(inv_h);
		gp.cache_w = out_w; gp.cache_h = out_h;
		gp.num_levels = pyramid.num_levels;
		gp.uniform_blur = local.no_depth ? 1 : 0;
		gp.energy_conserving = local.energy_conserving ? 1 : 0;
		gp.render_mode = local.render_mode;
		gp.sample_count = local.sample_count;
		gp.aperture_shape_mode = local.aperture_shape_mode;
		gp.aperture_map_index = local.aperture_map_index;
		gp.mask_angular = ((local.aperture_shape_mode == 3) ||
		                   (local.aperture_shape_mode == 2 && local.blade_curve < 0.15) ||
		                   (local.aperture_map_index > 0)) ? 1 : 0;
		gp.has_alpha = pyramid.has_alpha ? 1 : 0;
		gp.focal_distance = static_cast<float>(local.focal_distance);
		gp.anamorphic_ratio = static_cast<float>(local.anamorphic_ratio);
		gp.highlight_boost = static_cast<float>(local.highlight_boost);
		gp.bokeh_definition = static_cast<float>(local.bokeh_definition);
		gp.bokeh_gamma = static_cast<float>(local.bokeh_gamma);
		gp.highlight_scatter = static_cast<float>(local.highlight_scatter);
		gp.highlight_mode = local.highlight_mode;
		gp.highlights_low = static_cast<float>(local.highlights_low);
		gp.highlights_high = static_cast<float>(local.highlights_high);
		gp.highlights_softness = static_cast<float>(local.highlights_softness);
		gp.spherical_aberration_amount = static_cast<float>(local.spherical_aberration_amount);
		gp.spherical_aberration_scale = static_cast<float>(local.spherical_aberration_scale);
		gp.vignetting = static_cast<float>(local.vignetting);
		gp.vignetting_scale = static_cast<float>(local.vignetting_scale);
		gp.catadioptric = static_cast<float>(local.catadioptric);
		gp.ca_strength = static_cast<float>(local.ca_strength);
		gp.ca_rc = static_cast<float>(local.ca_rc);
		gp.ca_gm = static_cast<float>(local.ca_gm);
		gp.ca_by = static_cast<float>(local.ca_by);
		gp.near_blur_factor = static_cast<float>(local.near_blur_factor);
		gp.astigmatism = static_cast<float>(local.astigmatism);
		gp.astigmatism_sagittal = local.astigmatism_type_sagittal ? 1 : 0;
		gp.has_aperture_tex = (local.aperture_shape_mode == 4 && aperture_tex_world) ? 1 : 0;
		gp.has_iris_mod = (iris_mod_world && local.aperture_shape_mode != 4) ? 1 : 0;
		gp.aperture_texture_intensity = static_cast<float>(local.aperture_texture_intensity);
		gp.aperture_texture_scale = static_cast<float>(local.aperture_texture_scale);
		gp.aperture_texture_offset = static_cast<float>(local.aperture_texture_offset);
		gp.aperture_texture_invert = local.aperture_texture_invert ? 1 : 0;

		if (zluxGpuCanRender(&gp, aperture_tex_world ? 1 : 0, iris_mod_world ? 1 : 0)) {
			gpu_active = zlux_gpu::RunGather(gp, local, aperture_tex_world, iris_mod_world,
			                                 pyramid, vogel_luts_store,
			                                 signed_coc_cache, coc_disc_dist,
			                                 depth_cache, coc_tiles_dilated,
			                                 tiles_x, out_w, out_h, inv_w, inv_h,
			                                 gpu_far, gpu_near, gpu_bleed, gpu_matte, gpu_slot);
		}
	}
#endif // ZLUX_CUDA
	zlux_gpu::Trace(gpu_active ? "RC: gpu_active" : "RC: gpu inactive");
	ZLUX_PROF("cuda gather");

	// ── Half-resolution Far pre-pass ─────────────────────────────────────
	//
	// The Far layer is smooth by definition (pixels diverge only through
	// depth-of-field bokeh, never high-frequency scene detail), so rendering
	// it at half linear resolution and bilinear-upsampling is visually free
	// but cuts the per-pixel Far sample budget to 1/4. We parallelize the
	// pre-pass manually across all CPU cores because AE's iterate suite
	// cannot be invoked for a private buffer from inside RenderCore.
	// Combined with the main iterate (which now only runs the Near gather
	// for foreground bleed), this typically shaves 25-40% off 4K frames.
	std::vector<Color3> far_halfres;
	std::vector<float> far_alpha_halfres;
	std::vector<float> far_matte_halfres;
	std::vector<Color3> near_halfres;
	std::vector<float> near_alpha_halfres;
	std::vector<float> near_matte_halfres;
	A_long halfres_w = 0, halfres_h = 0;
	// Only FAST mode uses the half-res Far shortcut now. Final and Extreme
	// gather the Far layer per-pixel at FULL resolution (RenderPixelImpl falls
	// back to the direct GatherPass when rc.far_halfres is null) -- the
	// half-res buffer was the dominant source of blocky background bokeh under
	// inspection, and "Final" should mean final quality. Fast keeps it for
	// responsive scrubbing.
	if (needs_bokeh && !local.no_depth && pyramid.num_levels > 0 && local.render_mode == 1) {
		halfres_w = (out_w + 1) / 2;
		halfres_h = (out_h + 1) / 2;
		far_halfres.assign(static_cast<size_t>(halfres_w) * halfres_h,
		                   Color3{0.0, 0.0, 0.0});
		far_alpha_halfres.assign(static_cast<size_t>(halfres_w) * halfres_h, 0.0f);
		near_halfres.assign(static_cast<size_t>(halfres_w) * halfres_h,
		                    Color3{0.0, 0.0, 0.0});
		near_alpha_halfres.assign(static_cast<size_t>(halfres_w) * halfres_h, 0.0f);
		// Matte planes only when the layer has transparency -- otherwise the
		// full-res pass never reads them and we skip the memory + the alpha
		// accumulation in the half-res gathers.
		const bool hr_alpha = pyramid.has_alpha;
		if (hr_alpha) {
			far_matte_halfres.assign(static_cast<size_t>(halfres_w) * halfres_h, 1.0f);
			near_matte_halfres.assign(static_cast<size_t>(halfres_w) * halfres_h, 0.0f);
		}
		const PF_FpLong px_per_coc_pre = 0.15 / std::max(inv_w, inv_h);

		auto worker = [&](A_long y_start, A_long y_end) {
			for (A_long hy = y_start; hy < y_end; ++hy) {
				for (A_long hx = 0; hx < halfres_w; ++hx) {
					const A_long fx = std::min<A_long>(hx * 2, out_w - 1);
					const A_long fy = std::min<A_long>(hy * 2, out_h - 1);
					const size_t pidx = static_cast<size_t>(fy) * out_w + fx;
					const PF_FpLong depth_at     = depth_cache[pidx];
					const PF_FpLong signed_coc_at = signed_coc_cache[pidx];
					const PF_FpLong u = (static_cast<PF_FpLong>(fx) + 1.0) * inv_w;
					const PF_FpLong v = (static_cast<PF_FpLong>(fy) + 1.0) * inv_h;
					const size_t hidx = static_cast<size_t>(hy) * halfres_w + hx;
					// Provably-sharp tile: store the plain source tap; the
					// full-res pass short-circuits there anyway.
					const A_long ttx = fx / kCocTileSize;
					const A_long tty = fy / kCocTileSize;
					if (!tile_skip.empty() &&
					    tile_skip[static_cast<size_t>(tty) * tiles_x + ttx]) {
						far_halfres[hidx] = PercToLin(SampleMipLinear(pyramid.levels[0], u, v));
						if (hr_alpha) {
							far_matte_halfres[hidx] = static_cast<float>(
								SampleMipLinearCh(pyramid.levels[0], u, v, 3));
						}
						continue;
					}
					const PF_FpLong center_far =
						std::max<PF_FpLong>(0.0, signed_coc_at);
					// Near / focused pixels bordering the far field must not
					// leave the sharp source colour in this buffer: bilinear
					// sampling on the far side of a silhouette would smear it
					// into the blurred background (foreground-coloured
					// fringe), and the occlusion-sliver rescue in
					// RenderPixelImpl reads this buffer expecting a clean
					// background reconstruction. Gather with the tile's far
					// reach instead -- the Far gate rejects every non-far
					// tap, so the result is the background "seen around" the
					// occluder. Pixels with no far content nearby keep the
					// cheap source tap (their buffer texel is never blended
					// across a silhouette).
					PF_FpLong gather_r = center_far;
					if (center_far <= 0.001) {
						// Probed neighbour far CoC is both the trigger and the
						// reconstruction radius: per-pixel and spatially smooth
						// (the old dilated-tile max stepped at 16px tile
						// boundaries and showed as blocky bands). The wide
						// reach-gated ladder (same as the Final bleed-over
						// probe) also extends the reconstruction to every
						// pixel the far bleed can touch, so the stored
						// weight below doubles as the bleed alpha.
						const PF_FpLong nb_far = ProbeFarReachWide(
							signed_coc_cache.data(), out_w, out_h, fx, fy,
							px_per_coc_pre);
						if (nb_far > 0.012) gather_r = nb_far;
					} else {
						PF_FpLong sliver_far = 0.0;
						const PF_FpLong sliver = DetectCocSliver(
							signed_coc_cache.data(), out_w, out_h, fx, fy, signed_coc_at, &sliver_far);
						if (sliver > 0.01) gather_r = std::max(gather_r, sliver * sliver_far);
					}
					Color3 result = PercToLin(SampleMipLinear(pyramid.levels[0], u, v));
					// Far matte default = unblurred source coverage (matches the
					// plain source tap used when no far gather runs here).
					if (hr_alpha) {
						far_matte_halfres[hidx] = static_cast<float>(
							SampleMipLinearCh(pyramid.levels[0], u, v, 3));
					}
					if (gather_r > 0.001) {
						PassOutput f = GatherPass<DofPass::Far>(
							pyramid, aperture_tex_world, iris_mod_world,
							u, v, gather_r, local, depth_at, signed_coc_at,
							inv_w, inv_h,
							vogel_luts_store.data(), kNumVogelLUTs,
							signed_coc_cache.data(), out_w, out_h, coc_disc_dist_ptr);
						result = f.rgb;
						far_alpha_halfres[hidx] = static_cast<float>(f.weight);
						if (hr_alpha) far_matte_halfres[hidx] = static_cast<float>(f.matte);
					}
					far_halfres[hidx] = result;

					// Near layer at half res (rgb + coverage alpha). The
					// full-res pass bilinearly taps these instead of running
					// its own near gather -- in strongly near-blurred frames
					// the near gather dominates the whole frame cost, and
					// it is exactly as smooth as the far layer.
					const CoCTileData& ntile = coc_tiles_dilated[
						static_cast<size_t>(tty) * tiles_x + ttx];
					const PF_FpLong near_r = std::max(
						std::max<PF_FpLong>(0.0, -ntile.min_coc),
						std::max<PF_FpLong>(0.0, -signed_coc_at));
					if (near_r > 0.001) {
						PassOutput n = GatherPass<DofPass::Near>(
							pyramid, aperture_tex_world, iris_mod_world,
							u, v, near_r, local, depth_at, signed_coc_at,
							inv_w, inv_h,
							vogel_luts_store.data(), kNumVogelLUTs,
							signed_coc_cache.data(), out_w, out_h, coc_disc_dist_ptr);
						near_halfres[hidx] = n.rgb;
						near_alpha_halfres[hidx] = static_cast<float>(n.weight);
						if (hr_alpha) near_matte_halfres[hidx] = static_cast<float>(n.matte);
					}
				}
			}
		};

		const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
		const A_long num_threads = static_cast<A_long>(std::min<unsigned>(hw_threads, 32u));
		if (num_threads <= 1 || halfres_h < 4) {
			worker(0, halfres_h);
		} else {
			// Same crash-safety as ParallelRows: never let an exception escape a
			// worker (std::terminate), and if thread spawn fails mid-loop, join
			// the survivors and finish the remainder serially instead of letting
			// the joinable-thread vector terminate during unwinding.
			auto safe_worker = [&worker](A_long a, A_long b) {
				try { worker(a, b); } catch (...) {}
			};
			const A_long rows_per = (halfres_h + num_threads - 1) / num_threads;
			std::vector<std::thread> workers;
			workers.reserve(static_cast<size_t>(num_threads));
			A_long next = 0;
			try {
				for (A_long ti = 0; ti < num_threads; ++ti) {
					const A_long y_start = ti * rows_per;
					const A_long y_end = std::min<A_long>(halfres_h, (ti + 1) * rows_per);
					if (y_start >= y_end) break;
					workers.emplace_back(safe_worker, y_start, y_end);
					next = y_end;
				}
			} catch (...) {}
			for (auto& t : workers) t.join();
			if (next < halfres_h) safe_worker(next, halfres_h);
		}
	}

	ZLUX_PROF("halfres far");

	RenderRefcon rc{};
	rc.src_world = src_world;
	rc.depth_world = depth_world;
	rc.aperture_tex_world = aperture_tex_world;
	rc.iris_mod_world = iris_mod_world;
	rc.settings = &local;
	rc.apmap = apmap;
	rc.raw_depth_cache = raw_depth_cache.data();
	rc.depth_cache = depth_cache.data();
	rc.signed_coc_cache = signed_coc_cache.data();
	rc.coc_disc_dist = coc_disc_dist_ptr;
	rc.protect_mask_cache = protect_mask_cache.data();
	rc.coc_tiles_dilated = coc_tiles_dilated.data();
	rc.vogel_luts = vogel_luts_store.data();
	rc.num_vogel_luts = kNumVogelLUTs;
	rc.pyramid = &pyramid;
	// Hand the composite the GPU gather results. Null unless the kernel ran, in
	// which case every gather call site below falls back to GatherPass.
	if (gpu_active) {
		zlux_gpu::Trace("RC: wiring refcon");
		rc.gpu_far   = gpu_far;
		rc.gpu_near  = gpu_near;
		rc.gpu_bleed = gpu_bleed;
		rc.gpu_matte = gpu_matte;
	}
	rc.far_halfres = far_halfres.empty() ? nullptr : far_halfres.data();
	rc.far_alpha_halfres = far_alpha_halfres.empty() ? nullptr : far_alpha_halfres.data();
	rc.far_matte_halfres = far_matte_halfres.empty() ? nullptr : far_matte_halfres.data();
	rc.near_halfres = near_halfres.empty() ? nullptr : near_halfres.data();
	rc.near_alpha_halfres = near_alpha_halfres.empty() ? nullptr : near_alpha_halfres.data();
	rc.near_matte_halfres = near_matte_halfres.empty() ? nullptr : near_matte_halfres.data();
	rc.tile_skip = tile_skip.empty() ? nullptr : tile_skip.data();
	rc.halfres_w = halfres_w;
	rc.halfres_h = halfres_h;
	rc.inv_w = inv_w;
	rc.inv_h = inv_h;
	rc.out_w = out_w;
	rc.out_h = out_h;
	rc.tiles_x = tiles_x;

	// Final per-pixel pass, parallelized across all cores (see ParallelApply).
	// Replaces the single-threaded AE iterate suite that previously serialized
	// the whole compositing + near-gather stage on one core.
	zlux_gpu::Trace("RC: before ParallelApply", rc.gpu_far ? 1 : 0, rc.gpu_matte ? 1 : 0);
	if (is_float) {
		ParallelApply<PF_PixelFloat>(src_world, output, &rc);
	} else if (is_deep) {
		ParallelApply<PF_Pixel16>(src_world, output, &rc);
	} else {
		ParallelApply<PF_Pixel8>(src_world, output, &rc);
	}
	zlux_gpu::Trace("RC: after ParallelApply");
	// Record here, not at the gather stage: on the CPU path the gather happens
	// INSIDE ParallelApply, so an earlier measurement would report ~0 for it.
	// This span is the blur work in both cases, so the badge's numbers are
	// comparable between paths.
	if (needs_bokeh) {
		const float blur_ms = std::chrono::duration<float, std::milli>(
			std::chrono::steady_clock::now() - gather_t0).count();
		zlux_gpu::RecordPath(gpu_active ? zlux_gpu::Path::Gpu : zlux_gpu::Path::Cpu, blur_ms);
	}

	ZLUX_PROF("main apply");

	return err;
}

// Fills a buffer with the About-dialog body used both by the AE "About this
// Effect" menu item (PF_Cmd_ABOUT) and the in-panel About button.
static void FormatAboutMessage(AEGP_SuiteHandler& suites, char* out_buf)
{
	// out_buf is AE's return_msg (256 bytes incl. terminator) -- keep it tight.
	suites.ANSICallbacksSuite1()->sprintf(
		out_buf,
		"zluxDOF  v%d.%d.%d\r"
		"Photorealistic Depth of Field\r"
		"\r"
		"Vogel-spiral bokeh engine. Vintage glass soul:\r"
		"swirl, soap-bubble rims, onion rings, cat's-eye,\r"
		"chromatic fringing, halation & 80 real iris maps.\r"
		"\r"
		"\"Bokeh is not blur. Bokeh is character.\"\r"
		"\r"
		"crafted by zlux  \xb7  zluxia",
		MAJOR_VERSION,
		MINOR_VERSION,
		BUG_VERSION);
}

static PF_Err About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	FormatAboutMessage(suites, out_data->return_msg);
	return PF_Err_NONE;
}

static PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
	out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);
	// No PF_OutFlag_I_EXPAND_BUFFER: we are a strict 1:1 filter and never
	// write outside the source layer bounds. Keeping that flag forced AE
	// to validate our PreRender max_result_rect against an expanded buffer
	// shape, which breaks on adjustment layers (error 25::237). All gather
	// offsets clamp to edge during sampling, so no extra input context is
	// needed beyond what AE already guarantees.
	// No PF_OutFlag_PIX_INDEPENDENT either. It asserts that an output pixel does
	// not depend on the pixels around it, which lets AE leave the rows it does
	// not need (during field rendering) filled with garbage -- and zluxDOF is a
	// gather over a disc that can be tens of pixels wide, so it reads exactly
	// those rows. On an interlaced comp that garbage lands inside the bokeh. The
	// flag only ever bought a field-rendering optimisation we cannot use.
	out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
	                     PF_OutFlag_SEND_UPDATE_PARAMS_UI |
	                     PF_OutFlag_CUSTOM_UI;
	out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE |
	                       PF_OutFlag2_SUPPORTS_SMART_RENDER |
	                       PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

	// Kick off banner decode lazily; the first DRAW event will pick it up.
	// Doing it here (rather than on every effect instance) keeps load cost
	// to one GDI+ decode per process.
	zlux_banner::EnsureBannerLoaded();
	return PF_Err_NONE;
}

static PF_Err GlobalSetdown(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
	(void)in_data; (void)out_data; (void)params; (void)output;
	zlux_banner::ReleaseBanner();
#ifdef ZLUX_CUDA
	// Return the ~170 MB of device memory (and the pinned host staging) when AE
	// unloads the plug-in. Without this the CUDA context lived for the whole AE
	// session even while nothing used the effect, which is exactly the kind of
	// squatting that makes AE report "GPU out of memory" for unrelated effects.
	zlux_gpu::Shutdown();
#endif
	return PF_Err_NONE;
}

// ────────────────────────────────────────────────────────────────────────────
// Custom-UI banner: DRAW event
// ────────────────────────────────────────────────────────────────────────────
static void CopyAsciiToUtf16(const char* ascii, DRAWBOT_UTF16Char* out, size_t out_capacity)
{
	size_t i = 0;
	for (; ascii[i] != '\0' && (i + 1) < out_capacity; ++i) {
		out[i] = static_cast<DRAWBOT_UTF16Char>(static_cast<unsigned char>(ascii[i]));
	}
	out[i] = 0;
}

static PF_Err DrawBannerFallback(const DRAWBOT_Suites& d,
                                 DRAWBOT_SurfaceRef surface_ref,
                                 DRAWBOT_SupplierRef supplier_ref,
                                 const DRAWBOT_RectF32& rect)
{
	PF_Err err = PF_Err_NONE;
	DRAWBOT_BrushRef bg_brush = nullptr;
	DRAWBOT_BrushRef accent_brush = nullptr;
	DRAWBOT_BrushRef text_brush = nullptr;
	DRAWBOT_BrushRef sub_brush = nullptr;
	DRAWBOT_PathRef bg_path = nullptr;
	DRAWBOT_PathRef accent_path = nullptr;
	DRAWBOT_FontRef title_font = nullptr;
	DRAWBOT_FontRef sub_font = nullptr;

	DRAWBOT_ColorRGBA bg{0.08f, 0.10f, 0.14f, 1.0f};
	DRAWBOT_ColorRGBA accent{0.24f, 0.62f, 0.95f, 1.0f};
	DRAWBOT_ColorRGBA title_color{0.96f, 0.97f, 1.00f, 1.0f};
	DRAWBOT_ColorRGBA sub_color{0.62f, 0.70f, 0.82f, 1.0f};

	ERR(d.supplier_suiteP->NewBrush(supplier_ref, &bg, &bg_brush));
	ERR(d.supplier_suiteP->NewBrush(supplier_ref, &accent, &accent_brush));
	ERR(d.supplier_suiteP->NewBrush(supplier_ref, &title_color, &text_brush));
	ERR(d.supplier_suiteP->NewBrush(supplier_ref, &sub_color, &sub_brush));

	ERR(d.supplier_suiteP->NewPath(supplier_ref, &bg_path));
	ERR(d.path_suiteP->AddRect(bg_path, &rect));
	ERR(d.surface_suiteP->FillPath(surface_ref, bg_brush, bg_path, kDRAWBOT_FillType_Default));

	DRAWBOT_RectF32 accent_rect;
	accent_rect.left = rect.left;
	accent_rect.top = rect.top + rect.height - 3.0f;
	accent_rect.width = rect.width;
	accent_rect.height = 3.0f;
	ERR(d.supplier_suiteP->NewPath(supplier_ref, &accent_path));
	ERR(d.path_suiteP->AddRect(accent_path, &accent_rect));
	ERR(d.surface_suiteP->FillPath(surface_ref, accent_brush, accent_path, kDRAWBOT_FillType_Default));

	float default_size = 12.0f;
	ERR(d.supplier_suiteP->GetDefaultFontSize(supplier_ref, &default_size));
	ERR(d.supplier_suiteP->NewDefaultFont(supplier_ref, default_size * 2.0f, &title_font));
	ERR(d.supplier_suiteP->NewDefaultFont(supplier_ref, default_size, &sub_font));

	DRAWBOT_UTF16Char title_utf16[64];
	DRAWBOT_UTF16Char sub_utf16[128];
	CopyAsciiToUtf16("zluxDOF", title_utf16, sizeof(title_utf16) / sizeof(title_utf16[0]));
	char sub_buf[128];
#ifdef AE_OS_WIN
	_snprintf_s(sub_buf, sizeof(sub_buf), _TRUNCATE,
	            "Depth of Field  ·  by zlux · Zluxia  ·  v%d.%d",
	            MAJOR_VERSION, MINOR_VERSION);
#else
	snprintf(sub_buf, sizeof(sub_buf),
	         "Depth of Field  ·  by zlux · Zluxia  ·  v%d.%d",
	         MAJOR_VERSION, MINOR_VERSION);
#endif
	CopyAsciiToUtf16(sub_buf, sub_utf16, sizeof(sub_utf16) / sizeof(sub_utf16[0]));

	DRAWBOT_PointF32 title_origin;
	title_origin.x = rect.left + 14.0f;
	title_origin.y = rect.top + 32.0f;
	ERR(d.surface_suiteP->DrawString(surface_ref, text_brush, title_font, title_utf16,
	                                 &title_origin, kDRAWBOT_TextAlignment_Default,
	                                 kDRAWBOT_TextTruncation_None, 0.0f));

	DRAWBOT_PointF32 sub_origin;
	sub_origin.x = rect.left + 14.0f;
	sub_origin.y = rect.top + 52.0f;
	ERR(d.surface_suiteP->DrawString(surface_ref, sub_brush, sub_font, sub_utf16,
	                                 &sub_origin, kDRAWBOT_TextAlignment_Default,
	                                 kDRAWBOT_TextTruncation_None, 0.0f));

	if (title_font)  d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(title_font));
	if (sub_font)    d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(sub_font));
	if (bg_brush)    d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bg_brush));
	if (accent_brush) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(accent_brush));
	if (text_brush)  d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(text_brush));
	if (sub_brush)   d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(sub_brush));
	if (bg_path)     d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bg_path));
	if (accent_path) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(accent_path));
	return err;
}

// ── "GPU" / "CPU" badge in the banner's bottom-right corner ────────────────
//
// Whether the gather ran on the GPU was previously invisible, which is exactly
// how a session went by with the GPU path disabled while it "felt faster". This
// shows what the LAST render actually did, plus how long the blur took, so a
// silent fallback to the CPU (unsupported feature, no CUDA device, unhealthy
// device) reads as a state rather than as mysterious slowness.
static PF_Err DrawPathBadge(const DRAWBOT_Suites& d,
                            DRAWBOT_SurfaceRef surface_ref,
                            DRAWBOT_SupplierRef supplier_ref,
                            const DRAWBOT_RectF32& rect)
{
	PF_Err err = PF_Err_NONE;
	const int path = zlux_gpu::LastPath().load(std::memory_order_relaxed);
	if (path == static_cast<int>(zlux_gpu::Path::Unknown)) return err;   // nothing rendered yet

	const bool on_gpu = (path == static_cast<int>(zlux_gpu::Path::Gpu));
	const float ms = zlux_gpu::LastGatherMs().load(std::memory_order_relaxed);

	char label[64];
	std::snprintf(label, sizeof(label), "%s  %.0f ms", on_gpu ? "GPU" : "CPU", ms);

	DRAWBOT_UTF16Char text[64];
	CopyAsciiToUtf16(label, text, sizeof(text) / sizeof(text[0]));

	// Pill sized to the text; anchored bottom-right with a small inset.
	const float pad_x = 7.0f, pad_y = 3.0f, inset = 6.0f;
	const float text_w = 7.0f * static_cast<float>(std::strlen(label));
	const float bw = text_w + pad_x * 2.0f;
	const float bh = 15.0f;
	if (rect.width < bw + inset * 2.0f || rect.height < bh + inset * 2.0f) return err;

	DRAWBOT_RectF32 pill;
	pill.left   = rect.left + rect.width  - bw - inset;
	pill.top    = rect.top  + rect.height - bh - inset;
	pill.width  = bw;
	pill.height = bh;

	// Green when the GPU is doing the work, amber when it is not -- amber rather
	// than red because the CPU path is a correct, supported mode, not an error.
	DRAWBOT_ColorRGBA fill = on_gpu ? DRAWBOT_ColorRGBA{0.11f, 0.42f, 0.18f, 0.88f}
	                                : DRAWBOT_ColorRGBA{0.45f, 0.33f, 0.08f, 0.88f};
	DRAWBOT_ColorRGBA ink  = on_gpu ? DRAWBOT_ColorRGBA{0.68f, 1.00f, 0.74f, 1.0f}
	                                : DRAWBOT_ColorRGBA{1.00f, 0.88f, 0.60f, 1.0f};

	DRAWBOT_BrushRef bg_brush = nullptr, fg_brush = nullptr;
	DRAWBOT_PathRef  pill_path = nullptr;
	DRAWBOT_FontRef  font = nullptr;

	ERR(d.supplier_suiteP->NewBrush(supplier_ref, &fill, &bg_brush));
	ERR(d.supplier_suiteP->NewPath(supplier_ref, &pill_path));
	ERR(d.path_suiteP->AddRect(pill_path, &pill));
	ERR(d.surface_suiteP->FillPath(surface_ref, bg_brush, pill_path, kDRAWBOT_FillType_Default));

	ERR(d.supplier_suiteP->NewBrush(supplier_ref, &ink, &fg_brush));
	ERR(d.supplier_suiteP->NewDefaultFont(supplier_ref, 10.0f, &font));
	if (!err && fg_brush && font) {
		DRAWBOT_PointF32 origin{pill.left + pad_x, pill.top + bh - pad_y - 1.0f};
		ERR(d.surface_suiteP->DrawString(surface_ref, fg_brush, font, text,
		                                 &origin, kDRAWBOT_TextAlignment_Default,
		                                 kDRAWBOT_TextTruncation_None, 0.0f));
	}

	if (bg_brush)  d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bg_brush));
	if (fg_brush)  d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(fg_brush));
	if (pill_path) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(pill_path));
	if (font)      d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(font));
	return err;
}

static PF_Err DrawBannerImage(const DRAWBOT_Suites& d,
                              DRAWBOT_SurfaceRef surface_ref,
                              DRAWBOT_SupplierRef supplier_ref,
                              const DRAWBOT_RectF32& rect)
{
	PF_Err err = PF_Err_NONE;
	if (!zlux_banner::g_banner.load_ok ||
	    zlux_banner::g_banner.pixels_bgra.empty()) {
		return DrawBannerFallback(d, surface_ref, supplier_ref, rect);
	}

	DRAWBOT_ImageRef img_ref = nullptr;
	const int w = zlux_banner::g_banner.width;
	const int h = zlux_banner::g_banner.height;

	DRAWBOT_Boolean prefers_bgra = 0;
	d.supplier_suiteP->PrefersPixelLayoutBGRA(supplier_ref, &prefers_bgra);
	DRAWBOT_PixelLayout layout = prefers_bgra
	                                 ? kDRAWBOT_PixelLayout_32BGRA_Premul
	                                 : kDRAWBOT_PixelLayout_32ARGB_Premul;

	// Cover-fit: scale the image to FILL the control rect (no stretching, no
	// letterbox bars) and centre-crop the overflow axis. AE widens the custom
	// UI control with the panel, and the old letterbox left the banner
	// floating in a dark strip / looking broken at large widths. The crop is
	// done in the source buffer because Drawbot has no clip rect here and an
	// oversized blit would paint over neighbouring controls.
	const float scale_x = rect.width / static_cast<float>(w);
	const float scale_y = rect.height / static_cast<float>(h);
	const float scale = (scale_x > scale_y) ? scale_x : scale_y;
	int crop_w = static_cast<int>(std::ceil(rect.width / scale));
	int crop_h = static_cast<int>(std::ceil(rect.height / scale));
	crop_w = std::min(std::max(crop_w, 1), w);
	crop_h = std::min(std::max(crop_h, 1), h);
	const int sx0 = (w - crop_w) / 2;
	const int sy0 = (h - crop_h) / 2;

	std::vector<uint8_t> crop_buf(static_cast<size_t>(crop_w) * crop_h * 4);
	for (int yy = 0; yy < crop_h; ++yy) {
		const uint8_t* srow = &zlux_banner::g_banner.pixels_bgra[
			(static_cast<size_t>(sy0 + yy) * w + sx0) * 4];
		uint8_t* drow = &crop_buf[static_cast<size_t>(yy) * crop_w * 4];
		if (prefers_bgra) {
			for (int xx = 0; xx < crop_w * 4; ++xx) drow[xx] = srow[xx];
		} else {
			for (int xx = 0; xx < crop_w; ++xx) {
				drow[xx * 4 + 0] = srow[xx * 4 + 3]; // A
				drow[xx * 4 + 1] = srow[xx * 4 + 2]; // R
				drow[xx * 4 + 2] = srow[xx * 4 + 1]; // G
				drow[xx * 4 + 3] = srow[xx * 4 + 0]; // B
			}
		}
	}

	ERR(d.supplier_suiteP->NewImageFromBuffer(supplier_ref, crop_w, crop_h, crop_w * 4,
	                                          layout, crop_buf.data(), &img_ref));
	if (!err && img_ref) {
		DRAWBOT_PointF32 origin;
		origin.x = rect.left + (rect.width - crop_w * scale) * 0.5f;
		origin.y = rect.top + (rect.height - crop_h * scale) * 0.5f;

		// Background wash so transparent banner PNGs still sit on a neutral
		// dark strip, matching the AE panel chrome.
		DRAWBOT_BrushRef bg_brush = nullptr;
		DRAWBOT_PathRef bg_path = nullptr;
		DRAWBOT_ColorRGBA bg{0.09f, 0.11f, 0.14f, 1.0f};
		ERR(d.supplier_suiteP->NewBrush(supplier_ref, &bg, &bg_brush));
		ERR(d.supplier_suiteP->NewPath(supplier_ref, &bg_path));
		ERR(d.path_suiteP->AddRect(bg_path, &rect));
		ERR(d.surface_suiteP->FillPath(surface_ref, bg_brush, bg_path, kDRAWBOT_FillType_Default));
		if (bg_brush) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bg_brush));
		if (bg_path)  d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bg_path));

		// Scale + translate: DrawImage always blits at the image's native
		// pixel size, so we push a matrix that maps the (0..w, 0..h) image
		// space into our letterbox rect.
		DRAWBOT_MatrixF32 m{};
		m.mat[0][0] = scale;
		m.mat[1][1] = scale;
		m.mat[2][0] = origin.x;
		m.mat[2][1] = origin.y;

		ERR(d.surface_suiteP->PushStateStack(surface_ref));
		ERR(d.surface_suiteP->Transform(surface_ref, &m));
		DRAWBOT_PointF32 img_origin{0.0f, 0.0f};
		ERR(d.surface_suiteP->DrawImage(surface_ref, img_ref, &img_origin, 1.0f));
		ERR(d.surface_suiteP->PopStateStack(surface_ref));

		// ── GPU / CPU badge ────────────────────────────────────────────────
		// Reports the path the LAST render actually took, not the configured
		// one, so a silent CPU fallback is visible instead of just feeling
		// slow. Drawn after the image (hence the Pop above) so it is not
		// affected by the banner's scale matrix.
		DrawPathBadge(d, surface_ref, supplier_ref, rect);
	}

	if (img_ref) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(img_ref));
	return err;
}

// Renders the single-bokeh iris shape (active aperture map, blades, notch,
// catadioptric ring, spherical aberration, softness, matte box + a touch of
// chromatic fringe) into a BGRA buffer for the live panel preview. Mirrors the
// "Iris" display-mode math on a small fixed grid; field-edge behaviour
// (cat's-eye / swirl across the frame) is intentionally NOT duplicated here --
// the Preview Mode popup already offers Iris and Iris Array display modes for
// that.
static void RenderBokehThumb(const DOFSettings& s, uint8_t* bgra, int W, int H)
{
	// Own snapshot: this runs on the UI thread while renders are in flight, and
	// it used to share one mutable global with them.
	const zlux_apmap::ApMapRef apmap_ref = zlux_apmap::LoadApMap(s.aperture_map_index);
	const zlux_apmap::ApMap* apmap = apmap_ref.get();

	const bool is_poly = (s.aperture_shape_mode == 2 || s.aperture_shape_mode == 3);
	const bool has_apmap = (s.aperture_map_index > 0) && zlux_apmap::Active(apmap);
	const PF_FpLong anam = std::max<PF_FpLong>(0.1, s.anamorphic_ratio);
	const PF_FpLong edge_start = 1.0 - (0.02 + s.softness * 0.33);
	const PF_FpLong ba_rad = kTau / static_cast<PF_FpLong>(std::max<A_long>(3, s.aperture_blades));
	const PF_FpLong ns = (s.aperture_shape_mode == 3) ? std::max<PF_FpLong>(0.15, s.notch_scale) : s.notch_scale;
	const PF_FpLong nb = (s.notch_angle / kPi) * (ba_rad * 0.5);
	const PF_FpLong blade_shape = ClampValue<PF_FpLong>(s.blade_curve, -1.0, 1.0);
	// Negated vs the iris-display path: the preview scales the iris coords
	// (cp = p*rs), which inverts the channel-radius direction relative to the
	// gather (which offsets the SAMPLE position by *(1+rs)). Clamped so
	// extreme CA sliders can't blow a channel's iris past the viewport.
	const PF_FpLong r_s = 1.0 + ClampValue<PF_FpLong>(-( s.ca_rc       - 0.5 * s.ca_gm - 0.5 * s.ca_by) * 0.4, -0.45, 0.45);
	const PF_FpLong g_s = 1.0 + ClampValue<PF_FpLong>(-(-0.5 * s.ca_rc + s.ca_gm       - 0.5 * s.ca_by) * 0.4, -0.45, 0.45);
	const PF_FpLong b_s = 1.0 + ClampValue<PF_FpLong>(-(-0.5 * s.ca_rc - 0.5 * s.ca_gm + s.ca_by      ) * 0.4, -0.45, 0.45);

	auto iris = [&](const Vec2& p, PF_FpLong rs) -> PF_FpLong {
		const Vec2 cp{p.x * rs, p.y * rs};
		const PF_FpLong cd = Length(cp);
		if (cd >= 1.05) return 0.0;
		PF_FpLong m;
		if (is_poly) {
			m = GetPolygonalAperture(cp, s.aperture_blades, blade_shape, ns, nb);
			if (s.softness > 0.001) m *= 1.0 - SmoothStep(edge_start, 1.0, cd);
		} else {
			if (cd >= 1.0) return 0.0;
			m = 1.0 - SmoothStep(edge_start, 1.0, cd);
		}
		if (has_apmap) m *= zlux_apmap::Sample(apmap, cp.x, cp.y);
		if (s.onion_amount > 0.001)
			m *= OnionRingMask(ClampValue<PF_FpLong>(cd, 0.0, 1.0), s.onion_amount, s.onion_count);
		if (s.catadioptric > 0.1) m *= GetCatadioptricMask(cp, s.catadioptric);
		if (std::abs(s.spherical_aberration_amount) > 0.001)
			m *= ClampValue<PF_FpLong>(ComputeSphericalProfile(ClampValue<PF_FpLong>(cd, 0.0, 1.0), 1.0, s), 0.0, 2.8);
		m *= GetMatteBoxApertureMask(cp, s);
		return Clamp01(m);
	};

	auto mixc = [](PF_FpLong bg, PF_FpLong fg, PF_FpLong m) { return bg * (1.0 - m) + fg * m; };
	// Zoom the preview viewport out enough that the anamorphic-stretched iris
	// (which extends to ±1/anam horizontally) always fits with a margin --
	// otherwise wide ovals (anam < 1, e.g. NTSC 0.91) get clipped.
	const PF_FpLong view = std::max<PF_FpLong>(1.0, 1.0 / anam) * 1.18;
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			const PF_FpLong nx = (((x + 0.5) / W) * 2.0 - 1.0) * view;
			const PF_FpLong ny = (((y + 0.5) / H) * 2.0 - 1.0) * view;
			Vec2 p = Rotate({nx, ny}, s.bokeh_rotation_rad);
			p.x *= anam;
			const PF_FpLong rm = iris(p, r_s);
			const PF_FpLong gm = iris(p, g_s);
			const PF_FpLong bm = iris(p, b_s);
			uint8_t* px = &bgra[(static_cast<size_t>(y) * W + x) * 4];
			px[0] = static_cast<uint8_t>(ClampValue<PF_FpLong>(mixc(33, 255, bm), 0, 255));
			px[1] = static_cast<uint8_t>(ClampValue<PF_FpLong>(mixc(28, 245, gm), 0, 255));
			px[2] = static_cast<uint8_t>(ClampValue<PF_FpLong>(mixc(26, 245, rm), 0, 255));
			px[3] = 255;
		}
	}
}

// Decodes the iris-relevant params (matching the Render path's scaling) and
// blits a freshly rendered bokeh-shape thumbnail into the preview control.
static PF_Err DrawBokehPreview(const DRAWBOT_Suites& d,
                               DRAWBOT_SurfaceRef surface_ref,
                               DRAWBOT_SupplierRef supplier_ref,
                               const DRAWBOT_RectF32& rect,
                               PF_ParamDef* params[])
{
	PF_Err err = PF_Err_NONE;

	DOFSettings s{};
	s.aperture_shape_mode = params[ZLUXDOF_APERTURE_SHAPE]->u.pd.value;
	s.aperture_blades = ClampValue<A_long>(static_cast<A_long>(std::lround(params[ZLUXDOF_APERTURE_BLADES]->u.fs_d.value)), 3, 16);
	s.blade_curve = ClampValue<PF_FpLong>(params[ZLUXDOF_BLADE_CURVE]->u.fs_d.value * 0.01, -1.0, 1.0);
	s.notch_angle = params[ZLUXDOF_NOTCH_ANGLE]->u.fs_d.value * (kPi / 180.0);
	s.notch_scale = Clamp01(params[ZLUXDOF_NOTCH_SCALE]->u.fs_d.value * 0.01);
	s.softness = Clamp01(params[ZLUXDOF_SOFTNESS]->u.fs_d.value * 0.01);
	s.onion_amount = Clamp01(params[ZLUXDOF_ONION_RINGS]->u.fs_d.value * 0.01);
	s.onion_count = ClampValue<PF_FpLong>(params[ZLUXDOF_ONION_RING_COUNT]->u.fs_d.value, 3.0, 40.0);
	s.catadioptric = params[ZLUXDOF_CATADIOPTRIC_LENS]->u.bd.value
		? Clamp01(params[ZLUXDOF_CATADIOPTRIC_LENS_SCALE]->u.fs_d.value * 0.01) : 0.0;
	s.anamorphic_ratio = std::max<PF_FpLong>(0.1, ClampValue<PF_FpLong>(params[ZLUXDOF_ASPECT_RATIO]->u.fs_d.value, 0.0, 4.0));
	{ const PF_FpLong pr = AspectPresetRatio(params[ZLUXDOF_ASPECT_PRESET]->u.pd.value);
	  if (pr > 0.0) s.anamorphic_ratio = pr; }
	s.bokeh_rotation_rad = FIX_2_FLOAT(params[ZLUXDOF_BLADE_ANGLE]->u.ad.value) * (kPi / 180.0);
	s.spherical_aberration_amount = ClampValue<PF_FpLong>(params[ZLUXDOF_SPHERICAL_ABERRATION]->u.fs_d.value * 0.01, -1.0, 1.0);
	s.spherical_aberration_scale = Clamp01(params[ZLUXDOF_SPHERICAL_ABERRATION_SCALE]->u.fs_d.value * 0.01);
	s.matte_top = Clamp01(params[ZLUXDOF_MATTEBOX_TOP]->u.fs_d.value * 0.01);
	s.matte_bottom = Clamp01(params[ZLUXDOF_MATTEBOX_BOTTOM]->u.fs_d.value * 0.01);
	s.matte_left = Clamp01(params[ZLUXDOF_MATTEBOX_LEFT]->u.fs_d.value * 0.01);
	s.matte_right = Clamp01(params[ZLUXDOF_MATTEBOX_RIGHT]->u.fs_d.value * 0.01);
	s.aperture_map_index = ClampValue<A_long>(params[ZLUXDOF_APERTURE_MAP]->u.pd.value - 1, 0, 80);
	s.ca_rc = ClampValue<PF_FpLong>(params[ZLUXDOF_CA_RED_CYAN]->u.fs_d.value * 0.012, -1.2, 1.2);
	s.ca_gm = ClampValue<PF_FpLong>(params[ZLUXDOF_CA_GREEN_MAGENTA]->u.fs_d.value * 0.012, -1.2, 1.2);
	s.ca_by = ClampValue<PF_FpLong>(params[ZLUXDOF_CA_BLUE_YELLOW]->u.fs_d.value * 0.012, -1.2, 1.2);
	s.focal_distance = 0.5;

	// RenderBokehThumb takes its own snapshot; nothing to preload here.
	const int W = 200, H = 200;
	std::vector<uint8_t> buf(static_cast<size_t>(W) * H * 4);
	RenderBokehThumb(s, buf.data(), W, H);

	DRAWBOT_Boolean prefers_bgra = 0;
	d.supplier_suiteP->PrefersPixelLayoutBGRA(supplier_ref, &prefers_bgra);
	DRAWBOT_PixelLayout layout = prefers_bgra
		? kDRAWBOT_PixelLayout_32BGRA_Premul : kDRAWBOT_PixelLayout_32ARGB_Premul;
	if (!prefers_bgra) {
		for (size_t i = 0; i + 3 < buf.size(); i += 4) {
			const uint8_t B = buf[i], G = buf[i + 1], R = buf[i + 2], A = buf[i + 3];
			buf[i + 0] = A; buf[i + 1] = R; buf[i + 2] = G; buf[i + 3] = B;
		}
	}

	// Dark background wash, then the square thumbnail letterboxed in the rect.
	{
		DRAWBOT_BrushRef bgb = nullptr; DRAWBOT_PathRef bgp = nullptr;
		DRAWBOT_ColorRGBA bg{0.10f, 0.11f, 0.13f, 1.0f};
		ERR(d.supplier_suiteP->NewBrush(supplier_ref, &bg, &bgb));
		ERR(d.supplier_suiteP->NewPath(supplier_ref, &bgp));
		ERR(d.path_suiteP->AddRect(bgp, &rect));
		ERR(d.surface_suiteP->FillPath(surface_ref, bgb, bgp, kDRAWBOT_FillType_Default));
		if (bgb) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bgb));
		if (bgp) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bgp));
	}
	DRAWBOT_ImageRef img = nullptr;
	ERR(d.supplier_suiteP->NewImageFromBuffer(supplier_ref, W, H, W * 4, layout, buf.data(), &img));
	if (!err && img) {
		const float scale = (rect.width < rect.height ? rect.width : rect.height) / static_cast<float>(W);
		DRAWBOT_MatrixF32 m{};
		m.mat[0][0] = scale; m.mat[1][1] = scale;
		m.mat[2][0] = rect.left + (rect.width - W * scale) * 0.5f;
		m.mat[2][1] = rect.top + (rect.height - H * scale) * 0.5f;
		ERR(d.surface_suiteP->PushStateStack(surface_ref));
		ERR(d.surface_suiteP->Transform(surface_ref, &m));
		DRAWBOT_PointF32 o{0.0f, 0.0f};
		ERR(d.surface_suiteP->DrawImage(surface_ref, img, &o, 1.0f));
		ERR(d.surface_suiteP->PopStateStack(surface_ref));
	}
	if (img) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(img));
	return err;
}

// Draws the clickable aperture-map picker grid (the 80-shape montage) filling
// the control rect, plus a bright border around the currently-selected cell.
// Square-cell geometry for the aperture-map picker. AE hands the custom UI
// control the full panel width, and stretching the montage into it deformed
// every thumbnail into an oval. The grid instead keeps square cells sized by
// the limiting dimension and centres itself; the draw and the click handler
// both derive their mapping from this single function so they can never
// disagree.
struct ApMapPickerGrid { float left; float top; float cell; };
static ApMapPickerGrid ComputePickerGrid(float left, float top, float width, float height)
{
	const float cw = width / static_cast<float>(ZLUXDOF_APMAP_COLS);
	const float ch = height / static_cast<float>(ZLUXDOF_APMAP_ROWS);
	ApMapPickerGrid g;
	g.cell = (cw < ch) ? cw : ch;
	g.left = left + (width  - g.cell * ZLUXDOF_APMAP_COLS) * 0.5f;
	g.top  = top  + (height - g.cell * ZLUXDOF_APMAP_ROWS) * 0.5f;
	return g;
}

static PF_Err DrawPickerImage(const DRAWBOT_Suites& d,
                              DRAWBOT_SurfaceRef surface_ref,
                              DRAWBOT_SupplierRef supplier_ref,
                              const DRAWBOT_RectF32& rect,
                              int selected_index /* 0 = none, else 1..80 */)
{
	const ApMapPickerGrid grid = ComputePickerGrid(rect.left, rect.top, rect.width, rect.height);
	PF_Err err = PF_Err_NONE;

	// Dark background wash.
	{
		DRAWBOT_BrushRef bgb = nullptr; DRAWBOT_PathRef bgp = nullptr;
		DRAWBOT_ColorRGBA bg{0.10f, 0.11f, 0.13f, 1.0f};
		ERR(d.supplier_suiteP->NewBrush(supplier_ref, &bg, &bgb));
		ERR(d.supplier_suiteP->NewPath(supplier_ref, &bgp));
		ERR(d.path_suiteP->AddRect(bgp, &rect));
		ERR(d.surface_suiteP->FillPath(surface_ref, bgb, bgp, kDRAWBOT_FillType_Default));
		if (bgb) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bgb));
		if (bgp) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(bgp));
	}

	// Blit the montage into the square-cell grid area (clicks use the same
	// ComputePickerGrid mapping).
	if (zlux_apmap::g_picker.load_ok && !zlux_apmap::g_picker.pixels_bgra.empty()) {
		const int w = zlux_apmap::g_picker.width;
		const int h = zlux_apmap::g_picker.height;
		DRAWBOT_Boolean prefers_bgra = 0;
		d.supplier_suiteP->PrefersPixelLayoutBGRA(supplier_ref, &prefers_bgra);
		DRAWBOT_PixelLayout layout = prefers_bgra
			? kDRAWBOT_PixelLayout_32BGRA_Premul : kDRAWBOT_PixelLayout_32ARGB_Premul;
		std::vector<uint8_t> argb_copy;
		const void* src_buf = zlux_apmap::g_picker.pixels_bgra.data();
		if (!prefers_bgra) {
			argb_copy.resize(zlux_apmap::g_picker.pixels_bgra.size());
			for (size_t i = 0; i + 3 < argb_copy.size(); i += 4) {
				const uint8_t* s = &zlux_apmap::g_picker.pixels_bgra[i];
				argb_copy[i + 0] = s[3]; argb_copy[i + 1] = s[2];
				argb_copy[i + 2] = s[1]; argb_copy[i + 3] = s[0];
			}
			src_buf = argb_copy.data();
		}
		DRAWBOT_ImageRef img = nullptr;
		ERR(d.supplier_suiteP->NewImageFromBuffer(supplier_ref, w, h, w * 4, layout, src_buf, &img));
		if (!err && img) {
			DRAWBOT_MatrixF32 m{};
			m.mat[0][0] = grid.cell * ZLUXDOF_APMAP_COLS / static_cast<float>(w);
			m.mat[1][1] = grid.cell * ZLUXDOF_APMAP_ROWS / static_cast<float>(h);
			m.mat[2][0] = grid.left;
			m.mat[2][1] = grid.top;
			ERR(d.surface_suiteP->PushStateStack(surface_ref));
			ERR(d.surface_suiteP->Transform(surface_ref, &m));
			DRAWBOT_PointF32 o{0.0f, 0.0f};
			ERR(d.surface_suiteP->DrawImage(surface_ref, img, &o, 1.0f));
			ERR(d.surface_suiteP->PopStateStack(surface_ref));
		}
		if (img) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(img));
	}

	// Selection border (4 thin filled rects) around the active cell.
	const int n_cells = ZLUXDOF_APMAP_COLS * ZLUXDOF_APMAP_ROWS;
	if (selected_index >= 1 && selected_index <= n_cells) {
		const int idx0 = selected_index - 1;
		const float cw = grid.cell;
		const float ch = grid.cell;
		const float cx = grid.left + (idx0 % ZLUXDOF_APMAP_COLS) * cw;
		const float cy = grid.top  + (idx0 / ZLUXDOF_APMAP_COLS) * ch;
		const float t = 2.0f;
		DRAWBOT_ColorRGBA hi{0.30f, 0.72f, 1.0f, 1.0f};
		DRAWBOT_BrushRef hb = nullptr;
		ERR(d.supplier_suiteP->NewBrush(supplier_ref, &hi, &hb));
		auto fill = [&](float x, float y, float ww, float hh) {
			DRAWBOT_PathRef pp = nullptr; DRAWBOT_RectF32 rr{x, y, ww, hh};
			d.supplier_suiteP->NewPath(supplier_ref, &pp);
			d.path_suiteP->AddRect(pp, &rr);
			d.surface_suiteP->FillPath(surface_ref, hb, pp, kDRAWBOT_FillType_Default);
			if (pp) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(pp));
		};
		fill(cx, cy, cw, t);
		fill(cx, cy + ch - t, cw, t);
		fill(cx, cy, t, ch);
		fill(cx + cw - t, cy, t, ch);
		if (hb) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(hb));
	}
	return err;
}

// ── Depth Levels custom-UI control ──────────────────────────────────────────
// Photoshop-style mapping of the middle (gamma) handle: its normalized position
// p in [0,1] BETWEEN the black and white handles encodes Depth Gamma, with
// p = 0.5 <-> gamma 1.0 and a log-symmetric spread (p=0 -> 4.0, p=1 -> 0.25).
static inline PF_FpLong LevelsGammaToPos(PF_FpLong g)
{
	g = ClampValue<PF_FpLong>(g, 0.2, 4.0);
	return Clamp01(0.5 - 0.5 * (std::log(g) / std::log(4.0)));
}
static inline PF_FpLong LevelsPosToGamma(PF_FpLong p)
{
	p = Clamp01(p);
	return ClampValue<PF_FpLong>(std::pow(4.0, 1.0 - 2.0 * p), 0.2, 4.0);
}

// Geometry of the Levels control: a track plus TWO marker lanes -- gamma on top,
// black/white below -- so the gamma handle never hides the black/white ones, and
// every marker is clamped fully inside the control so none can vanish at the
// extremes. Shared by the drawer and the click hit-test so they cannot disagree.
struct LevelHandles {
	float hx0, hw, hist_h, base_y, hs;
	float gamma_cy, bw_cy;            // marker lane centre-y
	float bx_line, gx_line, wx_line;  // guide-line x at the TRUE value (clamped to track)
	float bx_m, gx_m, wx_m;           // marker centre x (on-canvas; black/white de-overlapped)
};
static LevelHandles ComputeLevelHandles(float left, float top, float width, float height,
                                        PF_FpLong black, PF_FpLong white, PF_FpLong gamma)
{
	LevelHandles h{};
	const float pad = 6.0f;
	h.hs = 11.0f;
	const float strip = h.hs * 2.0f + 4.0f;                  // two marker lanes + gap
	h.hx0 = left + pad;
	const float hy0 = top + pad;
	h.hw = std::max(1.0f, width - 2.0f * pad);
	h.hist_h = std::max(1.0f, height - 2.0f * pad - strip);
	h.base_y = hy0 + h.hist_h;
	h.gamma_cy = h.base_y + 2.0f + h.hs * 0.5f;                  // upper lane (gamma)
	h.bw_cy    = h.base_y + 2.0f + h.hs + 2.0f + h.hs * 0.5f;    // lower lane (black/white)

	const float lo = h.hx0, hi = h.hx0 + h.hw;
	h.bx_line = lo + static_cast<float>(Clamp01(black)) * h.hw;
	h.wx_line = lo + static_cast<float>(Clamp01(white)) * h.hw;
	h.gx_line = h.bx_line + static_cast<float>(LevelsGammaToPos(gamma)) * (h.wx_line - h.bx_line);

	// Marker centres clamped so the whole square stays inside the track.
	const float mlo = lo + h.hs * 0.5f, mhi = hi - h.hs * 0.5f;
	auto clampm = [&](float x) { return x < mlo ? mlo : (x > mhi ? mhi : x); };
	h.bx_m = clampm(h.bx_line);
	h.wx_m = clampm(h.wx_line);
	h.gx_m = clampm(h.gx_line);
	// De-overlap the black/white markers (same lane) so both stay grabbable even
	// when their values coincide.
	if (std::abs(h.wx_m - h.bx_m) < h.hs) {
		const float mid = 0.5f * (h.bx_m + h.wx_m);
		h.bx_m = clampm(mid - h.hs * 0.5f);
		h.wx_m = clampm(mid + h.hs * 0.5f);
	}
	return h;
}

// Draws the depth histogram (zlux_depthhist global) plus the black / gamma /
// white handles read live from the three sliders. Geometry comes from
// ComputeLevelHandles so the drawn markers and the click hit-test always agree.
static PF_Err DrawDepthLevels(const DRAWBOT_Suites& d,
                              DRAWBOT_SurfaceRef surface_ref,
                              DRAWBOT_SupplierRef supplier_ref,
                              const DRAWBOT_RectF32& rect,
                              PF_ParamDef* params[])
{
	PF_Err err = PF_Err_NONE;
	const PF_FpLong black = Clamp01(params[ZLUXDOF_DEPTH_BLACKPOINT]->u.fs_d.value);
	const PF_FpLong white = Clamp01(params[ZLUXDOF_DEPTH_WHITEPOINT]->u.fs_d.value);
	const PF_FpLong gamma = ClampValue<PF_FpLong>(params[ZLUXDOF_DEPTH_GAMMA]->u.fs_d.value, 0.2, 4.0);

	auto fill = [&](DRAWBOT_ColorRGBA c, float x, float y, float w, float hgt) {
		if (w <= 0.0f || hgt <= 0.0f) return;
		DRAWBOT_BrushRef b = nullptr; DRAWBOT_PathRef p = nullptr;
		DRAWBOT_RectF32 rr{ x, y, w, hgt };
		d.supplier_suiteP->NewBrush(supplier_ref, &c, &b);
		d.supplier_suiteP->NewPath(supplier_ref, &p);
		d.path_suiteP->AddRect(p, &rr);
		d.surface_suiteP->FillPath(surface_ref, b, p, kDRAWBOT_FillType_Default);
		if (p) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(p));
		if (b) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(b));
	};
	// A vertical guide line drawn with a contrasting "halo" casing behind the
	// core, so a near-black line reads on the dark track and a near-white line
	// reads on the light bars (style B: dark field, never-blending markers).
	auto guide = [&](DRAWBOT_ColorRGBA core, DRAWBOT_ColorRGBA casing, float cx, float y0, float hgt) {
		fill(casing, cx - 1.5f, y0, 3.0f, hgt);
		fill(core,   cx - 0.75f, y0, 1.5f, hgt);
	};
	// Photoshop-style triangle handle (apex up), filled + stroked outline so the
	// black handle stays visible against the dark strip via its light border.
	auto tri = [&](DRAWBOT_ColorRGBA face, DRAWBOT_ColorRGBA edge, float cx, float cy, float s) {
		const float half = s * 0.62f, up = s * 0.62f, dn = s * 0.45f;
		DRAWBOT_PathRef p = nullptr;
		d.supplier_suiteP->NewPath(supplier_ref, &p);
		d.path_suiteP->MoveTo(p, cx, cy - up);
		d.path_suiteP->LineTo(p, cx - half, cy + dn);
		d.path_suiteP->LineTo(p, cx + half, cy + dn);
		d.path_suiteP->Close(p);
		DRAWBOT_BrushRef b = nullptr;
		d.supplier_suiteP->NewBrush(supplier_ref, &face, &b);
		d.surface_suiteP->FillPath(surface_ref, b, p, kDRAWBOT_FillType_Default);
		DRAWBOT_PenRef pen = nullptr;
		d.supplier_suiteP->NewPen(supplier_ref, &edge, 1.2f, &pen);
		d.surface_suiteP->StrokePath(surface_ref, pen, p);
		if (pen) d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(pen));
		if (b)   d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(b));
		if (p)   d.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(p));
	};

	const LevelHandles h = ComputeLevelHandles(rect.left, rect.top, rect.width, rect.height, black, white, gamma);
	const float gy0 = rect.top + 6.0f;

	fill(DRAWBOT_ColorRGBA{ 0.039f, 0.043f, 0.051f, 1.0f }, rect.left, rect.top, rect.width, rect.height);
	fill(DRAWBOT_ColorRGBA{ 0.082f, 0.086f, 0.102f, 1.0f }, h.hx0, gy0, h.hw, h.hist_h);

	// Histogram bars (perceptual sqrt scaling so small near/far populations read).
	if (zlux_depthhist::g_valid) {
		const float invmax = 1.0f / static_cast<float>(std::max<A_long>(1, zlux_depthhist::g_max));
		const DRAWBOT_ColorRGBA barc{ 0.788f, 0.804f, 0.831f, 1.0f };
		for (int b = 0; b < zlux_depthhist::kBins; ++b) {
			const float bx0 = h.hx0 + (static_cast<float>(b) / zlux_depthhist::kBins) * h.hw;
			const float bw  = h.hw / static_cast<float>(zlux_depthhist::kBins);
			const float bh  = std::sqrt(static_cast<float>(zlux_depthhist::g_bins[b]) * invmax) * h.hist_h;
			if (bh < 0.5f) continue;
			fill(barc, bx0, h.base_y - bh, std::max(1.0f, bw), bh);
		}
	}

	// Vertical guide lines at the TRUE values, each with a contrasting casing.
	const DRAWBOT_ColorRGBA casing_lt{ 0.68f, 0.71f, 0.74f, 0.9f }; // light halo (for the black line)
	const DRAWBOT_ColorRGBA casing_dk{ 0.082f, 0.086f, 0.102f, 0.9f }; // dark halo (for gamma / white)
	guide(DRAWBOT_ColorRGBA{ 0.0f, 0.0f, 0.0f, 1.0f },     casing_lt, h.bx_line, gy0, h.hist_h);
	guide(DRAWBOT_ColorRGBA{ 0.60f, 0.63f, 0.66f, 1.0f },  casing_dk, h.gx_line, gy0, h.hist_h);
	guide(DRAWBOT_ColorRGBA{ 1.0f, 1.0f, 1.0f, 1.0f },     casing_dk, h.wx_line, gy0, h.hist_h);

	// Triangle handles: gamma in its own (upper) lane, black + white below.
	const DRAWBOT_ColorRGBA edge_lt{ 0.76f, 0.78f, 0.81f, 1.0f }; // light edge (dark black handle)
	const DRAWBOT_ColorRGBA edge_dk{ 0.039f, 0.043f, 0.051f, 1.0f }; // dark edge (light handles)
	tri(DRAWBOT_ColorRGBA{ 0.60f, 0.63f, 0.66f, 1.0f },  edge_dk, h.gx_m, h.gamma_cy, h.hs); // gamma (grey)
	tri(DRAWBOT_ColorRGBA{ 0.043f, 0.047f, 0.055f, 1.0f }, edge_lt, h.bx_m, h.bw_cy,   h.hs); // black + light edge
	tri(DRAWBOT_ColorRGBA{ 0.95f, 0.95f, 0.95f, 1.0f },  edge_dk, h.wx_m, h.bw_cy,    h.hs); // white
	return err;
}

static PF_Err DrawEvent(PF_InData* in_data, PF_OutData* out_data,
                        PF_ParamDef* params[], PF_LayerDef* output,
                        PF_EventExtra* event_extra)
{
	PF_Err err = PF_Err_NONE;
	if (event_extra->effect_win.area != PF_EA_CONTROL) return err;

	const A_long win_index = event_extra->effect_win.index;
	if (win_index != ZLUXDOF_BANNER && win_index != ZLUXDOF_APMAP_PICKER &&
	    win_index != ZLUXDOF_BOKEH_PREVIEW && win_index != ZLUXDOF_DEPTH_LEVELS) return err;

	if (win_index == ZLUXDOF_BANNER)            zlux_banner::EnsureBannerLoaded();
	else if (win_index == ZLUXDOF_APMAP_PICKER) zlux_apmap::EnsurePickerLoaded();
	// (the bokeh preview loads its aperture map inside DrawBokehPreview)

	DRAWBOT_Suites d{};
	ERR(AEFX_AcquireDrawbotSuites(in_data, out_data, &d));

	PF_EffectCustomUISuite1* custom_ui = nullptr;
	ERR(AEFX_AcquireSuite(in_data, out_data,
	                      kPFEffectCustomUISuite,
	                      kPFEffectCustomUISuiteVersion1,
	                      nullptr, (void**)&custom_ui));

	DRAWBOT_DrawRef drawing_ref = nullptr;
	if (!err && custom_ui) {
		ERR((*custom_ui->PF_GetDrawingReference)(event_extra->contextH, &drawing_ref));
		AEFX_ReleaseSuite(in_data, out_data,
		                  kPFEffectCustomUISuite,
		                  kPFEffectCustomUISuiteVersion1, nullptr);
	}

	DRAWBOT_SurfaceRef surface_ref = nullptr;
	DRAWBOT_SupplierRef supplier_ref = nullptr;
	if (!err && drawing_ref) {
		ERR(d.drawbot_suiteP->GetSupplier(drawing_ref, &supplier_ref));
		ERR(d.drawbot_suiteP->GetSurface(drawing_ref, &surface_ref));
	}

	if (!err && surface_ref && supplier_ref) {
		DRAWBOT_RectF32 r;
		r.left = event_extra->effect_win.current_frame.left + 0.5f;
		r.top = event_extra->effect_win.current_frame.top + 0.5f;
		r.width = static_cast<float>(event_extra->effect_win.current_frame.right -
		                             event_extra->effect_win.current_frame.left);
		r.height = static_cast<float>(event_extra->effect_win.current_frame.bottom -
		                              event_extra->effect_win.current_frame.top);
		if (win_index == ZLUXDOF_BANNER) {
			ERR(DrawBannerImage(d, surface_ref, supplier_ref, r));
		} else if (win_index == ZLUXDOF_APMAP_PICKER) {
			ERR(DrawPickerImage(d, surface_ref, supplier_ref, r,
			                    params[ZLUXDOF_APERTURE_MAP]->u.pd.value - 1));
		} else if (win_index == ZLUXDOF_DEPTH_LEVELS) {
			ERR(DrawDepthLevels(d, surface_ref, supplier_ref, r, params));
		} else {
			ERR(DrawBokehPreview(d, surface_ref, supplier_ref, r, params));
		}
	}

	AEFX_ReleaseDrawbotSuites(in_data, out_data);

	if (!err) {
		event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;
	}
	return err;
}

// Depth Levels drag state. Single UI thread, one drag at a time, so a file-scope
// int is sufficient. 0 = nothing grabbed, else which handle is being dragged.
enum { kLevNone = 0, kLevBlack = 1, kLevGamma = 2, kLevWhite = 3 };
static int g_levels_grab = kLevNone;
// Offset between the grabbed handle and the cursor at grab time, so a drag moves
// the handle RELATIVE to where it was picked up (no jump-to-cursor jolt -- the
// main "not smooth" complaint).
static PF_FpLong g_levels_grab_dx = 0.0;

// Force the custom-UI control to redraw immediately (PF_InvalidateRect +
// PF_EO_UPDATE_NOW) so the handle tracks the cursor smoothly, decoupled from the
// comp re-render the value change also queues (which is slow at large blur).
static void DepthLevelsRedrawNow(PF_InData* in_data, PF_OutData* out_data,
                                 PF_EventExtra* event_extra)
{
	PFAppSuite6* app = nullptr;
	if (AEFX_AcquireSuite(in_data, out_data, kPFAppSuite, kPFAppSuiteVersion6,
	                      nullptr, reinterpret_cast<void**>(&app)) == PF_Err_NONE && app) {
		(*app->PF_InvalidateRect)(event_extra->contextH, nullptr);
		AEFX_ReleaseSuite(in_data, out_data, kPFAppSuite, kPFAppSuiteVersion6, nullptr);
	}
	event_extra->evt_out_flags |= PF_EO_UPDATE_NOW;
}

// Maps an effective handle x (cursor + grab offset, screen space) to the grabbed
// handle's parameter and writes it. Geometry mirrors DrawDepthLevels (same pad).
// black/white stay ordered (each clamped off the other); gamma maps to its
// between-handles position.
static void DepthLevelsApply(PF_ParamDef* params[], const PF_UnionableRect& f,
                             PF_FpLong eff_x, int grab)
{
	const float pad = 6.0f;
	const float hx0 = static_cast<float>(f.left) + pad;
	const float hw  = static_cast<float>(f.right - f.left) - 2.0f * pad;
	if (hw <= 1.0f || grab == kLevNone) return;
	const PF_FpLong t = Clamp01((eff_x - hx0) / hw);
	const PF_FpLong black = Clamp01(params[ZLUXDOF_DEPTH_BLACKPOINT]->u.fs_d.value);
	const PF_FpLong white = Clamp01(params[ZLUXDOF_DEPTH_WHITEPOINT]->u.fs_d.value);
	if (grab == kLevBlack) {
		// Keep a small gap below white; max() guards against an inverted clamp
		// range when white sits at/near 0 (e.g. typed into the slider).
		const PF_FpLong bhi = std::max<PF_FpLong>(0.0, white - 0.02);
		params[ZLUXDOF_DEPTH_BLACKPOINT]->u.fs_d.value = ClampValue<PF_FpLong>(t, 0.0, bhi);
		params[ZLUXDOF_DEPTH_BLACKPOINT]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
	} else if (grab == kLevWhite) {
		const PF_FpLong wlo = std::min<PF_FpLong>(1.0, black + 0.02);
		params[ZLUXDOF_DEPTH_WHITEPOINT]->u.fs_d.value = ClampValue<PF_FpLong>(t, wlo, 1.0);
		params[ZLUXDOF_DEPTH_WHITEPOINT]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
	} else { // kLevGamma
		const PF_FpLong span = std::max<PF_FpLong>(1e-4, white - black);
		params[ZLUXDOF_DEPTH_GAMMA]->u.fs_d.value = LevelsPosToGamma((t - black) / span);
		params[ZLUXDOF_DEPTH_GAMMA]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
	}
}

// PF_Event_DRAG continuation: keep moving the handle grabbed on DO_CLICK.
static PF_Err DepthLevelsDoDrag(PF_InData* in_data, PF_OutData* out_data,
                                PF_ParamDef* params[], PF_LayerDef* output,
                                PF_EventExtra* event_extra)
{
	(void)output;
	if (event_extra->effect_win.area != PF_EA_CONTROL) return PF_Err_NONE;
	if (event_extra->effect_win.index != ZLUXDOF_DEPTH_LEVELS || g_levels_grab == kLevNone)
		return PF_Err_NONE;
	DepthLevelsApply(params, event_extra->effect_win.current_frame,
	                 static_cast<PF_FpLong>(event_extra->u.do_click.screen_point.h) + g_levels_grab_dx,
	                 g_levels_grab);
	event_extra->evt_out_flags |= PF_EO_HANDLED_EVENT;
	DepthLevelsRedrawNow(in_data, out_data, event_extra);
	return PF_Err_NONE;
}

static PF_Err BannerDoClick(PF_InData* in_data, PF_OutData* out_data,
                            PF_ParamDef* params[], PF_LayerDef* output,
                            PF_EventExtra* event_extra)
{
	(void)output;
	if (event_extra->effect_win.area != PF_EA_CONTROL) return PF_Err_NONE;
	const A_long win_index = event_extra->effect_win.index;

	if (win_index == ZLUXDOF_DEPTH_LEVELS) {
		// Grab the nearest handle in 2D (gamma sits in its own lane, so the click's
		// y disambiguates it from the black/white lane), pick it up IN PLACE
		// (offset by its true value position so the first drag step doesn't jolt),
		// and request drag events. The kept sliders are the precise fallback.
		const PF_UnionableRect& f = event_extra->effect_win.current_frame;
		const float fw = static_cast<float>(f.right - f.left);
		if (fw > 1.0f) {
			const PF_FpLong black = Clamp01(params[ZLUXDOF_DEPTH_BLACKPOINT]->u.fs_d.value);
			const PF_FpLong white = Clamp01(params[ZLUXDOF_DEPTH_WHITEPOINT]->u.fs_d.value);
			const PF_FpLong gamma = ClampValue<PF_FpLong>(params[ZLUXDOF_DEPTH_GAMMA]->u.fs_d.value, 0.2, 4.0);
			const LevelHandles h = ComputeLevelHandles(
				static_cast<float>(f.left), static_cast<float>(f.top),
				fw, static_cast<float>(f.bottom - f.top), black, white, gamma);
			const float cx = static_cast<float>(event_extra->u.do_click.screen_point.h);
			const float cy = static_cast<float>(event_extra->u.do_click.screen_point.v);
			auto d2 = [](float ax, float ay, float bx, float by) {
				const float dx = ax - bx, dy = ay - by; return dx * dx + dy * dy; };
			float best = d2(cx, cy, h.gx_m, h.gamma_cy); int grab = kLevGamma; PF_FpLong tline = h.gx_line;
			{ const float db = d2(cx, cy, h.bx_m, h.bw_cy); if (db < best) { best = db; grab = kLevBlack; tline = h.bx_line; } }
			{ const float dw = d2(cx, cy, h.wx_m, h.bw_cy); if (dw < best) { best = dw; grab = kLevWhite; tline = h.wx_line; } }
			g_levels_grab = grab;
			g_levels_grab_dx = static_cast<PF_FpLong>(tline) - cx; // grab in place (no jolt)
			event_extra->u.do_click.send_drag = TRUE;             // ask AE for PF_Event_DRAG
		}
		event_extra->evt_out_flags |= PF_EO_HANDLED_EVENT;
		DepthLevelsRedrawNow(in_data, out_data, event_extra);
		return PF_Err_NONE;
	}

	if (win_index == ZLUXDOF_BANNER) {
		AEGP_SuiteHandler suites(in_data->pica_basicP);
		FormatAboutMessage(suites, out_data->return_msg);
		out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
		event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;
		return PF_Err_NONE;
	}

	if (win_index == ZLUXDOF_APMAP_PICKER) {
		// Map the click position to a grid cell and write the Aperture Map
		// popup. screen_point and current_frame share the effect-window
		// coordinate space, so a plain subtract gives the in-control offset.
		const PF_UnionableRect& f = event_extra->effect_win.current_frame;
		const A_long fw = f.right - f.left;
		const A_long fh = f.bottom - f.top;
		if (fw > 0 && fh > 0) {
			// Same square-cell geometry the drawer uses; clicks in the dead
			// margins around the centred grid are ignored.
			const ApMapPickerGrid grid = ComputePickerGrid(
				static_cast<float>(f.left), static_cast<float>(f.top),
				static_cast<float>(fw), static_cast<float>(fh));
			const float lx = static_cast<float>(event_extra->u.do_click.screen_point.h) - grid.left;
			const float ly = static_cast<float>(event_extra->u.do_click.screen_point.v) - grid.top;
			if (grid.cell > 0.0f &&
			    lx >= 0.0f && ly >= 0.0f &&
			    lx < grid.cell * ZLUXDOF_APMAP_COLS &&
			    ly < grid.cell * ZLUXDOF_APMAP_ROWS) {
				const A_long col = ClampValue<A_long>(static_cast<A_long>(lx / grid.cell), 0, ZLUXDOF_APMAP_COLS - 1);
				const A_long row = ClampValue<A_long>(static_cast<A_long>(ly / grid.cell), 0, ZLUXDOF_APMAP_ROWS - 1);
				const A_long map_index = row * ZLUXDOF_APMAP_COLS + col + 1; // 1..80
				// Aperture Map popup: 1 = Off, 2 = Map 01 ... so value = index + 1.
				// Clicking the already-selected cell toggles the map OFF, so
				// deselecting never requires a trip to the popup.
				const A_long new_val = map_index + 1;
				params[ZLUXDOF_APERTURE_MAP]->u.pd.value =
					(params[ZLUXDOF_APERTURE_MAP]->u.pd.value == new_val) ? 1 : new_val;
				params[ZLUXDOF_APERTURE_MAP]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
			}
		}
		event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;
		return PF_Err_NONE;
	}

	return PF_Err_NONE;
}

static PF_Err HandleEvent(PF_InData* in_data, PF_OutData* out_data,
                          PF_ParamDef* params[], PF_LayerDef* output,
                          PF_EventExtra* extra)
{
	switch (extra->e_type) {
		case PF_Event_DRAW:
			return DrawEvent(in_data, out_data, params, output, extra);
		case PF_Event_DO_CLICK:
			return BannerDoClick(in_data, out_data, params, output, extra);
		case PF_Event_DRAG:
			return DepthLevelsDoDrag(in_data, out_data, params, output, extra);
		default:
			return PF_Err_NONE;
	}
}

static PF_Err UserChangedParam(PF_InData* in_data, PF_OutData* out_data,
                               PF_ParamDef* params[],
                               const PF_UserChangedParamExtra* which)
{
	if (!which) return PF_Err_NONE;
	if (which->param_index == ZLUXDOF_PRESET) {
		// User picked a lens preset -> bake its values into the live sliders
		// (editable in the panel) and the popup snaps back to Manual.
		const A_long pv = params[ZLUXDOF_PRESET]->u.pd.value;
		if (pv > 1) ApplyPresetToParams(pv, params);
	}
	return PF_Err_NONE;
}

static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
	PF_Err err = PF_Err_NONE;
	PF_ParamDef def;
	AEFX_CLR_STRUCT(def);

	// ── Banner ────────────────────────────────────────────────────────────
	// A zero-value color param promoted to a custom UI control draws the
	// zluxDOF brand strip at the top of the effect panel. The colour value
	// itself is never read — AE just needs a param slot to attach the
	// custom-draw event handler to. Flags match AE SDK Custom_ECW_UI sample
	// verbatim; adding CANNOT_TIME_VARY / CANNOT_INTERP here trips AE's
	// stream validation ("spatial interpolation method not allowed 29::15").
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	def.ui_flags = PF_PUI_CONTROL;
	def.ui_width = ZLUXDOF_BANNER_WIDTH;
	def.ui_height = ZLUXDOF_BANNER_HEIGHT;
	PF_ADD_COLOR(STR(StrID_Banner_Param_Name), 0, 0, 0, BANNER_DISK_ID);

	// ── Render Mode ───────────────────────────────────────────────────────
	// Global speed/quality governor: Fast (Preview) cuts the sample budget
	// for ~3x faster scrubbing, Final is the reference pipeline, Extreme
	// (Hero Shot) doubles the budget, relaxes the footprint tap cap and
	// gathers the Far layer per-pixel at full resolution for flawless
	// huge-CoC bokeh. Default: Final.
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(STR(StrID_Render_Mode), 3, 2, STR(StrID_Render_Mode_Choices), RENDER_MODE_DISK_ID);

	// ── Lens Preset ───────────────────────────────────────────────────────
	// "Manual" (index 1 in AE's 1-based popup values) is the default and
	// keeps every slider under the user's direct control. Any other choice
	// layers a baked lens-character curve on top during render — iris
	// geometry, softness, spherical / chromatic aberrations, optical
	// vignetting, catadioptric ring and astigmatism are frozen to the
	// preset. Sliders the preset doesn't touch (focus distance, blur
	// amount, sample quality, noise, …) continue to work normally.
	// SUPERVISE is required for AE to send PF_Cmd_USER_CHANGED_PARAM when the
	// popup changes -- that is what triggers the preset "bake into sliders".
	AEFX_CLR_STRUCT(def); def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(STR(StrID_Preset), 17, 1, STR(StrID_Preset_Choices), PRESET_DISK_ID);

	// ── Display ───────────────────────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_Display_Group), 0, DISPLAY_GROUP_START_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_POPUP(STR(StrID_Display_Mode), 8, 1, STR(StrID_Display_Mode_Choices), DISPLAY_MODE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(DISPLAY_GROUP_END_DISK_ID);

	// ── Depth & Focus ─────────────────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_Depth_Group), 0, DEPTH_GROUP_START_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_LAYER(STR(StrID_Depth_Layer), PF_LayerDefault_NONE, DEPTH_LAYER_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_POPUP(STR(StrID_Depth_Channel), 4, 1, STR(StrID_Depth_Channel_Choices), DEPTH_CHANNEL_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_CHECKBOXX(STR(StrID_Depth_Invert), 0, 0, DEPTH_INVERT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Depth_Focus), 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, PF_Precision_HUNDREDTHS, 0, 0, DEPTH_FOCUS_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_POINT(STR(StrID_Depth_SetFocus), 0, 0, FALSE, DEPTH_SET_FOCUS_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Depth_Curve), 0.0f, 4.0f, 0.0f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, DEPTH_CURVE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Depth_NearBlur), 0.0f, 200.0f, 0.0f, 200.0f, 0.0f, PF_Precision_INTEGER, 0, 0, DEPTH_NEAR_BLUR_DISK_ID);
	// Depth Levels: a custom-UI histogram of the depth map with draggable black /
	// gamma / white handles that drive the three sliders below. The colour value
	// itself is unused -- AE just needs a slot to attach the custom draw + click
	// handlers to (same idiom as the banner / bokeh preview / aperture picker).
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	def.ui_flags = PF_PUI_CONTROL;
	def.ui_width = ZLUXDOF_DEPTH_LEVELS_W;
	def.ui_height = ZLUXDOF_DEPTH_LEVELS_H;
	PF_ADD_COLOR(STR(StrID_Depth_Levels), 0, 0, 0, DEPTH_LEVELS_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Depth_Blackpoint), 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, DEPTH_BLACKPOINT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Depth_Whitepoint), 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, DEPTH_WHITEPOINT_DISK_ID);
	// Depth Gamma: remaps the focus falloff. 1 = linear; <1 pushes more depth
	// into the near side, >1 into the far side (DOF PRO-style depth remap).
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Depth_Gamma), 0.2f, 4.0f, 0.2f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, DEPTH_GAMMA_DISK_ID);
	// Depth Smoothing: box-blurs the signed CoC to suppress depth-map aliasing
	// / staircase banding at object silhouettes (0 = off).
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Depth_Smoothing), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_TENTHS, 0, 0, DEPTH_SMOOTHING_DISK_ID);
	// Edge Protect: at 0 the plugin behaves as if the slider didn't exist;
	// any non-zero value progressively attenuates near-field CoC on pixels
	// whose depth says "near" but whose source colour sits on a sharp edge
	// (wires, hair, antennas — anything the AI depth map misclassifies as
	// foreground). Higher = more aggressive bleed suppression.
	// Default is 0: the protection keeps thin near structures RAZOR-SHARP,
	// which reads as unnatural against a defocused background (a blurred
	// rooftop with crisp antennas/wires floating over it) compared with the
	// uniform, photoreal blur of DOF PRO / FL DoF. Out of the box we now let
	// every depth blur naturally; users hitting the AI-depth thin-detail
	// case (wires bokeh-smeared / green-fringed) can dial it up deliberately.
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Depth_ForegroundProtect), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_TENTHS, 0, 0, DEPTH_FOREGROUND_PROTECT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(DEPTH_GROUP_END_DISK_ID);

	// ── Aperture & Iris ───────────────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_Aperture_Group), PF_ParamFlag_START_COLLAPSED, APERTURE_GROUP_START_DISK_ID);
	// Live bokeh-shape preview: a custom-UI control that re-renders the current
	// iris/bokeh shape (blades, notch, aperture map, catadioptric, SA, softness,
	// matte, chromatic fringe) on every param change, so the user sees exactly
	// what the bokeh looks like without leaving Preview Mode = Rendered.
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	def.ui_flags = PF_PUI_CONTROL;
	def.ui_width = ZLUXDOF_BOKEH_PREVIEW_W;
	def.ui_height = ZLUXDOF_BOKEH_PREVIEW_H;
	PF_ADD_COLOR(STR(StrID_Bokeh_Preview), 0, 0, 0, BOKEH_PREVIEW_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_POPUP(STR(StrID_Aperture_Shape), 4, 1, STR(StrID_Aperture_Shape_Choices), APERTURE_SHAPE_DISK_ID);
	// Aperture Map (Library): "Off" + the 80 built-in shapes shipped in
	// aperture_lib/. The choices string is built here so we don't need an
	// 80-entry literal in the string table. When set to a shape it bakes that
	// grayscale iris mask into every bokeh (DOF PRO's aperture-map library).
	{
		char ap_choices[1024];
		strcpy(ap_choices, "Off");
		for (int i = 1; i <= 80; ++i) {
			char tmp[16];
			sprintf(tmp, "|Map %02d", i);
			strcat(ap_choices, tmp);
		}
		AEFX_CLR_STRUCT(def); PF_ADD_POPUP(STR(StrID_Aperture_Map), 81, 1, ap_choices, APERTURE_MAP_DISK_ID);
	}
	// Clickable picker grid: a custom-UI control (like the banner) that draws
	// the 80-shape montage and sets the Aperture Map popup on click. Starts
	// COLLAPSED so the big grid doesn't clutter the panel -- twirl it open to
	// browse. The colour value itself is unused.
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_START_COLLAPSED;
	def.ui_flags = PF_PUI_CONTROL;
	def.ui_width = ZLUXDOF_APMAP_PICKER_W;
	def.ui_height = ZLUXDOF_APMAP_PICKER_H;
	PF_ADD_COLOR(STR(StrID_ApMap_Picker), 0, 0, 0, APMAP_PICKER_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_LAYER(STR(StrID_Custom_Aperture), PF_LayerDefault_NONE, CUSTOM_APERTURE_LAYER_DISK_ID);
	// Aspect Preset: named anamorphic / broadcast ratios. "Custom" (1) leaves
	// the Aspect Ratio slider in control; any other entry overrides it.
	AEFX_CLR_STRUCT(def); PF_ADD_POPUP(STR(StrID_Aspect_Preset), 10, 1, STR(StrID_Aspect_Preset_Choices), ASPECT_PRESET_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Aspect_Ratio), 0.0f, 4.0f, 0.0f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, ASPECT_RATIO_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Aperture_Size), 0.0f, 100.0f, 0.0f, 100.0f, 15.0f, PF_Precision_HUNDREDTHS, 0, 0, APERTURE_SIZE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Sample_Count), 16.0f, 2048.0f, 16.0f, 2048.0f, 256.0f, PF_Precision_INTEGER, 0, 0, SAMPLE_COUNT_DISK_ID);
	// Bokeh Definition: how crisp a bokeh disc's edge is allowed to be. The
	// gather reads its colour from a MIP pyramid, and the footprint it reads at
	// is floored at a fraction of the blur radius; a point light is therefore
	// pre-blurred by that footprint before the aperture mask ever integrates it,
	// so the floor IS the disc's edge width. 0% = the v3.0 floor (35% of the
	// radius, creamy); 100% = no floor at all, leaving the Vogel inter-sample
	// spacing as the only limit (hard-edged, "bokeh balls"). Raise Sample Quality
	// alongside it -- the spacing is what caps definition once the floor is gone.
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Bokeh_Definition), 0.0f, 100.0f, 0.0f, 100.0f, 65.0f, PF_Precision_HUNDREDTHS, 0, 0, BOKEH_DEFINITION_DISK_ID);
	// Energy-Conserving (Physical) Bokeh: normalize the gather by geometric
	// coverage instead of the brightness-weighted sum -- a true linear-light
	// average that matches a real lens / 3D render and removes the firefly grain
	// at strong Brightness Boost. Punch then comes from Highlight Recovery (HDR) +
	// Scatter. Off = the DOF PRO-style punchy weighted gather (bit-identical).
	AEFX_CLR_STRUCT(def); PF_ADD_CHECKBOXX(STR(StrID_Energy_Conserving), 0, 0, ENERGY_CONSERVING_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Softness), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, SOFTNESS_DISK_ID);
	// Procedural onion rings (aspheric machining grooves inside the discs).
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Onion_Rings), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, ONION_RINGS_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Onion_Ring_Count), 3.0f, 40.0f, 3.0f, 40.0f, 14.0f, PF_Precision_INTEGER, 0, 0, ONION_RING_COUNT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Aperture_Blades), 3.0f, 16.0f, 3.0f, 16.0f, 6.0f, PF_Precision_INTEGER, 0, 0, APERTURE_BLADES_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_ANGLE(STR(StrID_Blade_Angle), 0.0f, BLADE_ANGLE_DISK_ID);
	// Bipolar blade curvature: −100 = concave (star-shape blades), 0 =
	// straight polygon, +100 = fully circular. Matches the DOF PRO
	// reference where negative curvature is essential for star / flower
	// iris silhouettes.
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Blade_Curve), -100.0f, 100.0f, -100.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, BLADE_CURVE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Notch_Angle), -180.0f, 180.0f, -180.0f, 180.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, NOTCH_ANGLE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Notch_Scale), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, NOTCH_SCALE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(APERTURE_GROUP_END_DISK_ID);

	// ── Lens Character ────────────────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_LensCharacter_Group), PF_ParamFlag_START_COLLAPSED, LENS_CHARACTER_GROUP_START_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Spherical_Aberration), -100.0f, 100.0f, -100.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, SPHERICAL_ABERRATION_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_CHECKBOXX(STR(StrID_Spherical_Aberration_Plus), 1, 0, SPHERICAL_ABERRATION_PLUS_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Spherical_Aberration_Scale), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, SPHERICAL_ABERRATION_SCALE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Optical_Vignetting), -100.0f, 100.0f, -100.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, OPTICAL_VIGNETTING_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Optical_Vignetting_Scale), 0.1f, 4.0f, 0.1f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, OPTICAL_VIGNETTING_SCALE_DISK_ID);
	// Range extended to 200 (v2.10): reference sweet-spot rigs need edge
	// elongation well past the old 100 ceiling.
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Astigmatism), 0.0f, 200.0f, 0.0f, 200.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, ASTIGMATISM_DISK_ID);
	// Field curvature: depth-independent sweet-spot edge blur (v2.10).
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Field_Curvature), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, FIELD_CURVATURE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Field_Sweet), 0.0f, 100.0f, 0.0f, 100.0f, 45.0f, PF_Precision_HUNDREDTHS, 0, 0, FIELD_SWEET_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_CHECKBOXX(STR(StrID_Catadioptric), 0, 0, CATADIOPTRIC_LENS_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Catadioptric_Scale), 0.0f, 100.0f, 0.0f, 100.0f, 30.0f, PF_Precision_HUNDREDTHS, 0, 0, CATADIOPTRIC_LENS_SCALE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(LENS_CHARACTER_GROUP_END_DISK_ID);

	// ── Chromatic Aberration ──────────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_CA_Group), PF_ParamFlag_START_COLLAPSED, CA_GROUP_START_DISK_ID);
	// Three independent opponent-axis fringe amounts (v2.12). Positive pushes
	// the named colour outward on far bokeh; negative flips the pair.
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_CA_RedCyan), -100.0f, 100.0f, -100.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, CA_RED_CYAN_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_CA_GreenMagenta), -100.0f, 100.0f, -100.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, CA_GREEN_MAGENTA_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_CA_BlueYellow), -100.0f, 100.0f, -100.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, CA_BLUE_YELLOW_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(CA_GROUP_END_DISK_ID);

	// ── Highlights / Bokeh Shaping ────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_Highlights_Group), PF_ParamFlag_START_COLLAPSED, HIGHLIGHTS_GROUP_START_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Highlights_Lower), 0.0f, 255.0f, 0.0f, 255.0f, 200.0f, PF_Precision_INTEGER, 0, 0, HIGHLIGHTS_LOWER_THRESHOLD_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Highlights_Upper), 0.0f, 255.0f, 0.0f, 255.0f, 255.0f, PF_Precision_INTEGER, 0, 0, HIGHLIGHTS_UPPER_THRESHOLD_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Highlights_Softness), 0.0f, 100.0f, 0.0f, 100.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, HIGHLIGHTS_SOFTNESS_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Highlights_Saturation), -100.0f, 100.0f, -100.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, HIGHLIGHTS_SATURATION_DISK_ID);
	// Range extended to 200 (v2.24): users want extra bokeh-brightness headroom.
	// The firefly grain that used to appear past ~100 is held down by the
	// boost-scaled colour-mip floor in the gather; Energy-Conserving mode removes
	// it entirely. Mapped *0.02 -> boost 0..4 (weight up to ~1+4*8 = 33x on peaks).
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Highlights_Enhancement), 0.0f, 200.0f, 0.0f, 200.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, HIGHLIGHTS_ENHANCEMENT_DISK_ID);
	// Bokeh Gamma drives the non-linear weighting of gather samples by
	// their luminance, which is how DOF PRO achieves its characteristic
	// crisp bokeh cores: a defocused specular highlight keeps its peak
	// brightness instead of being normalised away into a diffuse blob by
	// the unweighted average. 0 = linear (physical) gather; higher =
	// punchier cores, at the cost of slightly darker mid-range bokeh.
	// Tuned range 0..3 covers "subtle emphasis" through "HDR-feeling
	// cinematic bokeh" without quantization.
	// v2.21: default Bokeh Gamma 0.8 -> 1.4 (punchier specular cores). The big
	// anti-fog change is Highlight Recovery 0 -> 65 and a small Scatter 0 -> 15:
	// 8-bit footage clamps night lights / speculars to 1.0, so the energy-
	// conserving gather spreads them into dim grey "fog". Recovery extrapolates
	// those clipped pixels back above 1.0 (HDR) before the gather, so they
	// disperse into bright, clearly-edged bokeh instead of haze; Scatter adds a
	// gentle additive specular sprite for the crisp DOF-PRO core. Both only touch
	// pixels above the Highlights Lower Threshold, so non-clipped footage is
	// barely affected.
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Highlights_BokehGamma), 0.0f, 3.0f, 0.0f, 3.0f, 1.4f, PF_Precision_HUNDREDTHS, 0, 0, HIGHLIGHTS_BOKEH_GAMMA_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Highlights_Scatter), 0.0f, 100.0f, 0.0f, 100.0f, 15.0f, PF_Precision_HUNDREDTHS, 0, 0, HIGHLIGHTS_SCATTER_DISK_ID);
	// How Highlight Scatter combines with the gather.
	//   Additive     -- the historical behaviour: specular taps accumulate into
	//                   an un-normalised bucket that is layered on top, so a
	//                   bright point can push the result past the source's own
	//                   peak and clip to flat white.
	//   Preservative -- the specular emphasis is folded into the gather WEIGHTS
	//                   and renormalised, so the result stays a convex
	//                   combination of the sampled colours. It can never exceed
	//                   the brightest sample, exposure and dynamic range are
	//                   preserved, and the highlight is concentrated by
	//                   redistributing energy rather than inventing it.
	// v3.1: Preservative is the DEFAULT. Additive shipped as the default only
	// because it was the historical behaviour; it invents energy, and paired with
	// the crisper gather this release brings that reads as clipped white cores
	// where Preservative keeps the disc's colour and falloff.
	AEFX_CLR_STRUCT(def); PF_ADD_POPUP(STR(StrID_Highlights_Mode), 2, 2, STR(StrID_Highlights_Mode_Choices), HIGHLIGHTS_MODE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Highlights_Recovery), 0.0f, 100.0f, 0.0f, 100.0f, 65.0f, PF_Precision_HUNDREDTHS, 0, 0, HIGHLIGHTS_RECOVERY_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_COLOR(STR(StrID_Highlights_Tint), PF_MAX_CHAN8, PF_MAX_CHAN8, PF_MAX_CHAN8, HIGHLIGHTS_TINT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(HIGHLIGHTS_GROUP_END_DISK_ID);

	// ── Iris Texture ──────────────────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_ApTex_Group), PF_ParamFlag_START_COLLAPSED, APTEX_GROUP_START_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_LAYER(STR(StrID_ApTex_Layer), PF_LayerDefault_NONE, APTEX_LAYER_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_CHECKBOXX(STR(StrID_ApTex_Invert), 0, 0, APTEX_INVERT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_ApTex_Intensity), 0.0f, 100.0f, 0.0f, 100.0f, 100.0f, PF_Precision_HUNDREDTHS, 0, 0, APTEX_INTENSITY_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_ApTex_Scale), 0.0f, 200.0f, 0.0f, 200.0f, 100.0f, PF_Precision_HUNDREDTHS, 0, 0, APTEX_SCALE_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_ApTex_Offset), -100.0f, 100.0f, -100.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, APTEX_OFFSET_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(APTEX_GROUP_END_DISK_ID);

	// ── Matte Box ─────────────────────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_MatteBox_Group), PF_ParamFlag_START_COLLAPSED, MATTEBOX_GROUP_START_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_MatteBox_Top), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, MATTEBOX_TOP_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_MatteBox_Bottom), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, MATTEBOX_BOTTOM_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_MatteBox_Left), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, MATTEBOX_LEFT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_MatteBox_Right), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, MATTEBOX_RIGHT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(MATTEBOX_GROUP_END_DISK_ID);

	// ── Grain / Noise ─────────────────────────────────────────────────────
	AEFX_CLR_STRUCT(def); PF_ADD_TOPICX(STR(StrID_Noise_Group), PF_ParamFlag_START_COLLAPSED, NOISE_GROUP_START_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_FLOAT_SLIDERX(STR(StrID_Noise_Amount), 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_HUNDREDTHS, 0, 0, NOISE_AMOUNT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_CHECKBOXX(STR(StrID_Noise_Animated), 1, 0, NOISE_ANIMATED_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_CHECKBOXX(STR(StrID_Noise_Monochromatic), 1, 0, NOISE_MONOCHROMATIC_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_POPUP(STR(StrID_Noise_LumaDistribution), 2, 1, STR(StrID_Noise_LumaDistribution_Choices), NOISE_LUMA_DISTRIBUTION_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_POPUP(STR(StrID_Noise_MapDistribution), 3, 1, STR(StrID_Noise_MapDistribution_Choices), NOISE_MAP_DISTRIBUTION_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_ADD_COLOR(STR(StrID_Noise_Tint), PF_MAX_CHAN8, PF_MAX_CHAN8, PF_MAX_CHAN8, NOISE_TINT_DISK_ID);
	AEFX_CLR_STRUCT(def); PF_END_TOPIC(NOISE_GROUP_END_DISK_ID);

	if (!err) {
		PF_CustomUIInfo ci;
		AEFX_CLR_STRUCT(ci);
		ci.events = PF_CustomEFlag_EFFECT;
		ci.comp_ui_width = 0;
		ci.comp_ui_height = 0;
		ci.comp_ui_alignment = PF_UIAlignment_NONE;
		ci.layer_ui_width = 0;
		ci.layer_ui_height = 0;
		ci.layer_ui_alignment = PF_UIAlignment_NONE;
		ci.preview_ui_width = 0;
		ci.preview_ui_height = 0;
		err = (*(in_data->inter.register_ui))(in_data->effect_ref, &ci);
	}

	out_data->num_params = ZLUXDOF_NUM_PARAMS;
	return err;
}

static PF_Err UpdateParamsUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	const A_long shape = params[ZLUXDOF_APERTURE_SHAPE]->u.pd.value;
	const PF_Boolean is_notched = (shape == 3);
	const PF_Boolean is_custom = (shape == 4);
	const PF_Boolean blades_on = (shape == 2 || shape == 3);
	const PF_Boolean cata_on = params[ZLUXDOF_CATADIOPTRIC_LENS]->u.bd.value;
	const PF_Boolean spher_on = (std::abs(params[ZLUXDOF_SPHERICAL_ABERRATION]->u.fs_d.value) > 0.01);
	const PF_Boolean astig_on = (params[ZLUXDOF_ASTIGMATISM]->u.fs_d.value > 0.01);
	// A named Aspect Preset (anything but "Custom" = index 1) drives the ratio,
	// so grey out the manual Aspect Ratio slider.
	const PF_Boolean aspect_preset_on = (params[ZLUXDOF_ASPECT_PRESET]->u.pd.value != 1);

	// Context-aware greying. IMPORTANT: only read slider / popup / checkbox
	// VALUES here -- a layer param's u.ld.data is NOT populated during
	// UPDATE_PARAMS_UI, so probing it (e.g. "is a depth map assigned?") returns
	// null even when a layer IS set and would wrongly grey out whole groups.
	// Every rule below is value-based and reliable.
	const PF_Boolean vig_on = (std::abs(params[ZLUXDOF_OPTICAL_VIGNETTING]->u.fs_d.value) > 0.01);
	const PF_Boolean grain_on = (params[ZLUXDOF_NOISE_AMOUNT]->u.fs_d.value > 0.01);
	const PF_Boolean onion_on = (params[ZLUXDOF_ONION_RINGS]->u.fs_d.value > 0.01);
	const PF_Boolean fieldc_on = (params[ZLUXDOF_FIELD_CURVATURE]->u.fs_d.value > 0.01);

	const struct { PF_ParamIndex idx; PF_Boolean disabled; } rules[] = {
		// Aperture / iris geometry.
		{ZLUXDOF_ASPECT_RATIO,                  aspect_preset_on},
		{ZLUXDOF_CUSTOM_APERTURE_LAYER,         !is_custom},
		{ZLUXDOF_APERTURE_BLADES,               !blades_on},
		{ZLUXDOF_BLADE_ANGLE,                   !blades_on},
		{ZLUXDOF_BLADE_CURVE,                   !blades_on},
		{ZLUXDOF_NOTCH_ANGLE,                   !is_notched},
		{ZLUXDOF_NOTCH_SCALE,                   !is_notched},
		{ZLUXDOF_ONION_RING_COUNT,              !onion_on},
		{ZLUXDOF_FIELD_SWEET,                   !fieldc_on},
		// Lens character.
		{ZLUXDOF_CATADIOPTRIC_LENS_SCALE,       !cata_on},
		{ZLUXDOF_SPHERICAL_ABERRATION_PLUS,     !astig_on},
		{ZLUXDOF_SPHERICAL_ABERRATION_SCALE,    !spher_on},
		{ZLUXDOF_OPTICAL_VIGNETTING_SCALE,      !vig_on},
		// Grain -- sub-controls only matter once there is grain.
		{ZLUXDOF_NOISE_ANIMATED,                !grain_on},
		{ZLUXDOF_NOISE_MONOCHROMATIC,           !grain_on},
		{ZLUXDOF_NOISE_LUMA_DISTRIBUTION,       !grain_on},
		{ZLUXDOF_NOISE_MAP_DISTRIBUTION,        !grain_on},
		{ZLUXDOF_NOISE_TINT,                    !grain_on},
	};

	for (const auto& r : rules) {
		PF_ParamDef p;
		AEFX_CLR_STRUCT(p);
		p = *params[r.idx];
		if (r.disabled) {
			p.ui_flags |= PF_PUI_DISABLED;
		} else {
			p.ui_flags &= ~PF_PUI_DISABLED;
		}
		ERR(suites.ParamUtilsSuite3()->PF_UpdateParamUI(in_data->effect_ref, r.idx, &p));
	}

	return err;
}

static PF_Err Render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
	PF_Err err = PF_Err_NONE;
	PF_ParamDef depth_param;
	PF_ParamDef custom_aperture_param;
	PF_ParamDef iris_mod_param;
	AEFX_CLR_STRUCT(depth_param);
	AEFX_CLR_STRUCT(custom_aperture_param);
	AEFX_CLR_STRUCT(iris_mod_param);

	ERR(PF_CHECKOUT_PARAM(in_data, ZLUXDOF_DEPTH_LAYER, in_data->current_time, in_data->time_step, in_data->time_scale, &depth_param));
	if (err) {
		return err;
	}
	ERR(PF_CHECKOUT_PARAM(in_data, ZLUXDOF_CUSTOM_APERTURE_LAYER, in_data->current_time, in_data->time_step, in_data->time_scale, &custom_aperture_param));
	if (err) {
		PF_CHECKIN_PARAM(in_data, &depth_param);
		return err;
	}
	// Iris Texture modulator layer. Checkout is best-effort -- a missing
	// layer just means the feature is off, which is the default case.
	PF_CHECKOUT_PARAM(in_data, ZLUXDOF_APTEX_LAYER, in_data->current_time, in_data->time_step, in_data->time_scale, &iris_mod_param);

	PF_EffectWorld* depth_world = &depth_param.u.ld;
	PF_EffectWorld* aperture_tex_world = nullptr;
	if (custom_aperture_param.u.ld.data) {
		aperture_tex_world = &custom_aperture_param.u.ld;
	}
	PF_EffectWorld* iris_mod_world = nullptr;
	if (iris_mod_param.u.ld.data) {
		iris_mod_world = &iris_mod_param.u.ld;
	}
	const PF_Boolean has_depth = (depth_world->data != nullptr);

	DOFSettings s{};
	s.display_mode = params[ZLUXDOF_DISPLAY_MODE]->u.pd.value;
	s.depth_channel = params[ZLUXDOF_DEPTH_CHANNEL]->u.pd.value;
	s.focal_distance = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_DEPTH_FOCUS]), 0.0, 1.0);
	const PF_FpLong aperture_size_raw = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_APERTURE_SIZE]), 0.0, 100.0);
	s.blur_strength = aperture_size_raw * 0.01;
	s.focus_range = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_DEPTH_CURVE]) * 0.04, 0.0, 0.25);
	s.anamorphic_ratio = std::max<PF_FpLong>(0.1, ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_ASPECT_RATIO]), 0.0, 4.0));
	{ const PF_FpLong preset_ratio = AspectPresetRatio(params[ZLUXDOF_ASPECT_PRESET]->u.pd.value);
	  if (preset_ratio > 0.0) s.anamorphic_ratio = preset_ratio; }
	s.depth_gamma = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_DEPTH_GAMMA]), 0.2, 4.0);
	s.depth_smoothing = Clamp01(DecodeSlider(params[ZLUXDOF_DEPTH_SMOOTHING]) * 0.01);
	s.aperture_map_index = ClampValue<A_long>(params[ZLUXDOF_APERTURE_MAP]->u.pd.value - 1, 0, 80);
	s.highlight_boost = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_ENHANCEMENT]) * 0.02, 0.0, 4.0);
	s.ca_rc = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_CA_RED_CYAN]) * 0.012, -1.2, 1.2);
	s.ca_gm = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_CA_GREEN_MAGENTA]) * 0.012, -1.2, 1.2);
	s.ca_by = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_CA_BLUE_YELLOW]) * 0.012, -1.2, 1.2);
	s.ca_strength = std::max(std::abs(s.ca_rc), std::max(std::abs(s.ca_gm), std::abs(s.ca_by)));
	s.render_mode = ClampValue<A_long>(params[ZLUXDOF_RENDER_MODE]->u.pd.value, 1, 3);
	s.vignetting = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_OPTICAL_VIGNETTING]) * 0.01, -1.0, 1.0);
	s.vignetting_scale = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_OPTICAL_VIGNETTING_SCALE]), 0.1, 4.0);
	s.astigmatism = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_ASTIGMATISM]) * 0.01, 0.0, 2.0);
	s.field_curvature = Clamp01(DecodeSlider(params[ZLUXDOF_FIELD_CURVATURE]) * 0.01);
	s.field_sweet = Clamp01(DecodeSlider(params[ZLUXDOF_FIELD_SWEET]) * 0.01);
	s.bokeh_rotation_rad = DecodeAngleRad(params[ZLUXDOF_BLADE_ANGLE]);
	s.spherical_aberration_amount = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_SPHERICAL_ABERRATION]) * 0.01, -1.0, 1.0);
	s.aperture_blades = ClampValue<A_long>(static_cast<A_long>(std::lround(DecodeSlider(params[ZLUXDOF_APERTURE_BLADES]))), 3, 16);
	s.catadioptric = params[ZLUXDOF_CATADIOPTRIC_LENS]->u.bd.value ? Clamp01(DecodeSlider(params[ZLUXDOF_CATADIOPTRIC_LENS_SCALE]) * 0.01) : 0.0;
	s.softness = Clamp01(DecodeSlider(params[ZLUXDOF_SOFTNESS]) * 0.01);
	s.onion_amount = Clamp01(DecodeSlider(params[ZLUXDOF_ONION_RINGS]) * 0.01);
	s.onion_count = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_ONION_RING_COUNT]), 3.0, 40.0);
	s.auto_focus_point = DecodeAutoFocusPoint(params[ZLUXDOF_DEPTH_SET_FOCUS], in_data);
	s.aperture_shape_mode = params[ZLUXDOF_APERTURE_SHAPE]->u.pd.value;
	s.enable_highlight = (DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_ENHANCEMENT]) > 0.001);
	s.astigmatism_type_sagittal = params[ZLUXDOF_SPHERICAL_ABERRATION_PLUS]->u.bd.value;
	s.depth_invert = params[ZLUXDOF_DEPTH_INVERT]->u.bd.value;
	s.depth_blackpoint = Clamp01(DecodeSlider(params[ZLUXDOF_DEPTH_BLACKPOINT]));
	s.depth_whitepoint = Clamp01(DecodeSlider(params[ZLUXDOF_DEPTH_WHITEPOINT]));
	if (s.depth_whitepoint <= s.depth_blackpoint + 0.001) {
		s.depth_whitepoint = Clamp01(s.depth_blackpoint + 0.001);
	}
	// Map slider [-100, +100] → curvature [-1, +1]. Negative = concave
	// (star-shape), 0 = straight polygon, positive = circular.
	s.blade_curve = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_BLADE_CURVE]) * 0.01, -1.0, 1.0);
	s.notch_angle = DecodeSlider(params[ZLUXDOF_NOTCH_ANGLE]) * (kPi / 180.0);
	s.notch_scale = Clamp01(DecodeSlider(params[ZLUXDOF_NOTCH_SCALE]) * 0.01);
	s.spherical_aberration_scale = Clamp01(DecodeSlider(params[ZLUXDOF_SPHERICAL_ABERRATION_SCALE]) * 0.01);
	s.highlights_low = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_LOWER_THRESHOLD]), 0.0, 255.0);
	s.highlights_high = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_UPPER_THRESHOLD]), 0.0, 255.0);
	s.highlights_softness = Clamp01(DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_SOFTNESS]) * 0.01);
	s.highlights_saturation = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_SATURATION]) * 0.01, -1.0, 1.0);
	s.bokeh_gamma = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_BOKEH_GAMMA]), 0.0, 3.0);
	s.highlight_scatter = Clamp01(DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_SCATTER]) * 0.01);
	s.highlight_mode = params[ZLUXDOF_HIGHLIGHTS_MODE]->u.pd.value - 1;
	s.highlight_recovery = Clamp01(DecodeSlider(params[ZLUXDOF_HIGHLIGHTS_RECOVERY]) * 0.01);
	s.highlights_tint = {
		static_cast<PF_FpLong>(params[ZLUXDOF_HIGHLIGHTS_TINT]->u.cd.value.red) / 255.0,
		static_cast<PF_FpLong>(params[ZLUXDOF_HIGHLIGHTS_TINT]->u.cd.value.green) / 255.0,
		static_cast<PF_FpLong>(params[ZLUXDOF_HIGHLIGHTS_TINT]->u.cd.value.blue) / 255.0
	};
	s.near_blur_factor = ClampValue<PF_FpLong>(DecodeSlider(params[ZLUXDOF_DEPTH_NEAR_BLUR]) * 0.01, 0.0, 2.0);
	s.foreground_protect = Clamp01(DecodeSlider(params[ZLUXDOF_DEPTH_FOREGROUND_PROTECT]) * 0.01);
	s.sample_count = ClampValue<A_long>(static_cast<A_long>(std::lround(DecodeSlider(params[ZLUXDOF_SAMPLE_COUNT]))), 16, kMaxVogelSamples);
	s.bokeh_definition = Clamp01(DecodeSlider(params[ZLUXDOF_BOKEH_DEFINITION]) * 0.01);
	s.energy_conserving = params[ZLUXDOF_ENERGY_CONSERVING]->u.bd.value;
	s.matte_top = Clamp01(DecodeSlider(params[ZLUXDOF_MATTEBOX_TOP]) * 0.01);
	s.matte_bottom = Clamp01(DecodeSlider(params[ZLUXDOF_MATTEBOX_BOTTOM]) * 0.01);
	s.matte_left = Clamp01(DecodeSlider(params[ZLUXDOF_MATTEBOX_LEFT]) * 0.01);
	s.matte_right = Clamp01(DecodeSlider(params[ZLUXDOF_MATTEBOX_RIGHT]) * 0.01);
	s.aperture_texture_intensity = Clamp01(DecodeSlider(params[ZLUXDOF_APTEX_INTENSITY]) * 0.01);
	s.aperture_texture_scale = ClampValue<PF_FpLong>(
		DecodeSlider(params[ZLUXDOF_APTEX_SCALE]) * 0.01, 0.25, 4.0);
	s.aperture_texture_offset = ClampValue<PF_FpLong>(
		DecodeSlider(params[ZLUXDOF_APTEX_OFFSET]) * 0.01, -1.0, 1.0);
	s.aperture_texture_invert = params[ZLUXDOF_APTEX_INVERT]->u.bd.value;
	s.noise_amount = Clamp01(DecodeSlider(params[ZLUXDOF_NOISE_AMOUNT]) * 0.01);
	s.noise_animated = params[ZLUXDOF_NOISE_ANIMATED]->u.bd.value;
	s.noise_monochromatic = params[ZLUXDOF_NOISE_MONOCHROMATIC]->u.bd.value;
	s.noise_luma_distribution = params[ZLUXDOF_NOISE_LUMA_DISTRIBUTION]->u.pd.value;
	s.noise_map_distribution = params[ZLUXDOF_NOISE_MAP_DISTRIBUTION]->u.pd.value;
	s.noise_tint = {
		static_cast<PF_FpLong>(params[ZLUXDOF_NOISE_TINT]->u.cd.value.red) / 255.0,
		static_cast<PF_FpLong>(params[ZLUXDOF_NOISE_TINT]->u.cd.value.green) / 255.0,
		static_cast<PF_FpLong>(params[ZLUXDOF_NOISE_TINT]->u.cd.value.blue) / 255.0
	};
	s.current_time = in_data->current_time;
	s.no_depth = !has_depth;

	// (Lens presets now bake into the sliders via ApplyPresetToParams on user
	// change, so there is no render-time preset override.)

	PF_EffectWorld* src_world = &params[ZLUXDOF_INPUT]->u.ld;
	PF_EffectWorld* effective_depth = has_depth ? depth_world : src_world;
	// Legacy non-smart entry point: no bitdepth field to consult here, so
	// WorldIsFloat falls back to its rowbytes heuristic. Left explicit rather
	// than implicit so it is clear this path is the one that still infers.
	const ScopedWorkingBpc bpc_scope(0);
	const PF_Boolean out_is_float = WorldIsFloat(output) ? TRUE : FALSE;
	err = RenderCore(in_data, out_data, src_world, output, effective_depth, aperture_tex_world, iris_mod_world, s,
	                 PF_WORLD_IS_DEEP(output), out_is_float);

	PF_Err checkin_err = PF_CHECKIN_PARAM(in_data, &depth_param);
	PF_Err checkin_custom_err = PF_CHECKIN_PARAM(in_data, &custom_aperture_param);
	PF_CHECKIN_PARAM(in_data, &iris_mod_param);
	if (!err) {
		err = checkin_err;
	}
	if (!err) {
		err = checkin_custom_err;
	}
	return err;
}

struct SmartRenderInfo {
	DOFSettings settings;
};

static PF_Err PreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extraP)
{
	PF_Err err = PF_Err_NONE;
	PF_RenderRequest req = extraP->input->output_request;
	PF_CheckoutResult in_result;

	AEGP_SuiteHandler suites(in_data->pica_basicP);
	PF_Handle infoH = suites.HandleSuite1()->host_new_handle(sizeof(SmartRenderInfo));
	if (!infoH) return PF_Err_OUT_OF_MEMORY;

	SmartRenderInfo* infoP = reinterpret_cast<SmartRenderInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!infoP) return PF_Err_OUT_OF_MEMORY;
	extraP->output->pre_render_data = infoH;

	DOFSettings& s = infoP->settings;
	memset(&s, 0, sizeof(s));

	#define CHECKOUT_VAL(IDX, PDEF) \
		PF_ParamDef PDEF; AEFX_CLR_STRUCT(PDEF); \
		ERR(PF_CHECKOUT_PARAM(in_data, IDX, in_data->current_time, in_data->time_step, in_data->time_scale, &PDEF))
	#define CHECKIN_VAL(PDEF) ERR(PF_CHECKIN_PARAM(in_data, &PDEF))

	CHECKOUT_VAL(ZLUXDOF_DISPLAY_MODE, p_dm); s.display_mode = p_dm.u.pd.value; CHECKIN_VAL(p_dm);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_CHANNEL, p_dc); s.depth_channel = p_dc.u.pd.value; CHECKIN_VAL(p_dc);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_FOCUS, p_foc); s.focal_distance = ClampValue<PF_FpLong>(p_foc.u.fs_d.value, 0.0, 1.0); CHECKIN_VAL(p_foc);
	CHECKOUT_VAL(ZLUXDOF_APERTURE_SIZE, p_as);
	s.blur_strength = ClampValue<PF_FpLong>(p_as.u.fs_d.value, 0.0, 100.0) * 0.01;
	CHECKIN_VAL(p_as);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_CURVE, p_curve);
	s.focus_range = ClampValue<PF_FpLong>(p_curve.u.fs_d.value * 0.04, 0.0, 0.25);
	CHECKIN_VAL(p_curve);
	CHECKOUT_VAL(ZLUXDOF_ASPECT_RATIO, p_ar);
	s.anamorphic_ratio = std::max<PF_FpLong>(0.1, ClampValue<PF_FpLong>(p_ar.u.fs_d.value, 0.0, 4.0));
	CHECKIN_VAL(p_ar);
	{ CHECKOUT_VAL(ZLUXDOF_ASPECT_PRESET, p_apre);
	  const PF_FpLong preset_ratio = AspectPresetRatio(p_apre.u.pd.value);
	  if (preset_ratio > 0.0) s.anamorphic_ratio = preset_ratio;
	  CHECKIN_VAL(p_apre); }
	CHECKOUT_VAL(ZLUXDOF_DEPTH_GAMMA, p_dg); s.depth_gamma = ClampValue<PF_FpLong>(p_dg.u.fs_d.value, 0.2, 4.0); CHECKIN_VAL(p_dg);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_SMOOTHING, p_dsm); s.depth_smoothing = Clamp01(p_dsm.u.fs_d.value * 0.01); CHECKIN_VAL(p_dsm);
	CHECKOUT_VAL(ZLUXDOF_APERTURE_MAP, p_amap); s.aperture_map_index = ClampValue<A_long>(p_amap.u.pd.value - 1, 0, 80); CHECKIN_VAL(p_amap);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_ENHANCEMENT, p_he);
	s.highlight_boost = ClampValue<PF_FpLong>(p_he.u.fs_d.value * 0.02, 0.0, 4.0);
	s.enable_highlight = p_he.u.fs_d.value > 0.001;
	CHECKIN_VAL(p_he);
	CHECKOUT_VAL(ZLUXDOF_CA_RED_CYAN, p_crc); s.ca_rc = ClampValue<PF_FpLong>(p_crc.u.fs_d.value * 0.012, -1.2, 1.2); CHECKIN_VAL(p_crc);
	CHECKOUT_VAL(ZLUXDOF_CA_GREEN_MAGENTA, p_cgm); s.ca_gm = ClampValue<PF_FpLong>(p_cgm.u.fs_d.value * 0.012, -1.2, 1.2); CHECKIN_VAL(p_cgm);
	CHECKOUT_VAL(ZLUXDOF_CA_BLUE_YELLOW, p_cby); s.ca_by = ClampValue<PF_FpLong>(p_cby.u.fs_d.value * 0.012, -1.2, 1.2); CHECKIN_VAL(p_cby);
	s.ca_strength = std::max(std::abs(s.ca_rc), std::max(std::abs(s.ca_gm), std::abs(s.ca_by)));
	CHECKOUT_VAL(ZLUXDOF_RENDER_MODE, p_rm); s.render_mode = ClampValue<A_long>(p_rm.u.pd.value, 1, 3); CHECKIN_VAL(p_rm);
	CHECKOUT_VAL(ZLUXDOF_OPTICAL_VIGNETTING, p_ov);
	CHECKOUT_VAL(ZLUXDOF_OPTICAL_VIGNETTING_SCALE, p_ovs);
	s.vignetting = ClampValue<PF_FpLong>(p_ov.u.fs_d.value * 0.01, -1.0, 1.0);
	s.vignetting_scale = ClampValue<PF_FpLong>(p_ovs.u.fs_d.value, 0.1, 4.0);
	CHECKIN_VAL(p_ov); CHECKIN_VAL(p_ovs);
	CHECKOUT_VAL(ZLUXDOF_ASTIGMATISM, p_ast); s.astigmatism = ClampValue<PF_FpLong>(p_ast.u.fs_d.value * 0.01, 0.0, 2.0); CHECKIN_VAL(p_ast);
	CHECKOUT_VAL(ZLUXDOF_FIELD_CURVATURE, p_fc); s.field_curvature = Clamp01(p_fc.u.fs_d.value * 0.01); CHECKIN_VAL(p_fc);
	CHECKOUT_VAL(ZLUXDOF_FIELD_SWEET, p_fsw); s.field_sweet = Clamp01(p_fsw.u.fs_d.value * 0.01); CHECKIN_VAL(p_fsw);
	CHECKOUT_VAL(ZLUXDOF_BLADE_ANGLE, p_ba);
	s.bokeh_rotation_rad = FIX_2_FLOAT(p_ba.u.ad.value) * (kPi / 180.0);
	CHECKIN_VAL(p_ba);
	CHECKOUT_VAL(ZLUXDOF_SPHERICAL_ABERRATION, p_sa);
	s.spherical_aberration_amount = ClampValue<PF_FpLong>(p_sa.u.fs_d.value * 0.01, -1.0, 1.0);
	CHECKOUT_VAL(ZLUXDOF_SPHERICAL_ABERRATION_SCALE, p_sas);
	s.spherical_aberration_scale = Clamp01(p_sas.u.fs_d.value * 0.01);
	CHECKIN_VAL(p_sa); CHECKIN_VAL(p_sas);
	CHECKOUT_VAL(ZLUXDOF_APERTURE_BLADES, p_ab); s.aperture_blades = ClampValue<A_long>(static_cast<A_long>(std::lround(p_ab.u.fs_d.value)), 3, 16); CHECKIN_VAL(p_ab);
	CHECKOUT_VAL(ZLUXDOF_CATADIOPTRIC_LENS, p_cl2);
	CHECKOUT_VAL(ZLUXDOF_CATADIOPTRIC_LENS_SCALE, p_cls);
	s.catadioptric = p_cl2.u.bd.value ? Clamp01(p_cls.u.fs_d.value * 0.01) : 0.0;
	CHECKIN_VAL(p_cl2); CHECKIN_VAL(p_cls);
	CHECKOUT_VAL(ZLUXDOF_SOFTNESS, p_soft);
	s.softness = Clamp01(p_soft.u.fs_d.value * 0.01);
	CHECKOUT_VAL(ZLUXDOF_ONION_RINGS, p_onr); s.onion_amount = Clamp01(p_onr.u.fs_d.value * 0.01); CHECKIN_VAL(p_onr);
	CHECKOUT_VAL(ZLUXDOF_ONION_RING_COUNT, p_onc); s.onion_count = ClampValue<PF_FpLong>(p_onc.u.fs_d.value, 3.0, 40.0); CHECKIN_VAL(p_onc);
	CHECKIN_VAL(p_soft);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_SET_FOCUS, p_sf);
	s.auto_focus_point = DecodeAutoFocusPoint(&p_sf, in_data);
	CHECKIN_VAL(p_sf);
	CHECKOUT_VAL(ZLUXDOF_APERTURE_SHAPE, p_shape); s.aperture_shape_mode = p_shape.u.pd.value; CHECKIN_VAL(p_shape);
	CHECKOUT_VAL(ZLUXDOF_SPHERICAL_ABERRATION_PLUS, p_sap); s.astigmatism_type_sagittal = p_sap.u.bd.value; CHECKIN_VAL(p_sap);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_INVERT, p_di); s.depth_invert = p_di.u.bd.value; CHECKIN_VAL(p_di);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_BLACKPOINT, p_bp); s.depth_blackpoint = Clamp01(p_bp.u.fs_d.value); CHECKIN_VAL(p_bp);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_WHITEPOINT, p_wp); s.depth_whitepoint = Clamp01(p_wp.u.fs_d.value); CHECKIN_VAL(p_wp);
	if (s.depth_whitepoint <= s.depth_blackpoint + 0.001) s.depth_whitepoint = Clamp01(s.depth_blackpoint + 0.001);
	CHECKOUT_VAL(ZLUXDOF_BLADE_CURVE, p_bc); s.blade_curve = ClampValue<PF_FpLong>(p_bc.u.fs_d.value * 0.01, -1.0, 1.0); CHECKIN_VAL(p_bc);
	CHECKOUT_VAL(ZLUXDOF_NOTCH_ANGLE, p_na); s.notch_angle = p_na.u.fs_d.value * (kPi / 180.0); CHECKIN_VAL(p_na);
	CHECKOUT_VAL(ZLUXDOF_NOTCH_SCALE, p_ns); s.notch_scale = Clamp01(p_ns.u.fs_d.value * 0.01); CHECKIN_VAL(p_ns);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_LOWER_THRESHOLD, p_hl); s.highlights_low = ClampValue<PF_FpLong>(p_hl.u.fs_d.value, 0.0, 255.0); CHECKIN_VAL(p_hl);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_UPPER_THRESHOLD, p_hu); s.highlights_high = ClampValue<PF_FpLong>(p_hu.u.fs_d.value, 0.0, 255.0); CHECKIN_VAL(p_hu);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_SOFTNESS, p_hs); s.highlights_softness = Clamp01(p_hs.u.fs_d.value * 0.01); CHECKIN_VAL(p_hs);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_SATURATION, p_hsat); s.highlights_saturation = ClampValue<PF_FpLong>(p_hsat.u.fs_d.value * 0.01, -1.0, 1.0); CHECKIN_VAL(p_hsat);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_BOKEH_GAMMA, p_bg); s.bokeh_gamma = ClampValue<PF_FpLong>(p_bg.u.fs_d.value, 0.0, 3.0); CHECKIN_VAL(p_bg);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_SCATTER, p_hsc); s.highlight_scatter = Clamp01(p_hsc.u.fs_d.value * 0.01); CHECKIN_VAL(p_hsc);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_MODE, p_hmo); s.highlight_mode = p_hmo.u.pd.value - 1; CHECKIN_VAL(p_hmo);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_RECOVERY, p_hrc); s.highlight_recovery = Clamp01(p_hrc.u.fs_d.value * 0.01); CHECKIN_VAL(p_hrc);
	CHECKOUT_VAL(ZLUXDOF_HIGHLIGHTS_TINT, p_ht);
	s.highlights_tint = { static_cast<PF_FpLong>(p_ht.u.cd.value.red) / 255.0, static_cast<PF_FpLong>(p_ht.u.cd.value.green) / 255.0, static_cast<PF_FpLong>(p_ht.u.cd.value.blue) / 255.0 };
	CHECKIN_VAL(p_ht);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_NEAR_BLUR, p_nb); s.near_blur_factor = ClampValue<PF_FpLong>(p_nb.u.fs_d.value * 0.01, 0.0, 2.0); CHECKIN_VAL(p_nb);
	CHECKOUT_VAL(ZLUXDOF_DEPTH_FOREGROUND_PROTECT, p_fp); s.foreground_protect = Clamp01(p_fp.u.fs_d.value * 0.01); CHECKIN_VAL(p_fp);
	CHECKOUT_VAL(ZLUXDOF_SAMPLE_COUNT, p_sc); s.sample_count = ClampValue<A_long>(static_cast<A_long>(std::lround(p_sc.u.fs_d.value)), 16, kMaxVogelSamples); CHECKIN_VAL(p_sc);
	CHECKOUT_VAL(ZLUXDOF_BOKEH_DEFINITION, p_bdef); s.bokeh_definition = Clamp01(p_bdef.u.fs_d.value * 0.01); CHECKIN_VAL(p_bdef);
	CHECKOUT_VAL(ZLUXDOF_ENERGY_CONSERVING, p_ec); s.energy_conserving = p_ec.u.bd.value; CHECKIN_VAL(p_ec);
	CHECKOUT_VAL(ZLUXDOF_MATTEBOX_TOP, p_mt); s.matte_top = Clamp01(p_mt.u.fs_d.value * 0.01); CHECKIN_VAL(p_mt);
	CHECKOUT_VAL(ZLUXDOF_MATTEBOX_BOTTOM, p_mb); s.matte_bottom = Clamp01(p_mb.u.fs_d.value * 0.01); CHECKIN_VAL(p_mb);
	CHECKOUT_VAL(ZLUXDOF_MATTEBOX_LEFT, p_ml); s.matte_left = Clamp01(p_ml.u.fs_d.value * 0.01); CHECKIN_VAL(p_ml);
	CHECKOUT_VAL(ZLUXDOF_MATTEBOX_RIGHT, p_mr); s.matte_right = Clamp01(p_mr.u.fs_d.value * 0.01); CHECKIN_VAL(p_mr);
	CHECKOUT_VAL(ZLUXDOF_APTEX_INTENSITY, p_apti);
	s.aperture_texture_intensity = Clamp01(p_apti.u.fs_d.value * 0.01); CHECKIN_VAL(p_apti);
	CHECKOUT_VAL(ZLUXDOF_APTEX_SCALE, p_apts);
	s.aperture_texture_scale = ClampValue<PF_FpLong>(p_apts.u.fs_d.value * 0.01, 0.25, 4.0); CHECKIN_VAL(p_apts);
	CHECKOUT_VAL(ZLUXDOF_APTEX_OFFSET, p_apto);
	s.aperture_texture_offset = ClampValue<PF_FpLong>(p_apto.u.fs_d.value * 0.01, -1.0, 1.0); CHECKIN_VAL(p_apto);
	CHECKOUT_VAL(ZLUXDOF_APTEX_INVERT, p_apin);
	s.aperture_texture_invert = p_apin.u.bd.value; CHECKIN_VAL(p_apin);
	CHECKOUT_VAL(ZLUXDOF_NOISE_AMOUNT, p_nam); s.noise_amount = Clamp01(p_nam.u.fs_d.value * 0.01); CHECKIN_VAL(p_nam);
	CHECKOUT_VAL(ZLUXDOF_NOISE_ANIMATED, p_nanim); s.noise_animated = p_nanim.u.bd.value; CHECKIN_VAL(p_nanim);
	CHECKOUT_VAL(ZLUXDOF_NOISE_MONOCHROMATIC, p_nmono); s.noise_monochromatic = p_nmono.u.bd.value; CHECKIN_VAL(p_nmono);
	CHECKOUT_VAL(ZLUXDOF_NOISE_LUMA_DISTRIBUTION, p_nld); s.noise_luma_distribution = p_nld.u.pd.value; CHECKIN_VAL(p_nld);
	CHECKOUT_VAL(ZLUXDOF_NOISE_MAP_DISTRIBUTION, p_nmd); s.noise_map_distribution = p_nmd.u.pd.value; CHECKIN_VAL(p_nmd);
	CHECKOUT_VAL(ZLUXDOF_NOISE_TINT, p_nt);
	s.noise_tint = { static_cast<PF_FpLong>(p_nt.u.cd.value.red) / 255.0, static_cast<PF_FpLong>(p_nt.u.cd.value.green) / 255.0, static_cast<PF_FpLong>(p_nt.u.cd.value.blue) / 255.0 };
	CHECKIN_VAL(p_nt);
	s.current_time = in_data->current_time;

	// (No render-time preset override: lens presets bake into the sliders via
	// ApplyPresetToParams when the user picks one.)

	#undef CHECKOUT_VAL
	#undef CHECKIN_VAL

	// SDK-standard checkout. No manual expansion of req.rect: with
	// PF_OutFlag_I_EXPAND_BUFFER removed, AE treats us as a 1:1 filter and
	// returning the input's own result/max rects is both correct and safe
	// on adjustment layers (error 25::237). Gather samples that fall
	// outside the returned region simply clamp to edge during sampling.
	AEFX_CLR_STRUCT(in_result);
	ERR(extraP->cb->checkout_layer(in_data->effect_ref, ZLUXDOF_INPUT, ZLUXDOF_INPUT,
		&req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));
	UnionLRect(&in_result.result_rect,     &extraP->output->result_rect);
	UnionLRect(&in_result.max_result_rect, &extraP->output->max_result_rect);

	PF_CheckoutResult depth_result;
	AEFX_CLR_STRUCT(depth_result);
	extraP->cb->checkout_layer(in_data->effect_ref, ZLUXDOF_DEPTH_LAYER, ZLUXDOF_DEPTH_LAYER,
		&req, in_data->current_time, in_data->time_step, in_data->time_scale, &depth_result);

	PF_CheckoutResult aperture_result;
	AEFX_CLR_STRUCT(aperture_result);
	extraP->cb->checkout_layer(in_data->effect_ref, ZLUXDOF_CUSTOM_APERTURE_LAYER, ZLUXDOF_CUSTOM_APERTURE_LAYER,
		&req, in_data->current_time, in_data->time_step, in_data->time_scale, &aperture_result);

	// Iris Texture modulator -- drives the APTEX slider group's layer slot.
	// Independent of the Custom Iris Layer; can be loaded alongside any
	// polygonal / circular iris to add dust, oily-coating, or vintage-
	// imperfection textures on top of the aperture mask.
	PF_CheckoutResult iris_mod_result;
	AEFX_CLR_STRUCT(iris_mod_result);
	extraP->cb->checkout_layer(in_data->effect_ref, ZLUXDOF_APTEX_LAYER, ZLUXDOF_APTEX_LAYER,
		&req, in_data->current_time, in_data->time_step, in_data->time_scale, &iris_mod_result);

	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}

static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extraP)
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	SmartRenderInfo* infoP = reinterpret_cast<SmartRenderInfo*>(
		suites.HandleSuite1()->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));
	if (!infoP) return PF_Err_INTERNAL_STRUCT_DAMAGED;

	PF_EffectWorld* input_worldP = nullptr;
	PF_EffectWorld* output_worldP = nullptr;
	PF_EffectWorld* depth_worldP = nullptr;
	PF_EffectWorld* aperture_worldP = nullptr;
	PF_EffectWorld* iris_mod_worldP = nullptr;

	ERR(extraP->cb->checkout_layer_pixels(in_data->effect_ref, ZLUXDOF_INPUT, &input_worldP));
	ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));
	extraP->cb->checkout_layer_pixels(in_data->effect_ref, ZLUXDOF_DEPTH_LAYER, &depth_worldP);
	extraP->cb->checkout_layer_pixels(in_data->effect_ref, ZLUXDOF_CUSTOM_APERTURE_LAYER, &aperture_worldP);
	extraP->cb->checkout_layer_pixels(in_data->effect_ref, ZLUXDOF_APTEX_LAYER, &iris_mod_worldP);

	if (!err && input_worldP && output_worldP) {
		PF_EffectWorld* tex_world = nullptr;
		if (aperture_worldP && aperture_worldP->data) tex_world = aperture_worldP;
		PF_EffectWorld* iris_mod_world = nullptr;
		if (iris_mod_worldP && iris_mod_worldP->data) iris_mod_world = iris_mod_worldP;

		const bool sr_has_depth = (depth_worldP && depth_worldP->data);
		infoP->settings.no_depth = !sr_has_depth;
		PF_EffectWorld* sr_eff_depth = sr_has_depth ? depth_worldP : input_worldP;

		// AE states the working depth outright; no need to infer it from row
		// strides (which is ambiguous on very narrow worlds -- see WorldIsFloat).
		// Publishing it here also makes every WorldIsFloat call below this point,
		// on any of the checked-out layers, exact.
		const ScopedWorkingBpc bpc_scope(extraP->input->bitdepth);
		const PF_Boolean sr_is_float = WorldIsFloat(output_worldP) ? TRUE : FALSE;

		err = RenderCore(in_data, out_data, input_worldP, output_worldP, sr_eff_depth, tex_world, iris_mod_world,
		                 infoP->settings, PF_WORLD_IS_DEEP(output_worldP), sr_is_float);
	}

	suites.HandleSuite1()->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));
	return err;
}

} // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB2 inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		"zluxDOF",
		"ADBE zluxDOF",
		"MiLai",
		AE_RESERVED_INFO,
		"EffectMain",
		"https://www.adobe.com");

	return result;
}

PF_Err EffectMain(
	PF_Cmd cmd,
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output,
	void* extra)
{
	PF_Err err = PF_Err_NONE;

	try {
		switch (cmd) {
			case PF_Cmd_ABOUT:
				err = About(in_data, out_data, params, output);
				break;
			case PF_Cmd_GLOBAL_SETUP:
				err = GlobalSetup(in_data, out_data, params, output);
				break;
			case PF_Cmd_GLOBAL_SETDOWN:
				err = GlobalSetdown(in_data, out_data, params, output);
				break;
			case PF_Cmd_PARAMS_SETUP:
				err = ParamsSetup(in_data, out_data, params, output);
				break;
			case PF_Cmd_UPDATE_PARAMS_UI:
				err = UpdateParamsUI(in_data, out_data, params, output);
				break;
			case PF_Cmd_USER_CHANGED_PARAM:
				err = UserChangedParam(in_data, out_data, params,
					reinterpret_cast<const PF_UserChangedParamExtra*>(extra));
				break;
			case PF_Cmd_EVENT:
				err = HandleEvent(in_data, out_data, params, output,
					reinterpret_cast<PF_EventExtra*>(extra));
				break;
			case PF_Cmd_RENDER:
				err = Render(in_data, out_data, params, output);
				break;
			case PF_Cmd_SMART_PRE_RENDER:
				err = PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra));
				break;
			case PF_Cmd_SMART_RENDER:
				err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra));
				break;
		}
	} catch (PF_Err& thrown_err) {
		err = thrown_err;
	} catch (...) {
		// Defense in depth: a std::bad_alloc / std::system_error (e.g. a worker
		// thread failing to spawn) or any other C++ exception must NEVER escape
		// into AE's host -- an unhandled exception there is a hard "After Effects
		// crashed" dialog, not a recoverable effect error. Convert anything that
		// reaches here into a normal PF error so the host stays alive.
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	return err;
}
