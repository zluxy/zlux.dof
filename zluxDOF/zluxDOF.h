#pragma once

#ifndef ZLUXDOF_H
#define ZLUXDOF_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"

#ifdef AE_OS_WIN
	typedef unsigned short PixelType;
	#include <Windows.h>
	#ifdef max
		#undef max
	#endif
	#ifdef min
		#undef min
	#endif
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectUI.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"
#include "AEFX_SuiteHelper.h"
#include "Smart_Utils.h"

#include "zluxDOF_Strings.h"

// Version macros live in zluxDOF_Version.h -- shared with zluxDOFPiPL.r so
// the code and the PiPL resource version always stay in sync automatically.
#include "zluxDOF_Version.h"

// Slot order. AE persists parameter values by their disk IDs, not by slot
// position, so reordering this enum between releases is safe; disk IDs below
// must remain stable for saved projects to round-trip correctly.
enum {
	ZLUXDOF_INPUT = 0,

	ZLUXDOF_BANNER,
	// The dedicated About button was removed in v2.9.0 -- clicking the banner
	// opens the About dialog, which made the extra button slot redundant.
	ZLUXDOF_RENDER_MODE,
	ZLUXDOF_PRESET,

	ZLUXDOF_DISPLAY_GROUP_START,
	ZLUXDOF_DISPLAY_MODE,
	ZLUXDOF_DISPLAY_GROUP_END,

	ZLUXDOF_DEPTH_GROUP_START,
	ZLUXDOF_DEPTH_LAYER,
	ZLUXDOF_DEPTH_CHANNEL,
	ZLUXDOF_DEPTH_INVERT,
	ZLUXDOF_DEPTH_FOCUS,
	ZLUXDOF_DEPTH_SET_FOCUS,
	// v2.21 DoF Mode / Depth Encoding popups + Physical Lens sliders (Focal
	// Length, F-Stop, Sensor Width, Scene Near/Far) were removed in v2.22: the
	// plugin is always the Aperture (Simple) inverse-Z model on a Normalized
	// (linear) depth map. Their disk IDs are retired below.
	ZLUXDOF_DEPTH_CURVE,
	ZLUXDOF_DEPTH_NEAR_BLUR,
	// v2.27: Levels-style custom-UI control -- a depth histogram with draggable
	// black / gamma / white handles that drive the three sliders below it.
	ZLUXDOF_DEPTH_LEVELS,
	ZLUXDOF_DEPTH_BLACKPOINT,
	ZLUXDOF_DEPTH_WHITEPOINT,
	ZLUXDOF_DEPTH_GAMMA,
	ZLUXDOF_DEPTH_SMOOTHING,
	ZLUXDOF_DEPTH_FOREGROUND_PROTECT,
	ZLUXDOF_DEPTH_GROUP_END,

	ZLUXDOF_APERTURE_GROUP_START,
	ZLUXDOF_BOKEH_PREVIEW,
	ZLUXDOF_APERTURE_SHAPE,
	ZLUXDOF_APERTURE_MAP,
	ZLUXDOF_APMAP_PICKER,
	ZLUXDOF_CUSTOM_APERTURE_LAYER,
	ZLUXDOF_ASPECT_PRESET,
	ZLUXDOF_ASPECT_RATIO,
	ZLUXDOF_APERTURE_SIZE,
	ZLUXDOF_SAMPLE_COUNT,
	// v3.1: how much the gather is allowed to pre-average the source before it
	// integrates the disc. Governs how crisp a bokeh edge can get -- see the
	// footprint-floor block in GatherPass.
	ZLUXDOF_BOKEH_DEFINITION,
	ZLUXDOF_ENERGY_CONSERVING,
	ZLUXDOF_SOFTNESS,
	ZLUXDOF_ONION_RINGS,
	ZLUXDOF_ONION_RING_COUNT,
	ZLUXDOF_APERTURE_BLADES,
	ZLUXDOF_BLADE_ANGLE,
	ZLUXDOF_BLADE_CURVE,
	ZLUXDOF_NOTCH_ANGLE,
	ZLUXDOF_NOTCH_SCALE,
	ZLUXDOF_APERTURE_GROUP_END,

	ZLUXDOF_LENS_CHARACTER_GROUP_START,
	ZLUXDOF_SPHERICAL_ABERRATION,
	ZLUXDOF_SPHERICAL_ABERRATION_PLUS,
	ZLUXDOF_SPHERICAL_ABERRATION_SCALE,
	ZLUXDOF_OPTICAL_VIGNETTING,
	ZLUXDOF_OPTICAL_VIGNETTING_SCALE,
	ZLUXDOF_ASTIGMATISM,
	ZLUXDOF_FIELD_CURVATURE,
	ZLUXDOF_FIELD_SWEET,
	ZLUXDOF_CATADIOPTRIC_LENS,
	ZLUXDOF_CATADIOPTRIC_LENS_SCALE,
	ZLUXDOF_LENS_CHARACTER_GROUP_END,

	ZLUXDOF_CA_GROUP_START,
	// v2.12: CA reworked into three independent opponent-axis sliders
	// (the Mode popup + Longitudinal/Lateral pair were retired).
	ZLUXDOF_CA_RED_CYAN,
	ZLUXDOF_CA_GREEN_MAGENTA,
	ZLUXDOF_CA_BLUE_YELLOW,
	// "Extended CA Pass" checkbox removed in v2.8.0 -- it was a hidden ×1.67
	// strength multiplier with no optical meaning of its own; the CA sliders
	// cover the same range. Its disk ID is retired below.
	ZLUXDOF_CA_GROUP_END,

	ZLUXDOF_HIGHLIGHTS_GROUP_START,
	ZLUXDOF_HIGHLIGHTS_LOWER_THRESHOLD,
	ZLUXDOF_HIGHLIGHTS_UPPER_THRESHOLD,
	ZLUXDOF_HIGHLIGHTS_SOFTNESS,
	ZLUXDOF_HIGHLIGHTS_SATURATION,
	ZLUXDOF_HIGHLIGHTS_ENHANCEMENT,
	ZLUXDOF_HIGHLIGHTS_BOKEH_GAMMA,
	ZLUXDOF_HIGHLIGHTS_SCATTER,
	ZLUXDOF_HIGHLIGHTS_MODE,
	ZLUXDOF_HIGHLIGHTS_RECOVERY,
	ZLUXDOF_HIGHLIGHTS_TINT,
	ZLUXDOF_HIGHLIGHTS_GROUP_END,

	ZLUXDOF_APTEX_GROUP_START,
	ZLUXDOF_APTEX_LAYER,
	ZLUXDOF_APTEX_INVERT,
	ZLUXDOF_APTEX_INTENSITY,
	ZLUXDOF_APTEX_SCALE,
	ZLUXDOF_APTEX_OFFSET,
	ZLUXDOF_APTEX_GROUP_END,

	ZLUXDOF_MATTEBOX_GROUP_START,
	ZLUXDOF_MATTEBOX_TOP,
	ZLUXDOF_MATTEBOX_BOTTOM,
	ZLUXDOF_MATTEBOX_LEFT,
	ZLUXDOF_MATTEBOX_RIGHT,
	ZLUXDOF_MATTEBOX_GROUP_END,

	ZLUXDOF_NOISE_GROUP_START,
	ZLUXDOF_NOISE_AMOUNT,
	ZLUXDOF_NOISE_ANIMATED,
	ZLUXDOF_NOISE_MONOCHROMATIC,
	ZLUXDOF_NOISE_LUMA_DISTRIBUTION,
	ZLUXDOF_NOISE_MAP_DISTRIBUTION,
	ZLUXDOF_NOISE_TINT,
	ZLUXDOF_NOISE_GROUP_END,

	ZLUXDOF_NUM_PARAMS
};

// Disk IDs are the ABI contract with saved projects. Never change or reorder
// the existing numeric values; always append new IDs at the bottom.
enum {
	DISPLAY_GROUP_START_DISK_ID = 1,
	DISPLAY_MODE_DISK_ID,
	GAMMA_CORRECTION_DISK_ID,
	DISPLAY_GROUP_END_DISK_ID,

	DEPTH_GROUP_START_DISK_ID,
	DEPTH_LAYER_DISK_ID,
	DEPTH_CHANNEL_DISK_ID,
	DEPTH_INVERT_DISK_ID,
	DEPTH_FOCUS_DISK_ID,
	DEPTH_SET_FOCUS_DISK_ID,
	DEPTH_CURVE_DISK_ID,
	DEPTH_NEAR_BLUR_DISK_ID,
	DEPTH_BLACKPOINT_DISK_ID,
	DEPTH_WHITEPOINT_DISK_ID,
	DEPTH_GROUP_END_DISK_ID,

	APERTURE_GROUP_START_DISK_ID,
	APERTURE_SHAPE_DISK_ID,
	CUSTOM_APERTURE_LAYER_DISK_ID,
	ASPECT_RATIO_DISK_ID,
	APERTURE_SIZE_DISK_ID,
	SAMPLE_COUNT_DISK_ID,
	APERTURE_BLADES_DISK_ID,
	BLADE_ANGLE_DISK_ID,
	BLADE_CURVE_DISK_ID,
	NOTCH_ANGLE_DISK_ID,
	NOTCH_SCALE_DISK_ID,
	SPHERICAL_ABERRATION_DISK_ID,
	SPHERICAL_ABERRATION_PLUS_DISK_ID,
	SPHERICAL_ABERRATION_SCALE_DISK_ID,
	SPHERICAL_ABERRATION_OFFSET_DISK_ID,
	SOFTNESS_DISK_ID,
	OPTICAL_VIGNETTING_DISK_ID,
	OPTICAL_VIGNETTING_SCALE_DISK_ID,
	ASTIGMATISM_DISK_ID,
	// CA Mode / Longitudinal / Lateral removed in v2.12 (replaced by the
	// three opponent-axis sliders); IDs retired.
	RETIRED_CHROMATIC_ABERRATION_MODE_DISK_ID,
	RETIRED_LONG_CHROMATIC_ABERRATION_DISK_ID,
	RETIRED_LAT_CHROMATIC_ABERRATION_DISK_ID,
	// Param removed in v2.8.0 ("Extended CA Pass"); ID retired to keep later
	// disk IDs stable for saved projects.
	RETIRED_CHROMATIC_ABERRATION_PLUS_DISK_ID,
	CATADIOPTRIC_LENS_DISK_ID,
	CATADIOPTRIC_LENS_SCALE_DISK_ID,
	APERTURE_GROUP_END_DISK_ID,

	HIGHLIGHTS_GROUP_START_DISK_ID,
	HIGHLIGHTS_LOWER_THRESHOLD_DISK_ID,
	HIGHLIGHTS_UPPER_THRESHOLD_DISK_ID,
	HIGHLIGHTS_SOFTNESS_DISK_ID,
	HIGHLIGHTS_SATURATION_DISK_ID,
	HIGHLIGHTS_ENHANCEMENT_DISK_ID,
	HIGHLIGHTS_TINT_DISK_ID,
	HIGHLIGHTS_GROUP_END_DISK_ID,

	APTEX_GROUP_START_DISK_ID,
	APTEX_LAYER_DISK_ID,
	APTEX_INVERT_DISK_ID,
	APTEX_INTENSITY_DISK_ID,
	APTEX_SCALE_DISK_ID,
	APTEX_OFFSET_DISK_ID,
	APTEX_GROUP_END_DISK_ID,

	MATTEBOX_GROUP_START_DISK_ID,
	MATTEBOX_TOP_DISK_ID,
	MATTEBOX_BOTTOM_DISK_ID,
	MATTEBOX_LEFT_DISK_ID,
	MATTEBOX_RIGHT_DISK_ID,
	MATTEBOX_GROUP_END_DISK_ID,

	NOISE_GROUP_START_DISK_ID,
	NOISE_AMOUNT_DISK_ID,
	NOISE_ANIMATED_DISK_ID,
	NOISE_MONOCHROMATIC_DISK_ID,
	NOISE_LUMA_DISTRIBUTION_DISK_ID,
	NOISE_MAP_DISTRIBUTION_DISK_ID,
	NOISE_TINT_DISK_ID,
	NOISE_GROUP_END_DISK_ID,

	PRESET_DISK_ID,
	DEPTH_FOREGROUND_PROTECT_DISK_ID,

	// v2.0 additions
	BANNER_DISK_ID,
	// About button removed in v2.9.0 (banner click opens About); ID retired.
	RETIRED_ABOUT_BUTTON_DISK_ID,
	// BARREL_DISTORTION_DISK_ID was removed in v2.0 (never shipped); the
	// slot is retired to keep later disk IDs stable for saved projects.
	RETIRED_BARREL_DISTORTION_DISK_ID,
	LENS_CHARACTER_GROUP_START_DISK_ID,
	LENS_CHARACTER_GROUP_END_DISK_ID,
	CA_GROUP_START_DISK_ID,
	CA_GROUP_END_DISK_ID,

	// v2.1 additions
	HIGHLIGHTS_BOKEH_GAMMA_DISK_ID,

	// v2.2 additions
	HIGHLIGHTS_SCATTER_DISK_ID,
	HIGHLIGHTS_RECOVERY_DISK_ID,

	// v2.5 additions (DOF PRO parity pass)
	DEPTH_GAMMA_DISK_ID,
	DEPTH_SMOOTHING_DISK_ID,
	ASPECT_PRESET_DISK_ID,
	APERTURE_MAP_DISK_ID,
	APMAP_PICKER_DISK_ID,
	BOKEH_PREVIEW_DISK_ID,

	// v2.7 bloom / halation -- the whole feature was REMOVED in v2.27; IDs retired
	// (never reused) so later disk IDs stay stable for saved projects.
	RETIRED_HIGHLIGHTS_BLOOM_DISK_ID,
	RETIRED_HIGHLIGHTS_BLOOM_RADIUS_DISK_ID,
	RETIRED_HIGHLIGHTS_BLOOM_THRESHOLD_DISK_ID,

	// v2.8 additions
	RENDER_MODE_DISK_ID,

	// v2.9 additions (procedural onion rings)
	ONION_RINGS_DISK_ID,
	ONION_RING_COUNT_DISK_ID,

	// v2.10 additions (field curvature / sweet-spot blur)
	FIELD_CURVATURE_DISK_ID,
	FIELD_SWEET_DISK_ID,

	// v2.12 additions (opponent-axis chromatic aberration)
	CA_RED_CYAN_DISK_ID,
	CA_GREEN_MAGENTA_DISK_ID,
	CA_BLUE_YELLOW_DISK_ID,

	// v2.20/2.21 DoF Mode + Depth Encoding popups and the Physical Lens sliders.
	// Removed in v2.22 (single Aperture-Simple model on Normalized linear depth);
	// IDs retired -- never reused -- so later disk IDs stay stable for saved
	// projects and old projects round-trip (the retired params load and vanish).
	RETIRED_COC_MODEL_DISK_ID,
	RETIRED_DEPTH_ENCODING_DISK_ID,
	RETIRED_FOCAL_LENGTH_DISK_ID,
	RETIRED_FNUMBER_DISK_ID,
	RETIRED_SENSOR_WIDTH_DISK_ID,
	RETIRED_SCENE_NEAR_DISK_ID,
	RETIRED_SCENE_FAR_DISK_ID,

	// v2.23 Layered Bokeh checkbox + v2.25 Layered Bokeh mode popup -- the whole
	// feature was REMOVED in v2.26. Both IDs kept RESERVED (never reused) so the
	// disk IDs around them stay stable for saved projects.
	LAYERED_BOKEH_DISK_ID,

	// v2.24 energy-conserving gather (kept). sRGB Linearize + Background Inpaint
	// were REMOVED in v2.27; their IDs are retired (never reused) so later disk
	// IDs stay stable for saved projects.
	ENERGY_CONSERVING_DISK_ID,
	RETIRED_SRGB_LINEAR_DISK_ID,
	RETIRED_BG_INPAINT_DISK_ID,

	// v3.0 addition: Highlight Mode (Additive / Preservative).
	HIGHLIGHTS_MODE_DISK_ID,

	RETIRED_LAYERED_MODE_DISK_ID,
	// v2.27 Starburst params were prototyped then removed before shipping; their
	// disk IDs were never persisted, so nothing is reserved here.

	// v2.27: Depth Levels custom-UI control (histogram + handles).
	DEPTH_LEVELS_DISK_ID,

	// v3.1: Bokeh Definition (source pre-filter floor).
	BOKEH_DEFINITION_DISK_ID
};

// Aperture-map picker grid layout (clickable custom-UI thumbnail grid).
#define ZLUXDOF_APMAP_COLS 8
#define ZLUXDOF_APMAP_ROWS 10
#define ZLUXDOF_APMAP_PICKER_W 256
#define ZLUXDOF_APMAP_PICKER_H 320

// Live bokeh-shape preview custom-UI control dimensions.
#define ZLUXDOF_BOKEH_PREVIEW_W 220
#define ZLUXDOF_BOKEH_PREVIEW_H 220

// Depth Levels custom-UI control dimensions (depth histogram + black/gamma/white
// handles). AE hands the control the full panel width at draw time; these are the
// requested defaults.
#define ZLUXDOF_DEPTH_LEVELS_W 256
#define ZLUXDOF_DEPTH_LEVELS_H 132

// Effect-panel banner slot dimensions. The PNG is loaded from
// "zluxDOF_banner.png" next to the .aex (or via ZLUXDOF_BANNER_PATH
// env var). The drawer preserves aspect ratio, so any image scales to
// fit; shipping a 10:3 (400 x 120) asset fills the slot at 1:1 but any
// other aspect ratio will still render letterboxed and centred.
#define ZLUXDOF_BANNER_WIDTH   400
#define ZLUXDOF_BANNER_HEIGHT  100

extern "C" {
	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);
}

#endif // ZLUXDOF_H
