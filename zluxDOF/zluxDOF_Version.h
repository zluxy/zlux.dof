#pragma once

// ── Single source of truth for the plugin version ───────────────────────────
//
// Included by BOTH zluxDOF.h (GlobalSetup writes these into
// out_data->my_version) and zluxDOFPiPL.r (the PiPL resource version AE
// compares against). Because both sides expand the same macros, they can
// never drift apart and AE's "version mismatch" dialog (which silently
// degrades the load and drops the custom-UI banner) can never come back.
//
// BUMP VERSIONS HERE AND NOWHERE ELSE.
//
// ── HARD LIMITS -- read before bumping ──────────────────────────────────────
// AE packs the version into 32 bits (see PF_Vers_* in AE_Effect.h):
//
//   build   bits  0-8   max 511
//   stage   bits  9-10  max 3
//   bugfix  bits 11-14  max 15
//   subvers bits 15-18  max 15   <-- MINOR_VERSION
//   vers    bits 19-21  max 7    <-- MAJOR_VERSION
//
// MINOR_VERSION > 15 does NOT simply wrap: the two sides of this header
// disagree about how. PF_VERSION() masks each field (28 & 0xF == 12), while
// ZLUX_PIPL_VERSION below is plain arithmetic and lets the overflow carry into
// the major field (28 * 32768 sets bit 19). That is exactly how the plugin
// ended up reporting "Code version is 2.12 and PiPL version is 3.12" at
// MINOR_VERSION 28 -- and, more quietly, at every version from 2.16 onward.
//
// So: when MINOR_VERSION would reach 16, bump MAJOR and reset MINOR to 0.
// The static assertions in zluxDOF.cpp enforce this at compile time.
#define MAJOR_VERSION	3
#define MINOR_VERSION	1
#define BUG_VERSION		0
#define STAGE_VERSION	PF_Stage_DEVELOP
#define BUILD_VERSION	1

// Numeric stage value for the PiPL arithmetic: the resource pass cannot
// evaluate the PF_Stage_DEVELOP C enum. PF_Stage_DEVELOP is the FIRST member
// of the PF_Stage enum in AE_Effect.h, i.e. == 0 (a hand-assumed "1" here
// shipped a PiPL with a stray stage bit and AE threw "version mismatch:
// 2.11 vs 2.11 (158001)" -- the hex of the stage-less code version). Keep in
// sync with STAGE_VERSION above if the stage ever changes.
#define ZLUX_STAGE_NUM	0

// PF_VERSION bit layout as plain arithmetic, evaluatable by PiPLTool:
// vers<<19 | subvers<<15 | bug<<11 | stage<<9 | build.
#define ZLUX_PIPL_VERSION (MAJOR_VERSION * 524288 + MINOR_VERSION * 32768 + \
                           BUG_VERSION * 2048 + ZLUX_STAGE_NUM * 512 + BUILD_VERSION)
