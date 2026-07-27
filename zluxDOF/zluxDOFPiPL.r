#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "zluxDOF_Version.h"

#ifndef AE_OS_WIN
	#include <AE_General.r>
#endif
	
resource 'PiPL' (16000) {
	{	/* array properties: 12 elements */
		/* [1] */
		Kind {
			AEEffect
		},
		/* [2] */
		Name {
			"zluxDOF"
		},
		/* [3] */
		Category {
			"Zluxia"
		},
#ifdef AE_OS_WIN
	#ifdef AE_PROC_INTELx64
		CodeWin64X86 {"EffectMain"},
	#endif
#else
	#ifdef AE_OS_MAC
		CodeMacIntel64 {"EffectMain"},
		CodeMacARM64 {"EffectMain"},
	#endif
#endif
		/* [6] */
		AE_PiPL_Version {
			2,
			0
		},
		/* [7] */
		AE_Effect_Spec_Version {
			PF_PLUG_IN_VERSION,
			PF_PLUG_IN_SUBVERS
		},
		/* [8] */
		// Expanded from zluxDOF_Version.h -- the SAME header zluxDOF.h uses
		// for GlobalSetup's out_data->my_version, so the two can never drift
		// apart (the old hand-written literal here caused AE's version-
		// mismatch dialog whenever a bump forgot this file). Versions are
		// bumped in zluxDOF_Version.h only.
		AE_Effect_Version {
			ZLUX_PIPL_VERSION
		},
		/* [9] */
		AE_Effect_Info_Flags {
			0
		},
		/* [10] */
		AE_Effect_Global_OutFlags {
		0x06008400
		},
		AE_Effect_Global_OutFlags_2 {
		0x08001400
		},
		/* [11] */
		AE_Effect_Match_Name {
			"ADBE zluxDOF"
		},
		/* [12] */
		AE_Reserved_Info {
			0
		},
		/* [13] */
		AE_Effect_Support_URL {
			"https://www.adobe.com"
		}
	}
};

