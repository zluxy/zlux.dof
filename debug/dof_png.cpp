// Standalone harness: compiles the REAL zluxDOF.cpp and runs RenderCore on raw
// ARGB8 frames dumped from PNGs by prep.py. RenderCore ignores in_data/out_data,
// so we pass nullptr. Build with debug/build_png.bat.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include "../zluxDOF/zluxDOF.cpp"   // brings in the anon-namespace RenderCore

static std::vector<unsigned char> read_raw(const char* p, long expect) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (expect && n != expect) { fprintf(stderr, "%s size %ld != %ld\n", p, n, expect); exit(2); }
    std::vector<unsigned char> b(n);
    fread(b.data(), 1, n, f); fclose(f);
    return b;
}

static void make_world(PF_EffectWorld& w, unsigned char* data, A_long width, A_long height) {
    memset(&w, 0, sizeof(w));
    w.world_flags = PF_WorldFlag_WRITEABLE;   // 8-bit (not DEEP), not float
    w.data = reinterpret_cast<PF_PixelPtr>(data);
    w.rowbytes = width * 4;
    w.width = width;
    w.height = height;
    w.extent_hint.left = 0; w.extent_hint.top = 0;
    w.extent_hint.right = width; w.extent_hint.bottom = height;
}

int main(int argc, char** argv) {
    // dof_png photo.raw depth.raw out.raw W H mode focus blur enc [foc_mm fno sensor near far]
    if (argc < 9) { fprintf(stderr, "usage: see source\n"); return 1; }
    const char* photo = argv[1];
    const char* depth = argv[2];
    const char* outp  = argv[3];
    const A_long W = atol(argv[4]);
    const A_long H = atol(argv[5]);
    const A_long mode = atol(argv[6]);          // 1 Aperture, 2 Physical
    const double focus = atof(argv[7]);         // 0..1
    const double blur  = atof(argv[8]);         // 0..100 (Blur Amount)
    const A_long enc   = (argc > 9)  ? atol(argv[9])  : 1;   // 1 norm,2 linZ,3 disp
    const double fmm   = (argc > 10) ? atof(argv[10]) : 50.0;
    const double fno   = (argc > 11) ? atof(argv[11]) : 2.8;
    const double sw    = (argc > 12) ? atof(argv[12]) : 36.0;
    const double snear = (argc > 13) ? atof(argv[13]) : 1.0;
    const double sfar  = (argc > 14) ? atof(argv[14]) : 100.0;
    const A_long disp  = (argc > 15) ? atol(argv[15]) : 1;   // display mode

    const long bytes = (long)W * H * 4;
    auto src = read_raw(photo, bytes);
    auto dep = read_raw(depth, bytes);
    std::vector<unsigned char> out(bytes, 0);

    PF_EffectWorld src_w, dep_w, out_w;
    make_world(src_w, src.data(), W, H);
    make_world(dep_w, dep.data(), W, H);
    make_world(out_w, out.data(), W, H);

    // Optional aperture layers, so the custom-iris and iris-modulator paths can
    // be A/B'd against the CPU reference like every other mode.
    //   ZLUX_APTEX=file.raw:W:H   custom iris shape  (also set ZLUX_SHAPE=4)
    //   ZLUX_IRISTEX=file.raw:W:H iris modulator
    std::vector<unsigned char> ap_buf, ir_buf;
    PF_EffectWorld ap_w{}, ir_w{};
    PF_EffectWorld* ap_p = nullptr;
    PF_EffectWorld* ir_p = nullptr;
    auto load_layer = [](const char* spec, std::vector<unsigned char>& buf,
                         PF_EffectWorld& w) -> PF_EffectWorld* {
        if (!spec) return nullptr;
        char path[512]; int lw = 0, lh = 0;
        if (sscanf(spec, "%511[^:]:%d:%d", path, &lw, &lh) != 3) return nullptr;
        buf = read_raw(path, (long)lw * lh * 4);
        make_world(w, buf.data(), lw, lh);
        return &w;
    };
    ap_p = load_layer(getenv("ZLUX_APTEX"),   ap_buf, ap_w);
    ir_p = load_layer(getenv("ZLUX_IRISTEX"), ir_buf, ir_w);

    (void)mode; (void)enc; (void)fmm; (void)fno; (void)sw; (void)snear; (void)sfar;
    DOFSettings s{};
    s.display_mode = disp;       // 1 Rendered, 9 CoC heat, 2 depth, 3 focus map
    s.depth_channel = 1;         // Luminance
    s.focal_distance = focus;
    s.blur_strength = blur * 0.01;
    s.focus_range = getenv("ZLUX_FOCRANGE") ? atof(getenv("ZLUX_FOCRANGE")) : 0.0;
    s.anamorphic_ratio = 1.0;
    s.depth_gamma = 1.0;
    s.depth_blackpoint = 0.0;
    s.depth_whitepoint = 1.0;
    s.depth_invert = FALSE;      // white = near (standard)
    s.near_blur_factor = 1.0;    // ENABLE near blur so we can see foreground defocus
    s.sample_count = getenv("ZLUX_SAMPLES") ? atol(getenv("ZLUX_SAMPLES")) : 256;
    // v3.1 Bokeh Definition (0..1). 0 reproduces the v3.0 footprint floor.
    s.bokeh_definition = getenv("ZLUX_BDEF") ? atof(getenv("ZLUX_BDEF")) : 0.65;
    s.render_mode = getenv("ZLUX_RMODE") ? atol(getenv("ZLUX_RMODE")) : 2; // 1 Fast 2 Final 3 Extreme
    s.energy_conserving = getenv("ZLUX_ENERGY") ? (atol(getenv("ZLUX_ENERGY")) != 0) : FALSE;
    // (srgb_linear / bg_inpaint were retired from DOFSettings; the harness used to
    //  drive them via ZLUX_SRGB / ZLUX_INPAINT.)
    // ZLUX_NODEPTH reproduces "no depth layer connected" -- the uniform-blur
    // path AE takes when the Depth Map slot is empty. This is what the plugin
    // does by default in a fresh comp, and it was never covered by the harness.
    s.no_depth = getenv("ZLUX_NODEPTH") ? TRUE : FALSE;
    // Lens Character knobs used to A/B the CUDA port against the CPU reference.
    s.astigmatism = getenv("ZLUX_ASTIG") ? atof(getenv("ZLUX_ASTIG")) : 0.0;
    s.astigmatism_type_sagittal = getenv("ZLUX_SAGITTAL") ? TRUE : FALSE;
    s.anamorphic_ratio = getenv("ZLUX_ANAM") ? atof(getenv("ZLUX_ANAM")) : 1.0;
    // The rest of the Lens Character group, so a Lens Preset row can be replayed
    // here in full. Without these the preset table was only checkable by reading
    // it: every entry drives softness / spherical aberration / vignetting /
    // catadioptric / onion / field curvature, and none of them were reachable.
    // All are in DECODED units (0..1, or the signed -1..1 the sliders map to),
    // not the 0..100 panel units the LensPreset table stores.
    s.softness         = getenv("ZLUX_SOFT")   ? atof(getenv("ZLUX_SOFT"))   : 0.0;
    s.spherical_aberration_amount = getenv("ZLUX_SA")    ? atof(getenv("ZLUX_SA"))    : 0.0;
    s.spherical_aberration_scale  = getenv("ZLUX_SASC")  ? atof(getenv("ZLUX_SASC"))  : 0.0;
    s.vignetting       = getenv("ZLUX_VIG")    ? atof(getenv("ZLUX_VIG"))    : 0.0;
    s.vignetting_scale = getenv("ZLUX_VIGSC")  ? atof(getenv("ZLUX_VIGSC"))  : 1.0;
    s.catadioptric     = getenv("ZLUX_CATA")   ? atof(getenv("ZLUX_CATA"))   : 0.0;
    s.onion_amount     = getenv("ZLUX_ONION")  ? atof(getenv("ZLUX_ONION"))  : 0.0;
    s.onion_count      = getenv("ZLUX_ONIONN") ? atof(getenv("ZLUX_ONIONN")) : 12.0;
    s.field_curvature  = getenv("ZLUX_FCURV")  ? atof(getenv("ZLUX_FCURV"))  : 0.0;
    s.field_sweet      = getenv("ZLUX_FSWEET") ? atof(getenv("ZLUX_FSWEET")) : 0.45;
    s.blade_curve      = getenv("ZLUX_CURVE")  ? atof(getenv("ZLUX_CURVE"))  : 0.0;
    s.aperture_blades = getenv("ZLUX_BLADES") ? atol(getenv("ZLUX_BLADES")) : 6;
    s.aperture_shape_mode = getenv("ZLUX_SHAPE") ? atol(getenv("ZLUX_SHAPE")) : 1;
    // ZLUX_APMAP=1..80 selects a built-in aperture-map library shape, so the
    // library path can be A/B'd (CPU vs GPU) like every other mode. The map is
    // read from aperture_lib/ next to this exe -- the harness has none of the
    // .aex's embedded resources.
    s.aperture_map_index = getenv("ZLUX_APMAP") ? atol(getenv("ZLUX_APMAP")) : 0;
    s.aperture_texture_intensity = getenv("ZLUX_APINT") ? atof(getenv("ZLUX_APINT")) : 1.0;
    s.aperture_texture_scale = getenv("ZLUX_APSCALE") ? atof(getenv("ZLUX_APSCALE")) : 1.0;
    s.aperture_texture_offset = getenv("ZLUX_APOFF") ? atof(getenv("ZLUX_APOFF")) : 0.0;
    s.aperture_texture_invert = getenv("ZLUX_APINV") ? TRUE : FALSE;
    s.auto_focus_point = { 0.5, 0.5 };
    // Highlight engine -- default to the plugin's real defaults, override via env.
    s.bokeh_gamma     = getenv("ZLUX_GAMMA")   ? atof(getenv("ZLUX_GAMMA"))   : 0.8;
    s.highlight_recovery = getenv("ZLUX_REC")  ? atof(getenv("ZLUX_REC"))     : 0.0;
    s.highlight_scatter  = getenv("ZLUX_SCAT") ? atof(getenv("ZLUX_SCAT"))    : 0.0;
    s.highlight_mode     = getenv("ZLUX_HLMODE")? atol(getenv("ZLUX_HLMODE"))  : 1;  // v3.1: Preservative
    s.highlight_boost    = getenv("ZLUX_BOOST")? atof(getenv("ZLUX_BOOST"))   : 0.0;
    s.highlights_low = 200.0; s.highlights_high = 255.0; s.highlights_softness = 0.01;

    // ZLUX_REPEAT renders the same frame N times in one process. A single run
    // measures cold start -- CUDA context creation, module load, GPU clock ramp
    // -- which is not what rendering a real sequence costs. Repeating reports
    // the steady-state per-frame number honestly.
    const int reps = getenv("ZLUX_REPEAT") ? atoi(getenv("ZLUX_REPEAT")) : 1;
    PF_Err err = PF_Err_NONE;
    for (int rep = 0; rep < reps; ++rep) {
        if (reps > 1) fprintf(stderr, "----- frame %d/%d -----\n", rep + 1, reps);
        err = RenderCore(nullptr, nullptr, &src_w, &out_w, &dep_w,
                         ap_p, ir_p, s, FALSE, FALSE);
    }
    fprintf(stderr, "RenderCore err=%d\n", (int)err);

    FILE* f = fopen(outp, "wb");
    fwrite(out.data(), 1, bytes, f); fclose(f);
    printf("wrote %s (%ldx%ld)\n", outp, (long)W, (long)H);
    return 0;
}
