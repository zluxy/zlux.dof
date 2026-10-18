<p align="center">
  <img src="assets/reverie-banner.png" width="800" alt="Reverie photographic banner">
</p>

<h1 align="center">Reverie</h1>
<p align="center"><strong>Depth of field with the character of a lens.</strong><br>For Adobe After Effects on Windows.</p>

<p align="center">
  <img alt="Version 3.3.0" src="https://img.shields.io/badge/version-3.3.0-3366aa?style=flat-square">
  <img alt="After Effects 2022 or newer" src="https://img.shields.io/badge/After%20Effects-2022%2B-6e5db8?style=flat-square">
  <img alt="Windows x64" src="https://img.shields.io/badge/Windows-x64-58697b?style=flat-square">
  <img alt="CUDA and CPU renderers" src="https://img.shields.io/badge/renderer-CUDA%20%2F%20CPU-3f8c6a?style=flat-square">
  <img alt="Freeware" src="https://img.shields.io/badge/license-freeware-9b7445?style=flat-square">
</p>

<p align="center">
  <a href="https://github.com/zluxy/zlux.dof/releases/tag/v3.3.0"><strong>Download</strong></a> ·
  <a href="#start-with-the-image">Quick start</a> ·
  <a href="#one-real-test-frame">Example</a> ·
  <a href="#the-numbers">Test results</a> ·
  <a href="#installation">Installation</a>
</p>

---

## A lens first. Depth when you need it.

I built Reverie around the part I kept coming back to: a blur should look like it passed through glass. Set a radius and **Physical Lens** blurs the whole frame, no depth map required. When the shot needs selective focus, add a depth layer and place the focus point on your subject. The character of the lens is still yours to shape.

You can get started with just the radius. When you want to go further, there are 44 editable lens looks, 80 aperture maps, custom iris layers, blades, curvature, rotation, texture, cat's-eye clipping and highlight controls. The detailed controls stay out of the way until you need them.

## One real test frame

This CS:GO frame was used to check the Highlights fix. Both images come from the same source; the second frame was rendered by Reverie with its depth map. Click either image to inspect it at full size.

### Original frame

<p align="center"><a href="assets/csgo-original.png"><img src="assets/csgo-original.png" width="960" alt="Original CS:GO frame"></a></p>

### Reverie 3.3.0

<p align="center"><a href="assets/csgo-reverie.png"><img src="assets/csgo-reverie.png" width="960" alt="CS:GO frame rendered through Reverie"></a></p>

<details>
<summary>Show the depth layer used for this frame</summary>

<p align="center"><img src="assets/csgo-depth.png" width="800" alt="Depth layer used for the CS:GO render"></p>

</details>

The test uses **Cyclop 85mm f/1.5**, a **35 px** blur radius and a focus point on the subject. Highlight Boost is deliberately set to **200** with a lower threshold of **0**: this is an aggressive regression test for the repaired controls, not a recommended grade for every shot. The image was produced through the native render test host, not captured from the After Effects interface.

<!-- FUTURE IMAGE SLOT: assets/real-world-depth.png — source, actual depth layer and final render from one real-life shot. -->
<!-- FUTURE IMAGE SLOT: assets/lens-comparison.png — same source/focus/radius with several lens looks. -->

## Start with the image

| Goal | First move |
| --- | --- |
| **Blur the whole frame** | Add Reverie and set **Blur Radius (px)**. Leave Depth Map empty. |
| **Keep a subject sharp** | Choose **Depth Map (Optional)** and move **Focus Point** onto the subject. |
| **Widen the sharp zone** | Raise **In-Focus Width**. Use **Show Focus** to see where focus lands. |
| **Change the look** | Pick a **Lens Look**, then tune its values under **Aperture & Iris**. |

For grayscale depth, point focus is the quickest route. For an EXR Z pass, first expose the Z channel as an After Effects layer, for example through EXtractoR, then select that layer in Reverie. **Depth Input** lets you choose grayscale or Z if Auto guesses wrong, invert the direction and adjust foreground fill. **Focus With → Slider** accepts 0–100 for grayscale or the raw distance for Z.

## The numbers

These native test-host results exercise the image-processing paths; they do not claim full compatibility with every After Effects version or timeline setup.

| Check | Result |
| --- | ---: |
| Full-resolution lens frames across both stereo EXR eyes, with and without depth | **180 / 180** passed |
| CPU / CUDA render comparisons | **90 / 90** within test tolerance |
| Directed lens-preset switches | **0 settings leaks in 1,936** checks |

This release fixes Highlights controls that appeared active but did not change the frame as intended. **Bokeh Brightness Boost** and **Bokeh Gamma** now affect selected out-of-focus highlights. Tint and saturation work without Boost. Threshold endpoints, procedural iris obstruction and Smart Render data handling received fixes too.

## Where depth still needs care

A single image cannot reveal what was hidden behind a foreground object. Reverie extends nearby visible background under soft foreground edges, but that is an estimate. A poor depth edge or a large occlusion may still need a better map or manual cleanup. The native tests above also do not cover every GPU driver, third-party depth source or interaction inside the full AE interface.

## Installation

Download the [Windows release](https://github.com/zluxy/zlux.dof/releases/tag/v3.3.0). The ZIP includes `Reverie.aex`, `cudart64_12.dll` and an installer. Close After Effects, extract the ZIP and run `install-build.bat` as administrator. The installer backs up installed Reverie, Lensmith or zluxDOF binaries it replaces.

To install manually, put `build_out/Reverie.aex` and `build_out/cudart64_12.dll` together in a MediaCore plugin folder, for example `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\Zlux\`. Restart AE and open **Effect → Zluxia → Reverie**. The standalone `.aex` is attached to the release if you already have the CUDA runtime. Without a compatible NVIDIA GPU, Reverie renders on the CPU.

This build targets Windows x64, an AVX2-capable CPU and After Effects 2022 or newer. It has a new effect identity and control layout: add a fresh instance to existing projects. Old Lensmith and earlier Reverie instances are not migrated.

---

Reverie is free for personal and commercial use. It is not open source; redistribution, resale and modification are not permitted. See [LICENSE.txt](LICENSE.txt).

Created by zlux / zluxia. Adobe and After Effects are trademarks of Adobe Inc.
