# Methodology review: manipulation strength (C2, 2026-11-28)

Trigger: the user rejects discrete strength jumps and asked for a review of the whole manipulation methodology.

## What the code does today
- Genotype: one INTEGER gene per site (mmb/manipulator/vlm_manipulator.py casts to int64; DiscretePymooOptimizer). A gene value is an index into a lookup table.
- Image (mmb/manipulator/image/manipulator.py): a VQGAN code grid; a region gene k >= 1 moves every code of the region to its region_levels[k-1]-th nearest codeword. MAIN ladder: 1, 2, 4, ..., 8192 (14 exponential levels); decode goes through quantised indices only (codec.decode_batch -> get_codebook_entry).
- Text (mmb/manipulator/text/composite.py): a gene is an index into a per-site candidate table (synonyms / typos / char noise in MAIN; T-A words / T-B paraphrases now).
- Search: AGE-MOEA-II / NSGA-II / GA on integer genes; MMBA phase 2 resets whole sites to 0 (delta debugging).

## Problems
P1 Strength is not a quantity, it is a table index. Steps are neither uniform nor small in perceptual (image) or semantic (text) distance.
P2 Image rank is not image change. Rank 8192 lands on unrelated codes, so R0 = 100% image-only crossings at LPIPS 0.22. "Calibrating" by cutting the ladder (128 -> 32 -> 8 -> 4) truncates the table; the search compensates with more regions, so there is no real strength control.
P3 Text has almost no room: ImageNet 4 word sites, POPE 2, paraphrase bank 7-37 per prompt. The MAIN typo / homoglyph / zero-width operators were tokenizer attacks, not meaning-preserving edits.
P4 Every calibration round so far changed the OPERATOR DEFINITION, not one strength parameter. There is no single knob per channel and no trajectory of one knob.
P5 Minimality is counted in sites, not magnitude. A boundary pair differs by one discrete jump, so its distance is "the size of one edit by construction" (stated in the paper, RQ1). That is not an ON/OFF pair at the boundary.
P6 There is no common strength unit across channels (D8). Weights need a per-channel unit (LPIPS for the image, sentence-embedding distance for the text).

## Proposed continuous design
1. Genotype x in [0,1]^d (real), same sites. Channel weights w_img, w_txt; default (1,1) = uncalibrated.
2. Image: continuous latent displacement without quantisation. z_r' = z_r + w_img * x_r * s0 * d_r, where d_r is a fixed unit direction per region (e.g. toward the region's codes' live-codebook neighbourhood, or the top principal directions of the live codebook), decoded from continuous z. s0 is set so that (w = 1, x = 1) matches the MAIN operator's maximum displacement, which makes w = 1 literally "no calibration". Expectation: LPIPS smooth and monotone in x and w (to verify).
3. Text: text is discrete by nature, so valid text cannot be continuous. A continuous strength x maps to the candidate whose semantic distance is closest to x * w_txt * d_max in a dense, distance-ordered pool (T-A words + T-B paraphrases). The strength is continuous; the rendered output is piecewise constant, and the steps shrink as the pool grows. The validity filters cap the ceiling: report it.
4. Search: real-coded variation (SBX / polynomial mutation) for EVERY method (shared operators, balanced design). MMBA phase 2: (a) site removal (delta debugging), then (b) bisection on a global scale t of the remaining strengths, until the pair straddles the boundary within epsilon: a true ON/OFF pair with continuous distance.
5. Calibration: steps in (w_img, w_txt) from (1,1); target: text-only ~ image-only (McNemar) with both mostly crossing; raise w_txt before lowering w_img.

## Cost and risks
- Implementation ~1-2 days (C1): continuous decode path, real-coded optimizer adapters, text strength mapping, bisection phase, tests.
- Re-calibration ~10 GPU-h; then the full MAIN re-run (needed anyway).
- Risks: unquantised latents may decode off-manifold at large displacement (check realism and LPIPS); the text ceiling may stay low; the baselines must use the same real-coded genotype.
