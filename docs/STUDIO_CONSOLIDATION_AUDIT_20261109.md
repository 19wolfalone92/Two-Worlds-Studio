# Studio consolidation audit — 2026-11-09

## Scope and source of truth

Repository: `19wolfalone92/Two-Worlds-Studio` (public). Default branch: `main`.
At audit start: **54 branches (including main)** and **26 open pull requests**. The default-branch CI's latest reported push run was successful, but this is not a production-art or media quality certification.

This integration is a **staged candidate**, not a promise that the studio is fully consolidated or production-ready. Do not publish any content or run paid/quota-consuming GPU operations as part of the merge.

## Branch inventory and deduplication

- **main:** authoritative Studio v2/v3 foundation: ComfyUI/Kaggle handoff, source/continuity checks, release gate, Hugging Face read-only preflight, Cloudflare provider adapter and quota protection.
- **remaster/batch02-20261105** (190 commits ahead, 15 behind main): broadest recent private-production code lineage, derived from Batch03 and Batch04 and remaster feature work. Relative to main, its tree contains ~187 unique paths, 21 modified paths and lacks 45 newer main paths.
- **improve/story-glitch-studio-review-20261105** (189 ahead, 15 behind): near-duplicate of the remaster branch. Keep as historical checkpoint pending verification; do not blindly merge both.
- **production/story-glitch-batch04-20261105** (173 ahead, 15 behind) and **integration/story-glitch-batch03-20261104** (158 ahead, 15 behind): ancestors/preceding pipelines; importing entire trees again would duplicate historical private request transport and replace newer main safety gates.
- **feat/character-animation-engine**, **feat/character-art-studio-v2**, **feat/from-scratch-character-factory**, **feat/animation-production-v2**: ancestor modules for animation/art; newer remaster source includes their related source directories.
- **feature/studio-v2-*** and **feature/studio-v3-***, **fix/studio-v2-*** and **fix/studio-v3-***, **fix/comfy-***: newer `main` already contains their relevant modules and tests; do not regress to earlier versions.
- **feature/elizabeth-bones-neural-ru-voice-20261109**: newer specialized voice tools, separate approval and listening verification required.
- **feature/wolf-*** and `cleanup/migrate-wolf-ai-to-own-repo-20261107`: WOLF AI is a separate Android application; do not bundle its app, runtime, or workflows in the visual studio.

All other engineering branches are retained unchanged pending individual comparison and verified archival. No legacy branch is deleted by this integration.

## Integration policy

1. Start from the actual latest `main` commit.
2. Bring forward 99 missing, non-private code/test/doc paths from the remaster lineage without replacing the v2/v3 modules already on main.
3. Candidate updates to five overlapping camera/subtitles/hybrid-master files are included to exercise remaster timing and motion tests; **revert or reconcile if existing tests regress**.
4. Exclude `.github/*_requests` ciphertext, real unpublished assets, private request files/recipient keys and old branch-specific GPU workflows from the initial integration. Their safe preservation remains in original branches and private storage.
5. Run the combined unit test suite, import/compile checks, H.264/AAC/FFmpeg review smoke tests where supported by CI, and inspect every failure. Do not treat a `pytest`/CI pass as evidence of character artistic fidelity or live provider inference.
6. Integrate any remaining verified features and manually dispatched workflows only after separately reviewing their quota, secrets, privacy, branch guard and publication behaviors.
7. Merge the integration PR into `main` **only after all required tests pass and no source retention or rights regression remains**. Leave all old branches untouched until post-merge equivalence and backup audit.

## Known conflict surface

- `studio/smooth_camera.py`, `studio/hybrid_master.py`, `studio/subtitle_align.py`: frame clocks, smooth motion, subtitles and release-v2 contract need regression proof.
- `studio/actor_motion.py`, `studio/cinematic.py`, `studio/render.py`, `studio/identity_qc.py`, `studio/kaggle_api.py`, `studio/kaggle_flux.py`, `studio/art_director.py`, `studio/premium_assets.py`: legacy changes **not** copied over main, pending API-level reconciliation.
- Branch-specific GitHub Actions workflows and production request payloads have hard-coded refs and some quota-consuming steps. They must not be auto-enabled in main.
- Public GitHub must not contain private screenplay, video, audio, reference art, decryption keys or secrets.

## Completion checklist

- [x] Enumerate 54 branches and 26 open PRs
- [x] Compare the 53 non-main branches against main for ahead/behind and changed-path inventory
- [x] Identify the largest supersets and overlapping v2/v3 code
- [x] Create an isolated integration branch; retain old main and old feature branches
- [x] Verify all combined **offline** tests, previews, and sound/video contracts (four green PR checks; 391 unit tests, 13 skipped)
- [x] Integrate optional Elizabeth Bones voice scripts and guarded manual Kaggle shorts / character factory workflows (voice listening not yet verified)
- [ ] Verify private artifact handoffs with owner consent; no quota spending by default
- [x] Merge CI-verified integration to main via PR #51 (squash commit `31eb2ff915a834b419de2f5b9345921fa7acc17f`)
- [ ] Confirm post-merge equivalence/backups before any pruning

## Post-merge notes

Merged PR #51 successfully into `main` on 2026-11-09. The integration branch and merged main commit have identical tree SHA `c71daea857d4e17e5af954ce3551bb8764f195f6`. No historic branch was deleted. GitHub Actions main-push verification and external private asset handoff remain separate from the four green PR checks; no live GPU inference, finished STORY GLITCH episode, or published media is claimed.

