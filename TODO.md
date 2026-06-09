# Project: VaultWares Studio

Plan of record: `docs/plans/plan-v1-remote-first-20260609.md` (M0–M6). Legacy phases below are kept for history.

## M0: Remote Execution Foundation

- [x] `StageRunner` abstraction (`vaultwares_studio/runners/base.py`): StageContext, StageResult, CancelToken, cost estimates
- [x] `LocalStageRunner`: Popen line-streamed output, cancel (psutil process-tree kill), timeout
- [x] `HfJobsStageRunner`: consent gate before any network call, dataset-repo artifact transport (`jobs/<id>/<stage>/{in,out}`), launch/poll/cancel (≥10 s poll), log fetch, output download
- [x] Manifest schema v2: `schema_version`, per-stage `placement`/`runner`/`params`/`cost`, `spend_ledger`, v1 migration on load
- [x] `record_spend()` ledger helper
- [x] `pipeline._run_command` delegates to LocalStageRunner (live logs + cancel)
- [x] Stage placement defaults (reconstruction → remote; honored from M1)
- [x] Worker image v1: `docker/worker/Dockerfile` (COLMAP + nerfstudio + hub) + `vw_stage.py` + `tools/build_worker_image.ps1`
- [x] GUI Settings: HF token (OS keyring), artifact repo, default flavor, cost-confirm dialog, echo-test button
- [x] Tests: runner streaming/cancel/timeout, cost-denial, config round-trip, manifest v2 migration, spend ledger (19 passing)
- [ ] Live verification: echo job round-trip on `cpu-basic` (needs HF token; run from Settings → "Run Echo Test Job")
- [ ] Build & push `vw-studio-worker:0.1` to Docker Hub (needed before M1 remote reconstruction)

## M1: Real Reconstruction, Remote — next

- [ ] Worker recon entrypoint: ns-process-data (sequential) → ns-train splatfacto → ns-export gaussian-splat
- [ ] Quality presets (Draft/Standard/High) with flavor + cost estimates
- [ ] Full-attribute splat PLY (`splat_io.py`) — stop flattening through Open3D
- [ ] PLY→USD: 26.03 schema probe, UsdGeomPoints+primvars fallback
- [ ] Remote-stage wiring for `reconstruction` (placement honored)

## M2+: see plan file (viewport/staging, Robot Lab, Cosmos, packaging)

---

### Legacy phases (pre-plan history)

Phase 0 Infrastructure — complete. Phase 1 Capture & Reconstruction — frames/fallback PLY done; COLMAP SfM + gsplat training move to M1 (remote). Phase 2 USD — composition + smoke tests done; PLY→USD 26.03 moves to M1. Phase 3 Isaac Sim — deferred to M6. Phase 4 Cosmos — moves to M4.
