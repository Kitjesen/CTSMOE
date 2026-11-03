# V33 continuous terrain integration — 2026-09-24

User requested8192 environments and integration of all six newly designed continuous/asymmetric terrain families. Existing training stopped safely after44319 completed updates (model44318) for migration; expected STOP is reported as technical_failure by the old driver, not a behavior rejection. Existing target46194 is unchanged:1875 updates remain.

## Before migration

8192 environments,11.70s last iteration, overall training mean5.8895. Type means: stairs up6.1195,stairs down6.0314,obstacles6.0330,wave5.8770,uphill5.2256,downhill5.7963,rough slope5.8020,flat5.9878. These are curriculum statistics, not held-out success rates or posture certification. Hip actual peaks still approach0.411rad; landing cumulative maxima must not be interpreted as ordinary landing forces. Uphill remains the lowest mean. Disk88GB free.

## Migration contract

Retain80% original terrain distribution; allocate20% to continuous curbs5%, variable blocks4%, ramp-stairs-ramp4%, split-height2.5%, cross-slope bumps2%, split ramp/stair2.5%. No pit/high-platform environment. Teacher/student proportions remain75/25. Preserve retained environment types and levels; only newly assigned routes start level0. Model and both optimizer states unchanged by migration. Atlas40x22 cells remains9x9m, old columns0..19 unchanged; new20..25, unused pit26..39. Verify440 original cell geometry hashes in physical qualification.

New route course0..9: progress past final obstacle with whole body and root finish3.3m, mean velocity errors<=0.30/0.20m/s and0.20rad/s, no physical failure or cell boundary exit. Full-body boundary protection. New route commands forward0.4–0.9m/s,5s resampling, initial yaw±5degrees; other terrain commands unchanged. New routes use3cm horizontal chamfers at risers, not exact vertical faces. Max curbs/blocks40cm, ramp targets45degrees, ramp-stair steps30cm, split-height20cm, cross-slope14degrees with15cm bumps. Web long-route prototypes are not identical to compact training tiles.

## Verification and continuation

Local and remote9 focused tests passed; geometry88 tests passed locally; changed Python ruff passed. Initial physical qualification failed BEFORE optimization because a dynamically attached SubTerrainBaseCfg family attribute did not survive copying. Preserve first attempt; fix via declared/preserved family binding and requalify. No successful physics claim yet.

Planned1512 control steps (30.24seconds) zero optimizer updates across all60 family/level combinations and resets; validate finite trajectories,440 old hashes, model and both optimizers unchanged. Then16 integration updates and3-step assessment plumbing check, then remaining1859 updates, final1250-step matched assessment with all existing pose/tracking/actuator metrics plus route terminal events. The3-step check is not behavior validation. No real robot deployment. Do not duplicate GPU processes or restart old V32 driver.

## Physical qualification progress

Attempt02 exposed Isaac's strict manager signature validation: **kwargs is interpreted as a named mandatory argument. Both wrappers now use explicit original/continuous_original parameters matching the frozen manager configuration. Six top-level terrain functions preserve family identity through config copies. Attempt03 resumed44319 successfully, verified440 old cell mesh/origin witnesses and entered physical stepping. Focused tests now16 total (14runtime+2migration); remote tests passed. Both previous attempts performed0 optimizer updates.

Migration counts: new1639,retained6553. New curbs409,blocks328,ramp-stairs328,split-height205,cross-bumps164,split-route205. Original counts:flat1311,upstairs917,downstairs917,wave787,uphill656,downhill655,roughslope655,boxes655. Teacher6144/student2048. All newlevels0 for training (qualification temporarily coversalllevels without saving).

8192 continuation before migration ran163 updates; medianlast32seconds11.5067. First32 terrainmean5.8217,last32mean5.8634. Small training-statistic change, not proof of significant capability improvement.

## Qualified and training

Attempt03 intentionally stopped before any optimization: audit found Isaac command reset runs before episode_length is cleared. Success/boundary/tracking state is now cleared in the explicit command reset, never during periodic resampling. Added actual nonzero-episode-length reset regression and a per-step real-reset state-clear assertion in qualification.

Attempt04 passed1512 steps.440 original cells verified. Family resets[744,597,625,432,342,398]; all60 family/level combinations observed. Model/PPO/student optimizer hashes unchanged. Receipts:QUALIFICATION_FINALIZED.json,FROZEN.json,MIGRATION_FINAL.json. Local+remote16 runtime/migration tests passed. Final assessor preserves V31 old metrics and5variant route events, adds six-route teacher/student outcomes and balanced0..9 fixed tests. Separate2 teacher-mean tests passed locally (initial helper test failures fixed and rerun, not accepted on compile alone).

Observed actual optimization to44328 (9newupdates),10.347s latestiteration,8192environments. Original-terrain weightedmean5.8634. Newroutelevel0 atthispoint isexpected: first9updates cover4.32s, shortofwhole-routefinish. This confirms launch, not skill acquisition. Stage attempt04 supervisor196221; training197075 atobservation; use actualSTATUS/PID onfollowup. Target46194 unchanged.

Integration16 completed44335;3-step zero-update assessment plumbing check passed (not behavior acceptance). Supervisor transitioned automatically to remaining1859 updates, training PID197985. Latest16thiteration11.1257s; original weightedmean5.8598. Terrain means up6.1505/down5.9368,wave5.8539,uphill5.1936,downhill5.8046,rough5.7359,obstacles6.0412,flat5.9382. Remote assessor's two focused teacher-mean tests passed. Local112code hashes exactlymatch remoteFROZEN. Full assessment after46194; do not claim newterrain/posture capabilityvalidated fromthese16updates.
