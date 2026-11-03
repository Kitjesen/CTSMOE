# 34823 continuous-stairs and ONNX verification

User requested continuous-stair validation and an ONNX artifact for subsequent user-operated deployment, conditional on acceptable results. No robot is controlled by this task.

Frozen source: V21 model_34822.pt (34823 completed), SHA256 521a31d1acfc7e0906f4ac3260ee30f161af5b1c18278367d0c0045b7afe3fa5. No optimizer updates, training restart, reward changes or limit relaxation.

Run 144 fixed first episodes: upstairs/downstairs x levels 3/6/9 x vx 0.3/0.6/1.2 m/s x direct/2-second-settle x continue/stop on the physically verified outer landing x nominal/perturbed. Perturbed cases retain observation noise and add deterministic +/-5 degree yaw and +/-0.10 m lateral start offsets. This is a fixed robustness matrix, not a random success-rate estimate. No heading feedback is added. Retain exact action/torque/speed constraints and 25-second budget. Earlier original-stairs tests remain reported separately because the capped landing reduces the outer stair count.

Use the ONNX model in the actual simulation action loop. Compare every output with the checkpoint student on identical observations (atol=1e-4, rtol=1e-4). Capture all requested/limited/actual joints, torque/speed substep peaks, contacts, platform support, tracking and startup/transition posture. Record raycast-measured stair profiles and exact step rises. Hash sources before/after. Serialize evaluation under existing lock, reject occupied GPU.

Complete traversal requires the full robot collision hull and >=3 supported wheels on the actual terminal platform, no physical failure, and maximum lateral excursion <=0.5 m. Stop trials additionally require 2 seconds of support and linear speed <0.15 m/s and angular speed norm <0.3 rad/s. Report each failed dimension, including startup single-joint/RMS peaks and sustained/transient actuator limit excursions. Do not treat four-meter travel or support alone as successful stopping. No broad deployment release if any condition fails. Export conversion correctness and behavior qualification are separate.
