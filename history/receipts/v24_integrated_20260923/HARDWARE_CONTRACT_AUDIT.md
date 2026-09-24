# 15 厘米实机失败：已有证据与待核实项

2026-09-23。用户报告过 15 厘米台阶失败、髋持续内夹和偶发腿部卡住，但未提供当次电机故障码、实机关节时程或实际加载的 ONNX 哈希。此前称“15 厘米实机报错”的具体错误已确认属于不当表述；现不能识别是什么驱动错误，更不能把仿真内夹直接认定为那次故障的原因。

已核对的 V21 ONNX 候选包在 `../stair_coverage_v21_20260922/stair_release_validation/Thunder_CTSMoE_V21_34823_ONNX_CANDIDATE_NOT_RELEASED/`。其 README、`runtime_contract.json`、`reference_io.py` 要求历史观测 `[1,530]` + 单帧 `[1,53]`，输出 `[1,16]` 原始动作；12 腿关节顺序为 FR/FL/RR/RL 各 hip、thigh、calf，另四轮 FR/FL/RR/RL。策略 50 Hz、物理 200 Hz，腿动作缩放 0.25、轮 5，腿 PD 120/8、轮 0/1，腿/轮仿真峰值扭矩 120/17。V21 默认髋为 FR−0.20、FL+0.20、RR+0.20、RL−0.20 rad，髋执行目标软限 ±0.369，腿目标每 20 ms 最多变化 0.24 rad。该包标记 HOLD_BEHAVIOR_FAILED、hardware_deployed=false；固定楼梯矩阵 97/144，不是随机成功率。

V23 将镜像髋默认角改为 ±0.12 rad，因而相对角观测和动作零点一起改变。本地没有查到 V23 ONNX。若实际把旧 V21 ONNX 与新 V23 默认角部署参数混用，会有 0.08 rad 的零点差异；该静态差异方向并不能单独解释持续内夹。另一套 `D:/inovxio/brain/go2-mid360-v6b-publish/sim/packages/controllers/doso/thunder_v4/locomotion/controller.package.yaml` 为旧平地模型，仅 53 维输入、髋缩放 0.125、腿 PD 90/6.93，不兼容 CTS 的直接加载契约，但没有证据表明它用于用户那次实机测试。

下次实机复测前必须独立核对实际加载 ONNX SHA、部署程序和输入历史顺序、四髋镜像符号、16 个输出映射、默认角、动作缩放、PD、延迟/限幅与真机保护值，并取得当次故障码及原始请求/限位后目标/实际角/扭矩时程。当前 V24 工作仅限仿真与离线验证，不放行硬件。
