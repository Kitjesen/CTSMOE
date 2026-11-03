# 可追溯性与复现边界
当前112个顶层Python源码与V33 FROZEN.json哈希逐项比较。服务器文件快照有独立清单，Git归档不改变训练服务器文件。
历史 code 目录、配置、收据以原路径相对结构保存，不重写为假造的Git提交日期。
provenance/LOCAL_ARCHIVE.json与SERVER_ARCHIVE.json保存采集清单；PUBLICATION_MANIFEST.json是最终实际发布文件与哈希。连接辅助脚本如含凭据会排除并记录，不能使用采集清单宣称每一项都已发布。
权重文件.pt/.onnx未纳入Git，原存储继续保留；CHECKPOINT_REFERENCES.json是从收据提取的引用，不证明路径现在可读，也不是完整权重备份。
本次主要是归档发布，做哈希、Python语法与选定CPU回归检查；没有在新目录另启动GPU训练，不能宣称全新机器端到端已复现。
