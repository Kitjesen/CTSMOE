# 三处兼容性说明核查

2026-09-21。只处理用户hook指出的三个位置；没有修改训练逻辑、执行器、停止规则或hook配置。

## 修改
- NVIDIA SimulationApp诊断源码副本：去掉“RuntimeError证明Tracy插件未加载”的不成立断言，明确保留已有可选profiler清理异常边界，使后续应用关闭语句继续执行。异常原因不作推断。
- python-dateutil：把含糊的临时处理描述改为module-level gettz接口与惰性共享ZoneInfoFile缓存的兼容说明。get_zonefile_instance的函数属性缓存与弃用gettz的_CLASS_ZONE_INSTANCE是两个独立缓存，未将二者合并或修改。
- Matplotlib：把临时处理描述改为具体Formatter.format_ticks(levels)与legacy callable(direction, factor, levels)两种签名的分派说明。

## 来源证据
使用本次编辑前.before快照，在内存中移除先前已有的Compatibility rationale注释块：

|依赖|恢复的原始字节数|SHA256 base64url（与安装包RECORD一致）|
|---|---:|---|
|python-dateutil 2.9.0.post0 / zoneinfo/__init__.py|5889|KYg0pthCMjcp5MXSEiBJn3nMjZeNZav7rlJw5-tz1S4|
|Matplotlib 3.11.2 / axisartist/grid_finder.py|12734|oiiDAXGt9OUa_geD1Y9zWaLkok6ABZkT-ZP6kFyLawI|

两者为上游既有兼容实现。安装包RECORD未改动。本次更新仅进一步澄清注释/文档；修改后的源文件hash当然与原RECORD不同，因此保留before快照及verification.json作为审计证据。

## 验证
- 三份文件去除docstring后的AST前后一致：没有改变任何可执行语句或异常处理边界。
- 三份文件Python语法编译通过。
- Ruff前后对比无新增诊断；原SimulationApp和Matplotlib各有2项既有诊断，dateutil为0。未把既有诊断算作本次引入。
- 没有导入或执行NVIDIA模拟器，没有安装/升级依赖，没有启动训练。
- V14仍为32131轮行为验收后停止；本次文档修正不改变此前能力结论。

具体路径及修改前后SHA256见[verification.json](verification.json)。
