# Evo-1 固定快照来源

导入日期：2026-10-01。来源：安装发行版 `lerobot==0.6.1` 中的
`lerobot/policies/evo1/`，目标：本包的 `evo1/`，七个文件原名保留。
原始版权头与 Apache-2.0 许可保留。本地版本独立演化，不自动同步或持续追踪上游。

原文件 SHA256（仅记录导入来源，不作为持续同步要求）：

```text
acb52744fed3005f999ef1d15846398b5246ac35458ea374200df3a57f0a2c43  __init__.py
c87a6d0de041440245c89e974bcdd32e0985d54fb912b5b538bbe325bd222658  configuration_evo1.py
572073a68e71fd3e27acde538282cf9885542a8351f8322b00461200684aadd7  evo1_model.py
e77ca0a15bb5a6b91709af630afeb124feabc8dbd1b297b948550c4f6d802ab3  flow_matching.py
c523d814c2effc432c961ef7d1bf786b7a8d3968ed6809e4637aaac7204a56b4  internvl3_embedder.py
f34058d9da4e11fc46b2cdb1b8494bb997e0cb3aa30be74a4e97e0640eee35f9  modeling_evo1.py
0d42f73ef611486ff5d3d811fd0cefe0e1885536f4a627a1da1ce609634d9d24  processor_evo1.py
```

导入时只做以下源码调整：
- configuration_evo1.py：移除原生 `evo1` config 注册装饰器；RTC import 改为绝对路径。
- modeling_evo1.py：RTC import 改为绝对路径。
- processor_evo1.py：三个专属 step 注册名增加 `vla_ttt_` 前缀。
- 其他四个文件保持原文；不对固定快照执行自动格式化。

包外适配：
- VLATTTConfig / VLATTTPolicy 改为继承本地类；原生配置导入只用于 checkpoint 转换。
- checkpoint warm-start 以原生 loader 严格读取源权重，再严格复制到本地模型；
  后续加载转换 checkpoint、训练和推理使用本地模型。
- processor 恢复先将旧注册名对应的 steps 本地化，再执行本地 reconciliation。
- vla-ttt-train/eval/rollout 复用 LeRobot runner，通过有作用域的 factory 引用适配
  解决 0.6.1 对上游 Evo1Config 的硬编码分支。

