# RoboMME observation-memory 评测

T2MEM/AQMem 等使用 `HalfExecutionLifecycle` 的模型支持 demo 前缀。
旧 memoryless baseline 未声明此能力，继续只消费 reset 的最后一帧。

## 协议

1. simulator reset 返回示范帧序列及一个执行初始化帧。依据锁定的 RoboMME wrapper（c85e060），最后一帧是执行观测，不加入 demo。
2. policy reset 清空旧 episode 的 fast memory、动作缓存和执行时钟。
3. 客户端根据服务器声明的 `demo_stride`（当前 E/2=5）逐帧发送选中的 demo，调用独立 `demo` RPC。每条消息只有双相机、state 和 task goal，无动作、成功标签或 privileged subgoal。
4. 服务端只提交观察记忆更新，不生成动作、不消耗 flow noise、不推进执行时钟。state 字段经过普通预处理，但记忆写入仍只消费 VL 特征。
5. 首个执行观测从 step=0 开始，保留 demo 记忆；之后按既有 half_execution 协议运行。进入执行后拒绝追加 demo。
6. 单条 episode 出错就停止，不重试或把基础设施错误记为策略失败。示范生成时间计入 elapsed_s，demo RPC 次数另记为 demo_writes。

服务器命令示例（路径按设备调整）：

```bash
PYTHONPATH=src:packages/eval-client/src .venv/bin/python \
  -m lerobot_policy_vla_ttt.evaluation.server \
  --implementation t2mem --benchmark robomme \
  --checkpoint /path/to/step-00002000/pretrained_model \
  --vlm-path /path/to/InternVL3-1B-hf \
  --n-action-steps 10 --host 127.0.0.1 --port 8795
```

客户端使用独立 RoboMME 环境（需要相同版本 eval-client）：

```bash
PYTHONPATH=packages/eval-client/src environments/robomme/.venv/bin/python \
  -m vla_ttt_eval.runner --benchmark robomme --tasks all \
  --split test --episodes 50 --start-episode 0 --seed 1000 \
  --max-steps 1300 --server ws://127.0.0.1:8795 \
  --output outputs/eval/robomme-t2mem-new-run.jsonl
```

`1300` 为 MME-VLA 名义预算；实际使用官方 wrapper 的 remaining budget。输出独占创建，拒绝覆盖。仅有 pretrained_model 即可评测，不需要 optimizer/training.pt。

## 2026-10-15 实验

- 输入：`outputs/robomme-t2mem-seed1000-run01/step-00002000/pretrained_model`。
- 固定最终 checkpoint；中间 500/1000/1500 权重不用于 test 择优。
- 正式协议：test，16 tasks × 50 官方 episode，policy seed=1000+episode，H=20/E=10，32 次去噪，demo stride=5。
- 完整 CPU 回归：255 passed、7 skipped；随后将边界测试扩展为无 demo、执行首帧恰好落在 stride 网格和非网格三种情况，定向测试 6 passed。
- 真机烟雾测试：RouteStick val episode0，20 次 demo 写入后执行 3 个控制步；名义预算2，对应 wrapper remaining budget3。正常以 timeout 结束，测试退出码0。此结果只验证链路，不是能力评测。
- 实验脚本、日志、源文件哈希、逐 episode JSONL 和汇总位于 `artifacts/experiments/t2mem-robomme-run01-step2000-20261015/`；烟雾测试与正式 test 分开存放。
- 正式脚本逐任务独立环境进程，全部完成后汇总；source hash 变化、server退出或client异常立即停止。评测从独立 worktree 加载代码，避免其他 master 开发改变正在运行的实验版本。
