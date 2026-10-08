# Evo-1 / evo1_groot LIBERO 基线验收

日期：2026-10-08。

## 结论与范围

已验收本地 Evo-1、evo1_groot 的既有 LIBERO 两阶段训练产物、server-client 评测结果，以及独立基线快照的 CPU 回归。无需等待正在进行的 Titans 开发。

本结论不等于当前 foundation 脏工作树整体验收，不等于阶段 A 完成，也不宣称已完成同权重原生/本地完整 LIBERO 对照。没有合并、提交、修改 Titans 源码、安装依赖或启动训练；没有占用 GPU。

## 独立验收对象

- 基础提交：`0f248226d7f29b75c883d52ced06731dc99543c7`。
- 导出目录：`/tmp/vla-baseline-acceptance-CCSyxT`，通过 git archive 导出，不创建或修改 Git 分支。
- 加入当前已验证的 `configs/baselines/evo1_groot/libero/config.json`：SHA256 `7d65ea3732d808a9c6d76c53c3cd53cfdeb7713d298034efa3f34ffff254bd35`。
- 加入对应 `tests/unit/evo1_groot/test_libero_recipe.py`：SHA256 `29ce3963edd82df67254de479e9d8cbe2578b178a0acef964fa0ac2343eab2aa`。
- 相对该提交，当前本地 Evo-1、evo1_groot 模型目录、eval-client 及 cli.py 无差异；共享入口新增的记忆模型改动不在此次验收范围。
- 这是一份明确标识的候选基线快照，不声称恢复了历史训练时完整的未提交源码状态。临时目录不是长期发布存档。

## 本次实际验证

复用现有虚拟环境作为依赖来源，不修改其 editable 安装。通过 PYTHONPATH 指定临时目录的 src 和 packages/eval-client/src，并实际核对两个包的导入路径均位于该目录。CUDA_VISIBLE_DEVICES 为空，HF_HUB_OFFLINE 和 TRANSFORMERS_OFFLINE 为 1。

在临时目录运行 `python -m pytest tests -q -p no:cacheprovider`，允许 localhost socket：**69 passed, 6 skipped in 21.79s**。跳过的集成测试依赖显式配置的模型资产或 GPU；本次不将历史 GPU 实验证据冒充新跑的 GPU 测试。

工作树实验根目录：`/home/hello/lizk/vlas/vla-ttt/.worktrees/evo1-plugin-foundation`。

| 产物 | 审计结果 |
| --- | --- |
| 本地 Evo-1 两阶段训练 | 5000、80000 检查点文件齐备，两阶段日志都有 End of training |
| evo1_groot 两阶段训练 | 5000、80000 检查点文件齐备，两阶段日志都有 End of training |
| 本地 Evo-1 LIBERO | 40任务×20初态，520/800成功（65.0%），错误0 |
| evo1_groot LIBERO | 40任务×20初态，539/800成功（67.375%），错误0 |

逐任务重新调用结果校验器，核对 episode ID 0–19、policy seed=1000+episode、各任务使用同一 checkpoint 身份。重新计算评测头中记录的模型、配置、processor 及训练配置文件哈希，均与保留文件一致。

- 本地结果：`outputs/eval/libero-stage2-080000-20ep-20261005/`。
- GROOT结果：`outputs/eval/libero-evo1-groot-stage2-080000-20ep-20261008/`。
- 本地权重 SHA256：`94040fb0dd2c7974619839b86b4acb88df1f8a27280cc0b07ab4861cc1c6891e`。
- GROOT权重 SHA256：`3fac9ab39b3096809c36a291f61aa959f63062fd18d236c882218edcf91ea0c6`。

## 尚未验收与后续边界

1. 同一80k权重的原生/本地完整 LIBERO 对照仍待完成；此前原生服务被正在开发的 Titans 入口阻塞，不能把该次启动当作成功评测。
2. RoboMME 正式训练与评测仍待完成，本次不启动训练。
3. 本报告仅批准上述范围的技术验收，不自动执行 master 合并。落地主分支时应保留当前并行开发改动，单独整理基线提交及文档，不能直接合并整个脏工作树。
4. 报告放在主仓库 docs/experiments，避免修改另一会话可能正在编辑的 foundation 路线图；主仓库源码与分支未改变。

## 随后经用户授权落到本地 master

用户在验收后明确授权落地主分支。本地 master 快进到上述基线提交，并补入验收过的 LIBERO 配方、回归测试及历史实验文档；不纳入 `8580b54` RoboTTT 提交和 foundation 的未提交开发改动。上文“不合并”描述的是验收时的操作边界，不代表此次授权后的状态。

主目录 `/home/hello/lizk/vlas/vla-ttt` 已通过 `uv sync --frozen --offline` 创建独立 `.venv`。实际核对 policy 与 eval-client 导入路径均指向主目录，未修改 foundation 的 Python 环境。后续 policy 命令使用主目录 `.venv/bin/`。

旧文档与命令中的 worktree 路径保留为历史实验记录；数据、权重、日志、评测输出未移动，仍在原位置。主目录运行后续评测应显式使用旧产物的绝对路径，并使用新的输出目录，不能假定 master 的 `outputs/` 已含旧权重。仿真环境仍按各自 requirements/lockfile 隔离，后续可在主目录创建，不能直接迁移旧 venv。

没有推送远端，没有运行正式训练或 GPU 评测。原生/本地完整 LIBERO 对照仍是下一项待办。

落地后在主目录新 `.venv` 中重新执行离线 CPU 全套回归：**69 passed, 6 skipped in 27.27s**；`git diff --check` 通过。
