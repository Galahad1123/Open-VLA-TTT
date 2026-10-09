# Evo1-GR00T：与本地 Evo-1 对齐的 LIBERO 训练

## 范围与结论

工作树：`evo1-plugin-foundation`。对照旧实验的实际 `train_config.json`，
而不是官方大 batch recipe。保持新动作专家结构不变，不加载旧 Evo-1 策略权重。

训练使用同一个 `vla-ttt-train` / LeRobot trainer；数据、processor、MIN_MAX
归一化、action padding/loss mask、阶段冻结、保存/恢复接口沿用本地 Evo-1。
推理共用 `Evo1Inference`、WebSocket 协议、动作队列和 reset，只将
`--implementation local` 换成 `--implementation evo1_groot`。
保持 chunk=50、执行前10步、32次积分、连续 gripper，不改变 LIBERO 客户端。

调整 LIBERO preset：dropout 0→0.2、weight decay 1e-5→1e-3、warmup 300→1000、
关闭 FlashAttention。不要用 `baselines.train` 的简化默认值做本次正式训练：
它会重设 warmup、worker 数等。下面直接使用正式 CLI 并显式覆盖参数。

| 设置 | Stage 1 | Stage 2 |
| --- | --- | --- |
| 初始化 | 本地通用预训练 VLM + 随机动作头 | 本次 Stage 1 的005000 |
| VLM | 冻结，bf16权重 | 解冻，float32权重 |
| 前向 | bf16 autocast | bf16 autocast |
| 步数 / batch | 5000 / 8 | 80000 / 4 |
| 保存间隔 | 2500 | 10000 |

共同设置：seed=1000，AdamW lr=1e-5、wd=1e-3、clip=1，warmup=1000后cosine，
图像增强开启、448×448、4 workers、完整数据、无held-out loss。
Stage 2 `resume=false`：加载模型/processor，但新建优化器和调度器。
新策略用显式 `dtype=bfloat16` 驱动 Accelerator；旧策略根据 `use_amp` 回退，
外层 scaler 路径有差异，不能称为逐位数值等价实验。

## 本地资源与离线约束

- LIBERO：`/home/hello/lizk/datasets/libero`，实际离线读取成功，1693 episodes、
  273465 frames；解码首帧得到两路 `[3,256,256]` 图像、8维state、7维action。
- VLM：下方固定 snapshot；本地 safetensors / tokenizer / processor 文件存在，
  GPU测试确实从此路径加载。没有下载模型或数据。
- 直接用已安装 `.venv/bin/vla-ttt-train`，不执行 uv sync / pip install。
- 三个 OFFLINE 环境变量阻止 HF远程回退；缺失资产应报错，而不是下载。

## 正式命令

以下 Bash 块依次执行两个阶段，Stage 1 失败不会继续 Stage 2。
在 tmux 等持久终端中运行；不会自动恢复中断任务。
输出目录若已存在，应先检查其内容，使用新实验名或单独设计恢复命令，不删除旧结果。
`policy.type` 直接新建策略，因此无需 `initialize_groot`，也不使用 smoke checkpoint。

```bash
cd /home/hello/lizk/vlas/vla-ttt/.worktrees/evo1-plugin-foundation
(
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLM_PATH=/home/hello/.cache/huggingface/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171
RUN=libero-evo1-groot-scratch-seed1000
test -f "$VLM_PATH/model.safetensors"
test -f /home/hello/lizk/datasets/libero/meta/info.json
test ! -e "outputs/${RUN}-stage1"
test ! -e "outputs/${RUN}-stage2"
mkdir -p outputs/logs
for stage in stage1 stage2; do
  if [ "$stage" = stage1 ]; then
    source_args=(--policy.type=evo1_groot)
    steps=5000; batch=8; vlm_dtype=bfloat16; save_freq=2500
  else
    source_args=("--policy.path=outputs/${RUN}-stage1/checkpoints/005000/pretrained_model")
    steps=80000; batch=4; vlm_dtype=float32; save_freq=10000
  fi
  .venv/bin/vla-ttt-train "${source_args[@]}" \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=/home/hello/lizk/datasets/libero \
    --dataset.revision=86958911c0f959db2bbbdb107eb3e17c5f9c798e \
    --dataset.video_backend=pyav --dataset.return_uint8=true \
    --dataset.image_transforms.enable=true --dataset.use_imagenet_stats=false \
    --dataset.eval_split=0.0 \
    --policy.vlm_model_name="$VLM_PATH" --policy.vlm_dtype="$vlm_dtype" \
    --policy.dtype=bfloat16 --policy.use_amp=true --policy.device=cuda \
    --policy.training_stage="$stage" --policy.apply_training_stage_defaults=true \
    --policy.use_flash_attn=false --policy.enable_gradient_checkpointing=true \
    --policy.gradient_checkpointing_use_reentrant=false \
    --policy.image_resolution='[448,448]' --policy.chunk_size=50 --policy.n_action_steps=10 \
    --policy.max_state_dim=24 --policy.max_action_dim=24 --policy.task_field=task \
    --policy.num_inference_timesteps=32 --policy.binarize_gripper=false \
    --policy.dropout=0.2 --policy.optimizer_lr=1e-5 --policy.optimizer_weight_decay=1e-3 \
    --policy.optimizer_grad_clip_norm=1.0 --policy.scheduler_warmup_steps=1000 \
    --policy.push_to_hub=false --use_policy_training_preset=true \
    --batch_size="$batch" --steps="$steps" --resume=false \
    --save_checkpoint=true --save_checkpoint_to_hub=false --save_freq="$save_freq" \
    --log_freq=50 --env_eval_freq=0 --num_workers=4 --prefetch_factor=2 \
    --persistent_workers=true --seed=1000 --cudnn_deterministic=true \
    --wandb.enable=false --output_dir="outputs/${RUN}-${stage}" \
    2>&1 | tee "outputs/logs/${RUN}-${stage}.log"
done
)
```

训练后 policy 服务同样使用离线环境：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/vla-ttt-serve \
  --checkpoint outputs/libero-evo1-groot-scratch-seed1000-stage2/checkpoints/080000/pretrained_model \
  --implementation evo1_groot --benchmark libero --device cuda \
  --n-action-steps 10 --host 127.0.0.1 --port 8780
```

LIBERO client 继续在 `environments/libero/.venv` 中运行；正式能力比较应保持
旧版40任务×20初态的同一评测协议。本轮不运行完整闭环benchmark或长训练。

## 本次验证

- 新增 preset 回归测试先因 wd=1e-5 失败，调整后通过。
- 完整 `pytest tests -q`：69 passed、7 skipped。跳过的是未配置资产的 opt-in
  集成测试；首次沙箱执行中 PATH缺失导致4项CLI失败，禁止socket导致1项失败，
  使用虚拟环境PATH并允许localhost后上述5项均通过。
- 单独设置本地 VLM 路径的GPU测试：3 passed，覆盖 Stage 1 bf16权重、
  Stage 2 bf16权重及本次采用的Stage 2 float32权重；包含梯度有限性、阶段冻结、
  实际Accelerator/update_policy更新、动作padding和可重复采样。
  GPU测试使用合成batch，不是完整真实数据训练或LIBERO闭环评测。
- 从本文提取训练CLI参数，经实际LeRobot parser和validate解析，两阶段的dataset、
  steps、batch、seed、workers、优化器、scheduler、保存/日志频率与旧正式
  train_config.json逐项相等；Stage 1 pretrained_path=None，Stage 2加载其来源配置
  后正确解冻。Stage 2解析测试使用临时保存的Stage 1配置，不伪造正式权重。
- formatter、lint及git diff --check通过。本轮未修改模型架构、未提交/推送，
  未启动正式训练，未覆盖已有训练产物。
