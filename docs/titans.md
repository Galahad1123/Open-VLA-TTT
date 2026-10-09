# Evo1-GR00T VLM 中的 Titans 风格记忆

MAC、MAG 和 MAL 是独立注册的 LeRobot policy 类型（`mac`、`mag`、`mal`）。它们复用预训练 InternVL/Qwen2 权重和现有的 `GrootActionHead`，TTT 作用于语言层的隐藏特征。本实现将三种记忆连接结构适配到 VLA，不代表复现了 Titans 的语言模型实验结果。

## 架构与公共组件

`models/mac/connection.py`：先读取旧记忆状态，将持久记忆和检索结果作为 K/V 前缀提供给 attention；再将 attention 输出写入记忆，读取更新后的状态，并通过门控融合 attention 输出。每个当前 token 对应一个检索槽位，未选中的槽位会被掩蔽。这里显式检索旧状态，没有用社区实现中可学习的片段末尾 token 代替这一过程。

`models/mag/connection.py`：原有 attention 与 memory 并行处理同一份归一化输入，再通过非线性乘法门控融合两路输出。

`models/mal/connection.py`：直接将 memory 读出结果作为 attention 输入，不额外加入 `X + gamma * memory(X)` 形式的 attention 前残差适配器。三种结构均保留相同的 block 外层残差和原有 FFN。MAC/MAG 的门控采用 `2*sigmoid(linear(RMSNorm(readout)))`，初始值接近 1；这是本实现的适配选择，并非论文规定的常数。

`models/titans/` 负责共享的 VLM 执行与输入输出逻辑。模型组合 InternVL embedder 和现有 DiT；policy 复用 Evo-1 的无状态批次适配方法，不继承其 forward 或 episode 生命周期。每种架构都有独立的配置、连接结构、policy 和 processor 工厂模块。

四种记忆架构共用 `ttt/` 中的两层 GeLU fast MLP、可学习初始权重 W0、Q/K/V 投影、KV-MSE 和可微更新。`read()` 不包含残差、输出门控或时钟推进；`write()` 独立于具体架构。原有 `AssociativeMemory.forward()` 保留 RoboTTT 的“先更新、后读取”门控残差及参数名称。fast MLP 沿用现有 RoboTTT 风格核心，并未实现 Titans 完整的残差、归一化和卷积记忆网络配方。

## 首版固定语义

- 默认在保留的 14 个语言层中选择索引为 `10,12` 的层，索引从 0 开始。这是初始研究配置，尚未通过实验确定为最优设置。
- 默认每次完整观测在每个选中层执行一次内循环更新。选中 token 参与带掩码的 KV-MSE，损失按选中 token 数和特征维数归一化。各样本、各层分别维护独立的快权重与动量。
- `memory_input=all` 选择当前观测的全部有效位置；`visual` 仅选择有效的 `<IMG_CONTEXT>` 位置。这一选择同时控制 Q/K/V、更新系数的输入，以及直接进行 memory 变换或门控的位置。VLM 仍接收完整 prompt。MAC 从视觉位置检索到的内容对所有有效 attention query 可见。视觉位置的隐藏状态可能已混合其他信息，因此该消融检验的是位置选择，不能据此认定记忆只包含图像信息。持久记忆与检索前缀的位置不作为数据写入记忆。
- `memory_granularity=token` 按顺序写入选中位置，每个位置写入后立即读取。一次决策仍只推进一次时间 RoPE。MAC 首次构造上下文时仍读取观测前的旧状态，随后按 token 顺序写入 attention 输出。这是面向 VLA 的 token 更新消融，并非完整的自回归语言模型协议。
- attention 在当前观测内部采用完整的因果注意力，不跨观测保留历史原始 token 的 K/V cache。历史观测信息全部通过快记忆传递。这与论文 MAG/MAL 沿序列使用滑动窗口的方式不同，配置中显式记录为 `attention_scope=observation_causal`。
- 有效前缀的 key/value 始终可见；当前 token 的有效 key 遵循因果掩码。保留原有 Qwen2 token 位置与 RoPE，额外前缀的 key 不施加 RoPE。持久记忆参数在各次观测间共享，部署时固定，与 episode 快权重分开维护。不计算前缀位置的 query 输出。
- 完成观测级写入后，记忆可能包含当前完整观测的信息。因此，即便 attention 使用因果掩码，较早位置读取该记忆也不满足严格的 token 自回归约束。模型不会接收未来观测，动作标签和带噪动作块均不进入 VLM 记忆。
- 每次决策仅计算一次 VLM，所有 DiT 去噪步骤复用同一份上下文。执行动作队列时不更新记忆。只有完整动作块生成成功后才提交候选状态；失败则不提交。`reset()` 清空记忆和动作队列。
- 同一次读写使用相同的决策位置；`update_count` 统计实际写入次数，`decision_index`/`position` 统计至少发生一次写入的观测次数。padding、缺失视角和没有选中 token 的样本不参与写入或衰减。
- 冻结的预训练模块仍向输入传播梯度。`memory_only` 冻结 VLM/DiT 参数，仅训练记忆连接模块；`joint` 使用配置中的基线微调开关。冻结模块保持 eval 模式。
- 显式使用非重入梯度检查点，对纯函数 `(hidden,state)->(hidden,state)` 进行重算；反向传播期间不提交可变的 episode 状态。绕过原有 Hugging Face 语言模型梯度检查点包装，视觉编码器仍可使用梯度检查点。
- attention 使用支持高阶微分的 eager/math 后端，实际配置为 `use_flash_attn=False`。未引入优化的 Titans/CUDA 内核。记忆计算在 autocast 外执行，可搭配 bf16 基线权重使用。CPU 测试尚不能确定实际 GPU 性能或长上下文峰值显存。

## 更新规则选择（包括 RoboTTT）

默认仍为 `memory_update_rule=gradient_descent`。选择 `momentum_decay` 后使用：

```
S_t = eta_t * S_(t-1) - theta_t * gradient
W_t = (1 - alpha_t) * W_(t-1) + S_t
```

先对输入特征进行带掩码的均值聚合，再通过零初始化的线性投影调制正值学习率，以及经 sigmoid 约束的动量、衰减系数。三者的初始值分别为 0.1、0.9 和 0.01。可通过独立开关完全关闭动量或衰减。关闭数据依赖后，仍保留可学习的标量系数；同时关闭动量、遗忘与数据依赖，并使用相同的学习率时，可退化为普通 GD。

禁止写入的样本同时保持快权重和动量不变。reset 将权重恢复为 W0、动量清零；detach 保留数值并截断两者的梯度路径。checkpoint 保存可学习的更新器参数，不保存 episode 的快权重、动量或时钟。缺少新增字段的旧 RoboTTT checkpoint 默认使用 GD，参数键名保持兼容。

RoboTTT 选择该更新器时，仍保留原有的累积去噪更新方式和 RoPE 时钟选项，不会改用 VLM 的更新时序。

## 训练与评测

使用新的输出目录和已经训练好的本地 `evo1_groot` checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv/bin/python \
  -m lerobot_policy_vla_ttt.training.train_titans \
  --architecture mag \
  --checkpoint /path/to/evo1_groot/pretrained_model \
  --dataset-root /path/to/libero --dataset-repo local/libero --benchmark libero \
  --output outputs/libero-mag-gd --device cuda \
  --memory-layers 10 12 --memory-input all --memory-granularity observation \
  --memory-update-rule gradient_descent --memory-training memory_only \
  --steps 5000 --batch-size 2 --sequence-length 8 --decision-stride 10 --tbptt-length 4
```

以上仅为起始配置，尚未验证其收敛效果。将 `mag` 替换为 `mac` 或 `mal` 可切换架构；使用 `--memory-input visual` 进行输入模态消融，使用 `--memory-update-rule momentum_decay` 对比更新规则。运行 RoboTTT 时，使用 `--architecture robottt` 并省略 VLM 专用选项。更多参数见 [TTT 配置说明](../configs/ttt/README.md)。

训练器复用现有序列数据集、TBPTT 流程、归一化及 processor 保存机制。每个采样窗口从 W0 开始；每个片段先完成反向传播，再释放计算图；每个完整窗口批次执行一次优化器更新。可通过 `--segment-transfer` 按片段搬运输入。已有 checkpoint 可配合 `--memory-training joint` 热启动下一训练阶段，但不会精确恢复优化器状态、随机数状态或数据读取位置。加载 checkpoint 时，如果改变架构、更新规则或输入范围，会明确报错。

评测使用现有 server，选择 `--implementation mac`、`mag` 或 `mal`。推理外层使用 `no_grad`，仅在快权重更新处局部启用求导。需要执行记忆写入时，不允许外层使用 `torch.inference_mode()`。本实现未保证通用上游评测入口已具备这种 TTT 梯度边界。

在已实例化模型上设置 `config.memory_enabled=False` 可严格旁路记忆，同时旁路持久记忆前缀。这用于基线回归验证，不允许在 `memory_only` 模式下训练该路径。仅保留 register 或将门控设为接近零，均不视为严格旁路。

## 参考资料与验证边界

- [Titans，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)：第 2–3 节、图 4、附录 D。
- [arXiv v1](https://arxiv.org/html/2501.00663v1)：公式 22 中 MAC 前缀顺序为 P/R/X。NeurIPS 版本公式中的 P/X/R 顺序与其掩码示意图及图注存在冲突；本实现显式采用可见的 P/R K/V 前缀。
- [社区参考实现](https://github.com/lucidrains/titans-pytorch/tree/1d40c445fa1794fb28582c721786482b5b683e47)：`neural_memory.py`、`memory_models.py`、`mac_transformer.py`（MIT 许可证），用于参考算法、状态和掩码处理；未直接迁入其源码，也不将其视为论文的精确复现。
- 当前安装的 Transformers 5.5.4 提供 InternVL/Qwen2 预训练模块、视觉特征注入辅助方法、旋转位置编码和分组 KV head 工具，采用 Apache-2.0 许可证。

CPU 测试覆盖手算更新、掩码、Qwen2 连接结构差异、外层梯度、梯度检查点重算、状态时钟、reset/detach、严格重载、服务动作队列，以及本地数据集 CLI 训练。小型随机初始化的真实 InternVL 测试使用实际的视觉编码器、投影器、Qwen2 和 DiT，仅替换 tokenizer 与权重加载过程，以避免下载资产。这些测试不证明机器人任务成功率、预训练模型收敛性、GPU 吞吐，也不证明完整复现了 Titans 的实验配方。

已完成的检查和因依赖资产而跳过的测试，见 [2026-10-08 实现验证记录](experiments/2026-10-08-titans-vlm-implementation.md)。
