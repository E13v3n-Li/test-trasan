# Sanitizer 无害黑盒可追踪水印：代码说明

本包基于 HKU-TASR/Sanitizer 的真实源代码修改，基准提交见 `UPSTREAM.json`。
保留原目录、MIT 许可证和原仓库的两个预训练权重；没有向作者的 GitHub 仓库推送修改。

这是可运行的研究实现，不是已完成的研究结论。第一版正式接入的是 **CIFAR-10 + 原仓库 ResNet18**。
不包含图联邦改造，不宣称已复现论文数字，也不宣称人工图像天然安全。

## 1. 实现范围与基线

| 组件 | 实现状态 |
| --- | --- |
| 原 Sanitizer 训练与净化 | 保留，并修复下述影响训练/导出的实现问题 |
| D：直接无害追踪基线 | 独立副本、距离受限码本、人工查询、正常任务与自然输入安全损失 |
| E：候选鲁棒改进 | 可选的一步干净 SGD 展开后的指纹保持损失；包含二阶梯度，默认关闭 |
| 黑盒解码 | 本地模型标签、HTTP 标签接口、离线标签 JSON 三种入口 |
| 阈值校准 | 得分 + 第一/第二名间隔；需要全部注册身份和未知模型；允许拒绝判断 |
| 测试 | 匹配矩阵 CSV/JSON、正确/错误/拒绝归属、配对安全指标、可选贡献验证 |
| 修改测试 | 干净数据微调、全局 L1 非结构剪枝、模拟 int8 权重舍入 |
| 未实现 | TraMark 复现、蒸馏/合谋保证、生产级 API 服务、真正后端 int8 部署量化 |

`--robust-weight 0` 是 D；例如 `--robust-weight 0.2` 是 E 的候选设置。
`--safety-weight 0` 可作安全约束消融 C。原贡献保留通过 `--contribution-weight` 控制，默认关闭。
代码没有把“无害化损失”直接等同于无害证明，也没有把 E 的存在当作创新已经成立。

所有副本均从同一个净化模型独立初始化；新增指纹不参与后续联邦聚合。
因此只能追踪最终分发副本，不能追踪此前保存的公共中间版本。

## 2. 环境

原 `requirements.txt` 的 torch 2.4.0 / torchvision 0.18.0 不匹配，现改为 2.4.0 / 0.19.0，
并补齐原程序用到的 torchsummary、thop、einops。原图形工具改为 Agg，避免无桌面服务器导入失败。
`adv_samples` 才需要的 cleverhans 改为延迟导入；若运行原仓库该特殊实验，请另外安装 cleverhans。

建议新建环境。复用已有匹配的 PyTorch 环境时，不要盲目覆盖其 CUDA 包。

```bash
conda create -n sanitizer-trace python=3.10 -y
conda activate sanitizer-trace
pip install -r requirements.txt
pip install 'pytest>=8,<9'
```

仅运行追踪扩展，可先按机器选择匹配的 torch/torchvision，再执行：

```bash
pip install -r requirements-trace.txt
python trace_cli.py --help
```

本次实际验证环境：Python 3.12、torch 2.5.1+cpu、torchvision 0.20.1+cpu、NumPy 1.26.4。
这证明所列代码检查的兼容性，不表示不同环境会复现完全相同的实验数值。
CUDA 实验使用 `--device cuda:0`；默认 CPU。较大的 E 训练会显著增加显存和时间。

## 3. 最快检查代码能否运行

在仓库根目录运行：

```bash
python -m pytest -q
python examples/smoke_trace.py --output-dir trace_runs/synthetic_smoke
```

第二条完全不下载数据，使用极小网络、合成输入和放宽的性能约束。
它实际训练指纹并走完校准、测试、配对指标及三种模型修改，只用于检查接口连通。
其中未知模型为随机模型，**不能**用于宣称研究中的低误报率。
输出 `SMOKE_ONLY.json` 明确标记了这一点。已存在的输出目录不会被覆盖。

## 4. 从原 Sanitizer 得到基础模型与服务器材料

先完成原流程，再嵌入身份指纹。例如以下参数保留原 README 的 CIFAR10 实验结构：

```bash
python main_fed.py \
  --gpu 0 --dataset cifar --num_channels 3 --model cnn --iid --conflict 0 \
  --num_users 10 --num_classes 10 --all_clients \
  --epochs 200 --local_ep 10 --local_bs 64 --lr 0.01 --momentum 0.9 \
  --poisoning_rate 0.1 --implant_way cifar-pixel --wm_method white_block \
  --epochs_ul 20 --bs_ul 32 --lr_ul 0.01 --defense_data_ratio 0.05 \
  --topK_ratio 0.4 --clean_threshold 0.15 \
  --reverse_eps 2 --bs_re 128 --lr_re 0.01 --lambda_weight 0.01 --full_re 1 \
  --relearn_eps 50 --relearn_bs 32 --relearn_lr 0.005 \
  --seed 1 --trace_export_dir server_export/run1
```

这是耗时的真实训练，本代码包不附带已经完成此实验的新权重。
`--num_users` 改为大于 10 时，还要通过原参数提供覆盖所有客户端的三个字典，不能只改人数。

完成无害重学习之后，`server_export/run1/` 包含：

| 文件 | 用途 |
| --- | --- |
| `sanitized_base.pt` | 最终净化基础模型，CIFAR-10 原 ResNet18 |
| `defense.pt` | 原服务器防御数据，用于指纹嵌入 |
| `validation.pt` | 额外保留的训练集样本，用于指纹阶段模型选择 |
| `contribution_train.pt` / `contribution_test.pt` | 原贡献水印的固定规范查询，可选回放/验证 |
| `original_triggers.pt` | 服务器恢复出的旧触发器、掩码和目标，用于检查旧触发行为是否复活 |

防御数据和额外验证数据的索引都从所有客户端训练划分中排除；额外验证数据不用于原 Sanitizer 梯度训练。
贡献验证使用原机制的固定背景，两份贡献文件有意保存相同的规范查询，不声称它们是独立留出样本。
`original_triggers.pt` 是**恢复出的**触发器；若需测量客户端真正原始图案，还应使用原仓库的对应测试函数。
这整个目录都应由服务器保管。

原仓库附带的 `Pretrained_Backdoored_...pt` 与 `Pretrained_unlearned_...pt` 用于子网示例，
**不是本方案已经净化完成、可以直接用于论文实验的最终 M0**；扩展不会自动加载它们。

如果已有可信的最终净化 `.pth`（原 `utils.util.save_model` 的纯 state_dict），可直接作为 `--base`。
此时须自行提供训练前已保留且不重叠的服务器数据，不能事后把已用于客户端训练的数据称作独立防御集。

## 5. 生成秘密码本和身份副本

```bash
python trace_cli.py keys --output trace_private/key.pt \
  --recipients 10 --bits 64 --distance 20

python trace_cli.py embed \
  --key trace_private/key.pt --base server_export/run1/sanitized_base.pt \
  --defense-data server_export/run1/defense.pt \
  --validation-data server_export/run1/validation.pt \
  --output-dir trace_runs/D --device cuda:0 \
  --epochs 10 --lr 0.001 --batch-size 64 \
  --task-weight 1 --fingerprint-weight 1 --safety-weight 1 --robust-weight 0
```

这些超参数是起点，尚未经过 CIFAR-10 调优。若验证集上没有满足默认 2 个百分点准确率下降约束的模型，
程序明确报错，不会静默保存一个不达标的“最终”副本。放宽约束前应检查并记录原因。
准确率约束仅参与模型选择，安全性需要后面的独立测试，程序没有宣称自动通过安全认证。

E 组使用相同基础模型、码本和数据，改成新输出目录及 `--robust-weight 0.2 --robust-every 5 --inner-lr 0.001`。
它模拟一步干净 SGD 后的指纹损失，**不**模拟剪枝，也不代表任意多步微调的保证。

保留原贡献验证可额外加：

```bash
--contribution-data server_export/run1/contribution_train.pt --contribution-weight 1
```

若需研究复现，可给 `keys` 指定 `--seed 123`；这不是部署用的不可预测秘密。
不指定 seed 时使用系统随机秘密。固定密钥跨实验复用便于公平比较，但不要将其公开给接收者。

输出分成：

- `trace_runs/D/distribution/copy_000/model.pt` 等：交付副本，只含架构名与权重。
- `trace_runs/D/server_private/registry.json`：服务器持有的身份对应、模型 SHA256、训练参数和时间。
- `trace_runs/D/server_private/manifest.json`：用于构造评估清单的模板。
- `trace_runs/D/server_private/*_training.json`：逐轮损失、干净准确率、指纹匹配率。

交付时仅复制对应机构的 `model.pt` 及所需架构代码。
**不要下发 key、server_private、server_export 或导出的秘密 PNG。** 模型无需这些文件也能正常分类。

## 6. 未知模型与阈值校准

不能让解码器始终返回最相似身份。阈值需要真正的未知控制模型。
当前代码会拒绝缺少未知模型、缺少某个注册身份、重复 checkpoint 字节的校准清单。

可在公开 CIFAR-10 上独立训练无水印控制，用不同 seed 分别形成校准池和最终测试池：

```bash
python examples/train_clean_control.py --seed 101 --output controls/cal_101.pt --device cuda:0 --download
python examples/train_clean_control.py --seed 102 --output controls/cal_102.pt --device cuda:0
python examples/train_clean_control.py --seed 201 --output controls/test_201.pt --device cuda:0
python examples/train_clean_control.py --seed 202 --output controls/test_202.pt --device cuda:0
```

此脚本用于公开数据的离线研究对照，不意味着真实 FL 服务器可以集中读取机构私有数据。
训练脚本使用训练集内部留出集选择模型，不使用官方 test 集。应报告控制模型的正常性能，不能只用未训练模型。
四个模型只是使用示例，远不足以证明很低的误报率。可再加入其他合法来源和架构的控制；
本版模型加载器只支持原 ResNet18，其他架构需自行增加适配，HTTP 解码本身不依赖架构。

在仓库根目录创建 `calibration.json`，列出全部 10 份副本、基础模型和校准未知模型：

```python
import json
records = [dict(name=f'copy_{i}', path=f'trace_runs/D/distribution/copy_{i:03d}/model.pt', recipient=i)
           for i in range(10)]
records += [dict(name='base', path='server_export/run1/sanitized_base.pt', recipient=None),
            dict(name='unknown_101', path='controls/cal_101.pt', recipient=None),
            dict(name='unknown_102', path='controls/cal_102.pt', recipient=None)]
with open('calibration.json', 'w') as f:
    json.dump({'records': records}, f, indent=2)
```

清单路径相对于清单文件所在目录；`recipient: null` 表示未注册来源，整数表示真实身份。

```bash
python trace_cli.py calibrate --key trace_private/key.pt --manifest calibration.json \
  --output trace_private/thresholds_D.json --device cuda:0 \
  --variants 4 --max-false-rate 0.01 --min-recall 0.8
```

程序同时约束“未知模型被归到任何身份”和“已知副本被归错身份”的经验比例，
在候选得分/间隔边界上选择满足约束、正确归属最多的组合。达不到要求就报错，不伪造阈值。
校准只使用 calibration 查询。最终 test 查询的背景噪声和变体独立生成。
查询变体相互相关，不能把 256 个查询当作 256 个独立未知模型。
本版默认 64 位 × 4 变体 = 256 次预测，用同一批响应比较全部接收者。

## 7. 对疑似模型进行黑盒追踪

拿到模型文件时：

```bash
python trace_cli.py trace --key trace_private/key.pt --thresholds trace_private/thresholds_D.json \
  --checkpoint suspect.pt --output trace_private/suspect_result.json --device cuda:0
```

只有 API 时：

```bash
python trace_cli.py trace --key trace_private/key.pt --thresholds trace_private/thresholds_D.json \
  --endpoint https://YOUR-SERVICE/predict --output trace_private/suspect_result.json
```

示例 HTTP 合约：POST JSON `{"images": ["base64编码的PNG", ...]}`；响应 `{"labels": [3, 1, ...]}`。
每张图必须返回一个 0..9 的整数标签，采用 torchvision CIFAR-10 原顺序：
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck。
服务端需按原归一化处理 32×32 RGB 图像，不应再次改变类别映射。PNG 像素与本地查询经过一致量化。
不同服务需要改适配函数，不能直接把任意商业 API 当作此合约。若需认证，用 `--token-env VARIABLE_NAME`，
程序从环境变量读取 bearer token，不写入结果。

本地可使用这个只需模型文件的演示服务：

```bash
python examples/serve_label_api.py --checkpoint trace_runs/D/distribution/copy_000/model.pt --port 8765
# 另开终端，将上面的 --endpoint 改为 http://127.0.0.1:8765/predict
```

演示服务仅监听回环地址，不是生产服务。它不加载任何服务器秘密密钥。

也支持离线调用其他接口：

```bash
python trace_cli.py export-queries --key trace_private/key.pt --output-dir trace_private/probes
```

按 `request_manifest.json` 的文件顺序查询，保存 JSON：
`{"key_id":"从清单复制", "query_sha256":"从清单复制", "labels":[...]}`，
再用 `trace --responses returned_labels.json` 替代 `--endpoint`。

输出 `attributed` 或 `unattributable`，以及得分、间隔和身份（若可归属）。
只证明副本来源关联；不直接证明某人主动泄露。该程序不主动寻找外流服务，也不能绕过输入过滤。

## 8. 正式评估和修改实验

导出官方 test 集（与指纹嵌入数据分开）：

```bash
python trace_cli.py prepare-data --role test --data-root data/cifar --output trace_private/test.pt --download
```

创建 `evaluation.json`，保留已知副本，未知项改成 `controls/test_201.pt`、`controls/test_202.pt`。
**不要再次放入参与校准的 base 或未知 checkpoint。** 程序会按 SHA256 拒绝重复未知模型；
同一已知副本可分别使用 calibration/test 查询，但最终还应增加独立训练种子与密钥重复实验。

```bash
python trace_cli.py evaluate --key trace_private/key.pt --thresholds trace_private/thresholds_D.json \
  --manifest evaluation.json --output trace_runs/D/evaluation.json --device cuda:0 \
  --test-data trace_private/test.pt --base server_export/run1/sanitized_base.pt \
  --contribution-data server_export/run1/contribution_test.pt \
  --original-triggers server_export/run1/original_triggers.pt
```

会同时输出 `.json` 和匹配矩阵 `.csv`。完整安全评估要测试所有位点、三个位置变换和全部测试图像，
成本较高；调试可用 `--max-samples 128`，正式报告必须披露实际样本数。
目标 ASR 排除真实标签已经等于目标类的样本；同时报告 M0 在相同图案上的 ASR、最坏增量、正常准确率。
仅通过 `--manifest` 不加 test-data 时，只评估追踪，不评估无害性。

修改副本：

```bash
python trace_cli.py attack --checkpoint trace_runs/D/distribution/copy_000/model.pt \
  --method prune --amount 0.2 --output modified/prune20_000.pt

python trace_cli.py attack --checkpoint trace_runs/D/distribution/copy_000/model.pt \
  --method weight-round --output modified/round8_000.pt
```

`weight-round` 是每个参数张量的对称 int8 网格舍入后回存浮点，不是实际部署的 int8 后端量化。
微调数据必须来自独立攻击数据（`role=attack`），不能拿最终测试集来微调：

```bash
python trace_cli.py prepare-data --role attack --indices attack_indices.json \
  --data-root data/cifar --output trace_private/attack.pt
python trace_cli.py attack --checkpoint trace_runs/D/distribution/copy_000/model.pt \
  --method finetune --attack-data trace_private/attack.pt --epochs 5 --lr 0.001 \
  --output modified/ft_000.pt --device cuda:0
```

`attack_indices.json` 是 CIFAR train 整数索引列表，须与服务器防御、验证数据分开；
真实威胁实验可用泄露者自己的合法数据。代码不自动假设攻击者拥有服务器秘密查询。
修改之后更新评估清单中的路径、保留真实 recipient，使用**固定的原阈值**重新 evaluate，不能用最终攻击测试调阈值。
每种方法都报告正确归属、错误归属、拒绝归属、主任务性能和安全性，不能只报最优匹配得分。

## 9. 原仓库修复与实验可比性

这些修复使运行流程更接近论文描述，但可能改变原仓库输出；比较 D/E 时必须使用同一个修复版 M0。
若需原始代码复现，使用 `UPSTREAM.json` 指定提交另建对照，不要把修复版结果称作原作者公布数字。

1. `main_fed.py`：原循环会让后一个客户端接着前一个客户端的本地状态训练；现每轮固定共同起点。
2. 原脚本导入了剪枝函数但聚合的是未剪枝 `w`；现聚合剪枝后的完整模型参数。
3. 对被剪枝卷积通道，同时将对应 BatchNorm 的 affine 输出置零，避免零卷积仍产生常数输出。
4. 防御数据原来从客户端已划分的数据中再次抽样；现从所有客户端索引中排除，并额外保留指纹验证集。
5. `FedAvg` 增加样本数加权，整数 BN 计数保持整数；默认调用仍可使用等权。
6. `final_unharmful_retrain` 去掉硬编码 20 人和 2500 张，使用真实人数/数据长度/配置批量；修正剪裁梯度时机。
7. 重学习轮数和批量参数改为整数；颜色生成保留前 10 个原背景并支持更多索引。
8. 修复无桌面导入、缺失依赖、torch/torchvision 配对、少于 3 人的末尾测试及 save 目录创建。
9. 新增最终导出接口和随机种子初始化。训练中间模型的保护仍不在本扩展范围内。

代码目录：`traceability/` 实现协议，`trace_cli.py` 提供入口，`examples/` 提供演示和控制模型训练，
`tests/traceability/` 提供回归测试。`changes.patch` 记录相对指定上游提交的源码修改。

## 10. 下一步实验建议

先跑通 M0 与 D，再画副本×身份矩阵；随后用固定阈值测试微调和剪枝。
只有发现稳定失败，再比较 E 的收益。统一数据划分、密钥、查询数和主任务质量约束，并报告额外计算成本。
至少使用多个训练种子/密钥与更多独立未知模型。低误报需要足够样本，而不是少量模型上零误判。
TraMark 是应另行复现的外部黑盒追踪基线，本包没有伪装成已实现该方法。
