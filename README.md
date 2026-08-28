# LCM Benchmark 一键复现脚本（reproduce.sh）

本目录提供一个**可一键复现**的脚本，用来严格复现论文 *How Good are Learned Cost
Models, Really? Insights from Query Optimization Tasks*（SIGMOD 2025，DOI
10.1145/3725309，代码仓库 DataManagementLab/lcm-eval）的训练与评估实验。

同一个脚本 reproduce.sh 同时支持：

* 本地 CPU 训练 + 评估（一条命令）
* 打包自包含的远端 GPU 训练 + 评估（打包 + 远端一键 + 结果导回）
* 论文 T4 微调（retrain）
* 指定全部或部分模型 / 目标库 / 随机种子

## 核心设计原则

1. **严格遵循官方命令**。训练 / 评估 / 重训命令逐参数对应官方
   src/scripts/exp_runner/exp_train_model.py、exp_predict_all.py、
   exp_retrain_model.py，不新增官方 main.py 不支持的参数。
2. **官方仓库与数据保持只读**。lcm-eval/（官方源码）和 data/runs/（OSF 数据）
   一律不修改；官方缺失、需要兼容处理的统计文件统一派生到 prepared/。
3. **全部使用相对目录**。所有路径基于本脚本所在目录计算，整个目录可整体拷贝到
   任意机器，无需 AI 辅助即可按本文档逐步复现。
4. **不复制散乱文件**。派生文件只写 prepared/，数据用软链复用，不在仓库里堆
   重复文件。

## 目录结构

    version pro/
    ├── reproduce.sh          一键复现主脚本
    ├── prepare_stats.py      派生官方缺失统计文件的脚本（只读 OSF 数据，写 prepared/）
    ├── README.md             本文档
    ├── lcm-eval/             官方仓库（只读，由 setup 克隆；已存在则跳过）
    ├── data/runs/            官方 OSF 数据（只读；setup 下载，或 --link 软链复用）
    ├── prepared/             派生统计文件（statistics_complex_workload_combined.json 等）
    ├── models/               训练产物（checkpoint）
    ├── evaluation/           评估产物（逐查询预测 CSV）
    ├── logs/                 训练 / 评估 / 重训日志
    ├── outgoing/             GPU 打包与结果归档（tar.gz + sha256）
    ├── gpu_results/          远端结果暂存
    ├── .venv/                本地 CPU Python 环境（setup 创建）
    ├── .venv-gpu/            远端 GPU Python 环境（gpu-setup 创建）
    └── .tools/               uv 工具（setup 自动安装）

## 模型与实验范围

官方共 7 个基础模型，3 个目标库，默认 3 个随机种子：

    flat, zeroshot, dace, mscn, e2e, qppnet, query_former   （模型）
    imdb, baseball, tpc_h_pk                                 （目标库）
    0, 1, 2                                                   （种子）

按官方 classes.py 的划分：

* workload-agnostic（用其余 19 库训练，目标库做 test）：flat、zeroshot、dace
* workload-driven（目标库单库训练，官方 loader 内部 80/10/10 划分）：mscn、e2e、
  qppnet、query_former

全部训练任务规模：7 模型 × 3 库 × 3 seed = 63（对应 v3 的 T1）。

评估 workload 对应论文 E1–E4：

* imdb：join_order_full（连接顺序）、scan_costs_percentiles（访问路径）、
  physical_plan（物理算子）、physical_index_plan（加索引）
* baseball / tpc_h_pk：scan_costs_percentiles、physical_plan

T4 微调（retrain）：在 imdb 上对官方 index_retraining 与 seq_retraining
两个 workload 做 --mode retrain。默认模型严格对应官方 exp_retrain_model.py
（as-is 只启用了 ZeroShotModelConfig）＝ zeroshot × 3 seed = 3 个任务。

注意：官方 main.py 的 retrain 分支（main.py:370）对 loss_class_name=="QLoss"
且 final_mlp_kwargs=None 的模型（flat/e2e/mscn/query_former）会直接抛
AttributeError，这是官方代码的已知限制（官方文件也并未启用这些模型）；官方
命令可正常 retrain 的是 zeroshot、dace、qppnet。脚本对不支持模型仅提示、不阻断，
生成的命令仍与官方完全一致。

## 环境要求

* Linux（bash 4+），curl、git、tar、sha256sum
* 可访问 GitHub、PyPI、download.pytorch.org、data.dgl.ai（首次 setup 需要下载）
* 本地 CPU 训练：只需 CPU 内存（建议 32GB+，flat 训练较快，神经模型在 CPU 上偏慢）
* 远端 GPU 训练：NVIDIA GPU + CUDA 11.8/12.1 驱动即可，无需 sudo（用 uv 装独立 Python）

## 一、本地 CPU：环境安装 setup

    bash reproduce.sh setup

setup 依次完成：

1. 克隆官方仓库到 lcm-eval/（已存在则跳过，保持只读）
2. 创建 Python 环境：优先用系统 python3（3.9–3.11）建 venv；否则自动下载 uv 并安装
   Python 3.10（官方代码含 list[...] 注解，Python 3.8 无法运行）
3. 安装官方同版本的 torch 2.3.0（CPU）、dgl 1.1.3 及运行依赖
4. 用官方 download_from_osf.py 下载 OSF runs 数据（约 1.4GB，解压后约 10GB）
   到 data/runs/
5. 生成官方缺失的派生统计文件到 prepared/

如果你已经有官方 runs 数据（例如 v3 的 data/raw/runs），可以用只读软链复用，
不复制任何文件：

    bash reproduce.sh setup --link /path/to/existing/runs

完成后即可训练。

## 二、训练 train

    # 全部模型 × 全部库 × 全部种子（63 个任务）
    bash reproduce.sh train

    # 部分模型 / 库 / 种子
    bash reproduce.sh train --models flat,zeroshot --dbs imdb --seeds 0,1

    # 本地 GPU 训练（本机有 CUDA 时）
    bash reproduce.sh train --models e2e,mscn --device cuda:0 --parallel 2

    # 多卡轮转
    bash reproduce.sh train --models qppnet,e2e --gpus 0,1

    # 训练完成后自动评估（一条命令完成训练 + 评估）
    bash reproduce.sh train --with-eval

常用参数：

    --models a,b,c    模型列表（默认全部 7 个）
    --dbs a,b,c       目标库（默认 imdb,baseball,tpc_h_pk）
    --seeds a,b,c     随机种子（默认 0,1,2）
    --device cpu|cuda:N   训练设备（默认 cpu）
    --parallel N      并行任务数（默认 1）
    --gpus a,b        多卡轮转（把 CUDA_VISIBLE_DEVICES 分发给任务）
    --num-workers N   追加官方可选的 --num_workers（默认不传，与官方命令一致）
    --wandb-project X / --wandb-name X   覆盖默认的 cost-eval / 名称
    --force           忽略已有 checkpoint，强制重训
    --dry-run         只打印生成的官方命令，不真正执行
    --with-eval       训练完成后自动执行评估

脚本默认跳过已存在的 checkpoint（flat 是 models/flat/<db>_<seed>.txt，神经模型是
models/<model>/<db>/<model>_<seed>.pt），用 --force 可重训。

## 三、评估 eval / predict

评估按官方 exp_predict_all.py 的分组方式：把每个文件夹下的全部 *.json
workload 一次传给 --test_workload_runs，输出写 evaluation/<model>/<db>/<folder>/。

    # 全部模型 × 全部库 × 全部种子（论文 E1–E4）
    bash reproduce.sh eval

    # 只看某个模型 / 库 / 种子
    bash reproduce.sh eval --models flat,mscn --dbs imdb --seeds 0

    # 只看某一类评估 workload
    bash reproduce.sh eval --eval-set physical_plan --dbs imdb
    bash reproduce.sh eval --eval-set core        # 论文全部（默认）

    # 评估任意自定义 workload 文件
    bash reproduce.sh eval --test-workload /path/to/workload.json

--eval-set 可选：core（默认，等于 paper）、paper、join_order_full、
scan_costs_percentiles、physical_plan、physical_index_plan。

评估会跳过没有对应 checkpoint 的模型（需要先 train）。

## 四、T4 微调 retrain

    bash reproduce.sh retrain                       # 默认 zeroshot × imdb × 3 seed
    bash reproduce.sh retrain --models zeroshot,dace --seeds 0

官方 exp_retrain_model.py 从 retraining_models/<model>/imdb 目录直接加载
checkpoint 续训，因此脚本会先把 models/<model>/imdb/ 的基础 checkpoint 复制到
models/retrained/<model>/imdb/，再执行与官方一致的 --mode retrain 命令。
运行前需要先 train 对应模型的 imdb。

官方 retrain 分支（main.py:370）只对配置了 final_mlp_kwargs 的模型有效；
zeroshot/dace/qppnet 可正常 retrain，flat/e2e/mscn/query_former 会因官方
代码的 QLoss 分支抛 AttributeError（官方文件也只启用了 zeroshot）。脚本默认
zeroshot，其它模型需显式 --models 指定，且会提示上述官方限制。

## 五、打包远端 GPU 训练（在本地执行）

    # 打包全部模型（含数据与脚本），默认不评估
    bash reproduce.sh gpu-bundle

    # 只打包部分模型，并让远端训练后自动评估
    bash reproduce.sh gpu-bundle --models e2e,query_former --dbs imdb --seeds 0,1 --with-eval

    # 训练后自动做 T4 微调
    bash reproduce.sh gpu-bundle --models zeroshot,dace --dbs imdb --seeds 0 --with-retrain

产物在 outgoing/lcm_gpu_bundle_<时间戳>.tar.gz 和同名 .sha256。包内包含官方源码、
reproduce.sh、prepare_stats.py、本库训练所需的全部数据（只拷需要的部分）以及
precomputed 的 prepared 统计，是自包含的。可用 --out <相对路径.tar.gz> 指定产物
文件名（相对本脚本目录，例如 --out outgoing/my_bundle.tar.gz）。

把两个文件上传到 GPU 服务器，然后：

    sha256sum -c lcm_gpu_bundle_*.tar.gz.sha256
    tar -xzf lcm_gpu_bundle_*.tar.gz
    cd lcm_gpu_bundle
    bash gpu-run.sh

gpu-run.sh 会自动完成：环境安装（gpu-setup：uv + Python 3.10 + torch cu121 +
dgl，首次需要联网，无需 sudo）→ 训练 →（若打包时带 --with-eval 则评估）→（带
--with-retrain 则微调）→ 打包结果到 outgoing/lcm_gpu_results_<时间戳>.tar.gz。

取回结果（在本地执行）：

    bash reproduce.sh import-results /path/to/lcm_gpu_results_*.tar.gz

导入器先校验 sha256，再把 models/、evaluation/、logs/ 拷入本地对应目录
（--no-clobber，拒绝覆盖已有结果）。

## 六、严格遵循官方命令的说明

脚本生成的命令与官方 exp 脚本逐参数一致：

训练（对应 exp_train_model.py）：

    python main.py --mode train --wandb_project cost-eval --model_type <m>
      --device <d> --model_dir models/<m>/<db> --target_dir evaluation/<m>/<db>
      --workload_runs <19 个训练库 或 1 个目标库> --statistics_file <统计>
      --seed <s> --wandb_name <m>/<db>/<s>
      [--test_workload_runs ...] [--column_statistics ...] [--word_embeddings ...]
      [--hyperparameter_path ...]

评估（对应 exp_predict_all.py）：

    python main.py --mode predict --model_type <m>
      --test_workload_runs <某文件夹全部 json> --statistics_file <统计>
      --model_dir models/<m>/<db> --target_dir evaluation/<m>/<db>/<folder> --seed <s>
      [--column_statistics ...] [--word_embeddings ...] [--hyperparameter_path ...]

重训（对应 exp_retrain_model.py）：

    python main.py --mode retrain --wandb_project cost-eval-retrain
      --model_type <m> --device <d> --model_dir models/retrained/<m>/imdb
      --target_dir evaluation/retrained/<m>/imdb
      --workload_runs <index_retraining.json seq_retraining.json>
      --statistics_file <统计> --seed <s> --wandb_name <m>/imdb/<s>
      [--test_workload_runs ...] [--column_statistics ...] [--word_embeddings ...]
      [--hyperparameter_path ...]

注意：

* 官方命令本来就不带 --num_workers 和评估的 --device；本脚本默认照抄官方，
  只有显式传 --num-workers 才会追加 --num_workers。
* 不传 v3 私有补丁参数（如 --epochs、--batch_size、--early_stopping_patience），
  保证与官方 main.py 参数完全兼容。

## 七、官方缺失文件的兼容处理（只写 prepared/）

官方源码需要的几个统计文件，OSF 数据里并没有完整提供；v3 已逐项验证过兼容方案，
本脚本用 prepare_stats.py 从官方数据派生，全部写入 prepared/，不改官方任何文件：

    statistics_complex_workload_combined.json
      官方 flat/dace/e2e 需要，OSF 未提供。
      由 parsed_plans/statistics_workload_combined.json 派生，补充 workers_planned
      数值特征（RobustScaler 等价，v3 已验证内容一致）。

    prepared/qpp_stats/<db>.json
      官方 qppnet 指向 json/<db>/feature_stats_training_data_scans_joins_physical.json，
      OSF 未提供；OSF 提供的 feature_statistics_combined.json 与该文件字节相同，
      因此直接字节拷贝。

    prepared/statistics_zeroshot_train.json
      zeroshot 需要把 categorical value_dict 补全（例如 credit.member.photograph
      的 data_type=bytea），否则官方 featurization 会崩溃；v3 已验证。

另外，官方 zeroshot 的 hyperparameter 指向 conf/tuned_hyperparameters/...，
但仓库实际没有该目录；脚本按 v3 验证结果使用实际存在的
conf/zeroshot_hyperparameters/tune_est_best_config.json，参数名与官方完全一致。

## 八、常见问题

* 脚本默认 WANDB_MODE=disabled（离线模式，不联网、不要求账号）。如需上报 wandb，
  运行前 export WANDB_MODE=online 并自行登录。
* setup 若因网络慢而中断，可重跑（已完成的步骤会跳过，checkpoint/数据已存在不重复）。
* 训练失败日志在 logs/<mode>/，对应任务以 TRAIN__<模型>__<库>__seed<种子>.log、
  EVAL__...、RETRAIN__... 命名，方便逐条排查。
* 断点续跑：已完成的 checkpoint 会自动跳过；评估只需对应 checkpoint 存在。
* 磁盘占用：OSF 数据约 10GB、GPU bundle 每个 100–500MB 取决于模型范围。
* 想先看命令而不执行：所有子命令都支持 --dry-run。
