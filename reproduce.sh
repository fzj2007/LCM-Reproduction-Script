#!/usr/bin/env bash
# =============================================================================
# lcm-eval 一键复现脚本 (reproduce.sh)
#
# 严格按照官方仓库 DataManagementLab/lcm-eval（SIGMOD 2025, DOI 10.1145/3725309）
# 的 exp_train_model.py / exp_predict_all.py / exp_retrain_model.py 逐参数生成
# 训练、评估(predict)、重训(retrain)命令：
#
#   训练  -> python main.py --mode train  --wandb_project cost-eval ...
#   评估  -> python main.py --mode predict --test_workload_runs ...
#   重训  -> python main.py --mode retrain --wandb_project cost-eval-retrain ...
#
# 子命令：
#   setup           克隆官方仓库 + 建 Python 环境 + 下载/链接 OSF runs + 生成派生统计
#   train           本地训练（全部/部分模型、目标库、seed、device）
#   eval / predict  本地评估（对论文 E1-E4 评估负载逐文件夹 predict）
#   retrain         论文 T4 微调（imdb index/seq retraining）
#   gpu-bundle      打包自包含的远端 GPU 训练(+评估) 包（同目录 tar.gz + sha256）
#   gpu-setup       在远端 GPU 主机安装环境（uv + torch cu121 + dgl）
#   gpu-run         在远端 GPU 主机一键 训练(+评估)+打包（读取 bundle 内参数）
#   gpu-pack        在远端 GPU 主机把 models/evaluation/logs 打成 tar.gz + sha256
#   import-results  校验并导入远端结果（拒绝覆盖已有结果）
#   list-models     列出模型元数据
#   help            帮助
#
# 约束（对应本任务要求）：
#   - 官方仓库 lcm-eval/ 与下载数据 data/runs 保持只读；所有派生兼容文件写入 prepared/；
#   - 所有路径相对本脚本目录，整个目录可整体拷贝到任意机器，无需 AI 辅助即可复现；
#   - 训练/评估/重训命令逐参数与官方 exp_*.py 一致（不额外传官方不支持的参数）。
#   - 本地 CPU 训练：bash reproduce.sh setup && bash reproduce.sh train
#   - 打包远端 GPU 训练：bash reproduce.sh gpu-bundle ...，远端 gpu-run 一键完成。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 目录（全部相对脚本目录，可整体拷贝）----
REPO_DIR="$SCRIPT_DIR/lcm-eval"
SRC_DIR="$REPO_DIR/src"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_GPU_DIR="$SCRIPT_DIR/.venv-gpu"
TOOLS_DIR="$SCRIPT_DIR/.tools"
UV="$TOOLS_DIR/uv"
DATA_DIR="$SCRIPT_DIR/data"
RUNS_DIR="$DATA_DIR/runs"
PREPARED_DIR="$SCRIPT_DIR/prepared"
MODELS_DIR="$SCRIPT_DIR/models"
EVAL_DIR="$SCRIPT_DIR/evaluation"
LOGS_DIR="$SCRIPT_DIR/logs"
OUTGOING_DIR="$SCRIPT_DIR/outgoing"
GPU_RESULTS_DIR="$SCRIPT_DIR/gpu_results"
RETRAIN_MODELS_DIR="$MODELS_DIR/retrained"
RETRAIN_EVAL_DIR="$EVAL_DIR/retrained"

# ---- 官方元数据（与 classes/classes.py 一致）----
ALL_MODELS="flat,zeroshot,dace,mscn,e2e,qppnet,query_former"
TARGET_DBS="imdb,baseball,tpc_h_pk"
DEFAULT_SEEDS="0,1,2"
# workload-agnostic（用其余 19 库训练 + 目标库作 test）
AGNOSTIC_MODELS="flat,zeroshot,dace"
# workload-driven（单目标库训练，官方 loader 内部 80/10/10 划分）
DRIVEN_MODELS="mscn,e2e,qppnet,query_former"
# 官方 database_list（排除 tpc_h，用 tpc_h_pk），共 20 个
ALL_DB_WORDS=(
  accidents airline baseball basketball carcinogenesis consumer credit employee
  fhnk financial geneea genome hepatitis imdb movielens seznam ssb tournament
  tpc_h_pk walmart
)
# 官方 retrain 范围（论文 T4：imdb）。
# 严格对应官方 scripts/exp_runner/exp_retrain_model.py：该文件 as-is 只启用了
# ZeroShotModelConfig（其余模型被注释掉）。默认即为 zeroshot。
# 说明：官方 main.py 的 retrain 分支（main.py:370）对 loss_class_name=="QLoss" 且
# final_mlp_kwargs 为 None 的模型配置会抛 AttributeError（flat/e2e/mscn/query_former），
# 这些模型官方未启用，脚本会给出提示；zeroshot/dace/qppnet 可正常 retrain。
RETRAIN_DEFAULT_MODELS="zeroshot"

# 官方 classes.py 在 import 时用 ast.literal_eval 读取 NODE00..NODE05，
# 仓库自带 .env 只含 NODE00/01；这里注入占位值（仅用于绕过 import 检查）。
NODE_ENV='{"hostname": "localhost", "python": "3.10"}'

# ---- 命令行全局参数 ----
MODELS_SPEC=""
DBS_SPEC=""
SEEDS_SPEC=""
DEVICE_SPEC="cpu"
NWORKERS_ARG=""
PARALLEL_SPEC="1"
FORCE=0
DRY_RUN=0
WANDB_PROJECT_ARG=""
WANDB_NAME_ARG=""
EVAL_SET_ARG=""
TEST_WORKLOADS=()
GPU_IDS_SPEC=""
WITH_EVAL=0
WITH_RETRAIN=0
OUT_ARCHIVE=""
PYTHON_BIN_OVERRIDE="${PYTHON_BIN_OVERRIDE:-}"

# ---- 基础工具 ----
log()  { printf '\033[1;32m[reproduce]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[reproduce]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[reproduce]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "缺少系统命令: $1"; }

is_in_list() {
  local item="$1" csv="$2" e
  for e in ${csv//,/ }; do [ "$e" = "$item" ] && return 0; done
  return 1
}

valid_models() {
  local spec="$1" out="" m
  [ "$spec" = all ] && spec="$ALL_MODELS"
  for m in ${spec//,/ }; do
    is_in_list "$m" "$ALL_MODELS" || die "不支持的模型: $m（可用: $ALL_MODELS）"
    out="${out:+$out,}$m"
  done
  echo "$out"
}
valid_dbs() {
  local out="" d
  for d in ${1//,/ }; do
    is_in_list "$d" "$TARGET_DBS" || die "不支持的目标库: $d（可用: $TARGET_DBS）"
    out="${out:+$out,}$d"
  done
  echo "$out"
}
valid_seeds() {
  local out="" s
  for s in ${1//,/ }; do
    case "$s" in ''|*[!0-9]*) die "非法 seed: $s（应为非负整数）" ;; esac
    out="${out:+$out,}$s"
  done
  echo "$out"
}

# 选择 Python：GPU 主机优先 .venv-gpu，否则 .venv（可用 PYTHON_BIN 覆盖）
python_bin() {
  if [ -n "$PYTHON_BIN_OVERRIDE" ]; then
    printf '%s\n' "$PYTHON_BIN_OVERRIDE"
  elif [ "${LCM_GPU_HOST:-0}" = "1" ] && [ -x "$VENV_GPU_DIR/bin/python" ]; then
    printf '%s\n' "$VENV_GPU_DIR/bin/python"
  else
    printf '%s\n' "$VENV_DIR/bin/python"
  fi
}
require_env() {
  local py; py="$(python_bin)"
  [ -x "$py" ] || die "未找到 Python 环境（$py）。本地请先运行 ./reproduce.sh setup；GPU 主机请运行 ./reproduce.sh gpu-setup"
}
require_data() {
  [ -d "$RUNS_DIR/parsed_plans" ] || die "缺少训练数据 $RUNS_DIR。请先运行 ./reproduce.sh setup（或 setup --link <已有runs>）"
}
require_prepared() {
  [ -f "$PREPARED_DIR/statistics_complex_workload_combined.json" ] || die "缺少 prepared/statistics_complex_workload_combined.json。请先运行 ./reproduce.sh setup"
}

export_node_env() {
  export NODE00="$NODE_ENV" NODE01="$NODE_ENV" NODE02="$NODE_ENV" \
         NODE03="$NODE_ENV" NODE04="$NODE_ENV" NODE05="$NODE_ENV"
  export LOCAL_ROOT_PATH="$SCRIPT_DIR"
}

# =============================================================================
# 模型元数据（严格对应官方 classes/classes.py 与 scripts/exp_runner/*.py）
# =============================================================================

model_is_agnostic() { is_in_list "$1" "$AGNOSTIC_MODELS"; }
model_is_driven()   { is_in_list "$1" "$DRIVEN_MODELS"; }

# 官方 get_data_base_dir()：模型 -> 输入表示目录（相对 RUNS_DIR）
input_root_of() {
  case "$1" in
    flat|zeroshot|dace)  printf '%s\n' "$RUNS_DIR/parsed_plans" ;;
    mscn|e2e|query_former) printf '%s\n' "$RUNS_DIR/augmented_plans_baseline" ;;
    qppnet)              printf '%s\n' "$RUNS_DIR/json" ;;
    *) die "未知模型: $1" ;;
  esac
}

# 官方 get_statistics()：模型/库 -> 统计文件。
#   - flat/dace/e2e 官方指向 parsed_plans/statistics_complex_workload_combined.json，
#     该文件 OSF 不提供，须由 statistics_workload_combined.json 派生（+workers_planned）；
#     派生文件写入 prepared/（data/runs 保持只读）。
#   - zeroshot 另需 bytea 等类别补丁（官方文件会因 credit.member.photograph 的 bytea
#     缺失而崩溃，v3 已验证）；同样写入 prepared/。
#   - qppnet 官方指向 json/<db>/feature_stats_training_data_scans_joins_physical.json，
#     OSF 也不提供（= feature_statistics_combined.json 的字节拷贝），放入 prepared/qpp_stats/。
stats_file_for() {
  local model="$1" db="$2"
  case "$model" in
    mscn|query_former)
      if [ "$db" = imdb ]; then
        printf '%s\n' "$RUNS_DIR/augmented_plans_baseline/imdb/statistics_combined.json"
      else
        printf '%s\n' "$RUNS_DIR/augmented_plans_baseline/$db/statistics.json"
      fi ;;
    qppnet)
      printf '%s\n' "$PREPARED_DIR/qpp_stats/$db.json" ;;
    zeroshot)
      printf '%s\n' "$PREPARED_DIR/statistics_zeroshot_train.json" ;;
    flat|dace|e2e)
      printf '%s\n' "$PREPARED_DIR/statistics_complex_workload_combined.json" ;;
    *) die "未知模型: $model" ;;
  esac
}

# retrain 用统计文件：qppnet 用 json/imdb/feature_stats_retraining.json（OSF 提供）
retrain_stats_for() {
  local model="$1"
  if [ "$model" = qppnet ]; then
    printf '%s\n' "$RUNS_DIR/json/imdb/feature_stats_retraining.json"
  else
    stats_file_for "$model" imdb
  fi
}

# 官方 get_column_stats / get_word_embeddings / hyperparameter 追加参数
extra_args_append() {
  local model="$1" db="$2"
  case "$model" in
    mscn|e2e|query_former)
      CMD_ARR+=(
        --column_statistics "$SRC_DIR/cross_db_benchmark/datasets/$db/column_statistics.json"
        --word_embeddings "$RUNS_DIR/sentences/$db/word2vec.m"
      ) ;;
    qppnet)
      CMD_ARR+=( --column_statistics "$SRC_DIR/cross_db_benchmark/datasets/$db/column_statistics.json" ) ;;
    zeroshot)
      # 官方 ZeroShotModelConfig.hyperparameter 指向 conf/tuned_hyperparameters/...（仓库中不存在），
      # 实际文件在 conf/zeroshot_hyperparameters/（v3 验证）。参数本身与官方一致。
      CMD_ARR+=( --hyperparameter_path "$SRC_DIR/conf/zeroshot_hyperparameters/tune_est_best_config.json" ) ;;
  esac
}

# 官方 get_training_workloads()：agnostic = 排除目标库的其余 19 库；driven = 目标库单个 workload
train_workloads_append() {
  local model="$1" db="$2" root d
  root="$(input_root_of "$model")"
  if model_is_agnostic "$model"; then
    for d in "${ALL_DB_WORDS[@]}"; do
      [ "$d" = "$db" ] && continue
      CMD_ARR+=( "$root/$d/workload_100k_s1_c8220.json" )
    done
  else
    CMD_ARR+=( "$root/$db/workload_100k_s1_c8220.json" )
  fi
}

# 官方 get_test_workloads()：仅 agnostic 在 train/retrain 模式传 --test_workload_runs（目标库 workload）
test_workload_for() {
  local model="$1" db="$2"
  if model_is_agnostic "$model"; then
    printf '%s\n' "$(input_root_of "$model")/$db/workload_100k_s1_c8220.json"
  fi
}

# 官方 get_model_dir / get_eval_dir：models/<model>/<db>、evaluation/<model>/<db>
model_dir_for() { printf '%s/%s/%s\n' "$MODELS_DIR" "$1" "$2"; }
eval_dir_for()  { printf '%s/%s/%s\n' "$EVAL_DIR" "$1" "$2"; }

# =============================================================================
# 官方命令构建（逐参数顺序与 exp_train_model.py / exp_predict_all.py /
# exp_retrain_model.py 一致）：
#   train/retrain: --mode --wandb_project --model_type --device --model_dir
#                  --target_dir --workload_runs --statistics_file --seed
#                  --wandb_name [--test_workload_runs][--column_statistics]
#                  [--word_embeddings][--hyperparameter_path]
#   predict:       --mode --model_type --test_workload_runs --statistics_file
#                  --model_dir --target_dir --seed [--column_statistics]
#                  [--word_embeddings][--hyperparameter_path]
# 说明：官方命令不含 --num_workers/--device(predict)；仅当用户显式传 --num-workers
# 时才会追加 --num_workers（默认完全照抄官方命令）。
# =============================================================================

build_train_cmd() {
  local mode="$1" model="$2" db="$3" seed="$4" tw
  CMD_ARR=(
    "$(python_bin)" main.py --mode "$mode"
    --wandb_project "${WANDB_PROJECT_ARG:-cost-eval}"
    --model_type "$model"
    --device "$DEVICE_SPEC"
    --model_dir "$(model_dir_for "$model" "$db")"
    --target_dir "$(eval_dir_for "$model" "$db")"
    --workload_runs
  )
  train_workloads_append "$model" "$db"
  CMD_ARR+=(
    --statistics_file "$(stats_file_for "$model" "$db")"
    --seed "$seed"
    --wandb_name "${WANDB_NAME_ARG:-$model/$db/$seed}"
  )
  [ -n "$NWORKERS_ARG" ] && CMD_ARR+=( --num_workers "$NWORKERS_ARG" )
  tw="$(test_workload_for "$model" "$db")"
  [ -n "$tw" ] && CMD_ARR+=( --test_workload_runs "$tw" )
  extra_args_append "$model" "$db"
}

# 收集某个文件夹下的评估 workload（.json 平铺），存到全局 EVAL_WLS
collect_eval_workloads() {
  local model="$1" db="$2" folder="$3" root wl
  EVAL_WLS=()
  root="$(input_root_of "$model")"
  if [ "$folder" = custom ]; then
    for wl in "${TEST_WORKLOADS[@]}"; do EVAL_WLS+=( "$wl" ); done
  else
    for wl in "$root/$db/$folder"/*.json; do
      [ -f "$wl" ] || continue
      # 官方 OSF 数据含少量 parsed_plans 为空的 workload 文件（如 imdb/join_order_full
      # 的 job_light_7/18/40/42/57/59/60/69）。官方 main.py predict 对空 workload 会在
      # LightGBM bst.predict 处崩溃（1 维输入）。这里跳过空文件并告警（仅剔除空文件，
      # 非空 workload 的官方命令逐参数不变），与 v3 复现口径一致。
      if "$(python_bin)" -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if len(d.get("parsed_plans",[])or[])==0 else 1)' "$wl" 2>/dev/null; then
        warn "跳过空 workload（0 条查询，官方 predict 无法处理）: $wl"
        continue
      fi
      EVAL_WLS+=( "$wl" )
    done
  fi
}

build_eval_cmd() {
  local model="$1" db="$2" seed="$3" folder="$4" wl
  # run_task_queue 已在调用本函数前收集并校验 EVAL_WLS；不要再次扫描，
  # 否则空 workload 的跳过告警会重复打印两遍。
  [ "${#EVAL_WLS[@]}" -gt 0 ] || die "找不到 $model/$db/$folder 下的评估 workload"
  CMD_ARR=( "$(python_bin)" main.py --mode predict --model_type "$model"
    --test_workload_runs )
  for wl in "${EVAL_WLS[@]}"; do CMD_ARR+=( "$wl" ); done
  CMD_ARR+=(
    --statistics_file "$(stats_file_for "$model" "$db")"
    --model_dir "$(model_dir_for "$model" "$db")"
    --target_dir "$(eval_dir_for "$model" "$db")/$folder"
    --seed "$seed"
  )
  [ -n "$NWORKERS_ARG" ] && CMD_ARR+=( --num_workers "$NWORKERS_ARG" )
  extra_args_append "$model" "$db"
}

build_retrain_cmd() {
  local model="$1" db="$2" seed="$3" tw root
  root="$(input_root_of "$model")"
  CMD_ARR=(
    "$(python_bin)" main.py --mode retrain
    --wandb_project cost-eval-retrain
    --model_type "$model"
    --device "$DEVICE_SPEC"
    --model_dir "$RETRAIN_MODELS_DIR/$model/$db"
    --target_dir "$RETRAIN_EVAL_DIR/$model/$db"
    --workload_runs
  )
  CMD_ARR+=( "$root/imdb/index_retraining/index_retraining.json" "$root/imdb/seq_retraining/seq_retraining.json" )
  CMD_ARR+=( --statistics_file "$(retrain_stats_for "$model")" --seed "$seed" --wandb_name "$model/$db/$seed" )
  [ -n "$NWORKERS_ARG" ] && CMD_ARR+=( --num_workers "$NWORKERS_ARG" )
  tw="$(test_workload_for "$model" "$db")"
  [ -n "$tw" ] && CMD_ARR+=( --test_workload_runs "$tw" )
  extra_args_append "$model" "$db"
}

# =============================================================================
# 任务执行基础设施（并行 + 日志 + 失败累计；路径含空格全部用 %q 转义）
# =============================================================================

# 把 CMD_ARR 写成一个可执行任务脚本；$3 可选（如 "export CUDA_VISIBLE_DEVICES=1"）
emit_task_script() {
  local script="$1" log="$2" extra_env="${3:-}" w
  {
    printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
    printf 'cd %q\n' "$SRC_DIR"
    printf 'export OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q NUMEXPR_NUM_THREADS=%q\n' \
      "${OMP_NUM_THREADS:-2}" "${MKL_NUM_THREADS:-2}" "${OPENBLAS_NUM_THREADS:-2}" "${NUMEXPR_NUM_THREADS:-2}"
    printf 'export DGLBACKEND=pytorch WANDB_MODE=%q PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=%q\n' \
      "${WANDB_MODE:-disabled}" "${PYTHONHASHSEED:-0}"
    printf 'export NODE00=%q NODE01=%q NODE02=%q NODE03=%q NODE04=%q NODE05=%q\n' \
      "$NODE_ENV" "$NODE_ENV" "$NODE_ENV" "$NODE_ENV" "$NODE_ENV" "$NODE_ENV"
    printf 'export LOCAL_ROOT_PATH=%q\n' "$SCRIPT_DIR"
    [ -n "$extra_env" ] && printf '%s\n' "$extra_env"
    printf 'mkdir -p %q\n' "$(dirname "$log")"
    for w in "${CMD_ARR[@]}"; do printf '%q ' "$w"; done
    printf ' > %q 2>&1\n' "$log"
  } > "$script"
  chmod +x "$script"
}

# 打印 CMD_ARR（dry-run）
print_cmd() {
  local label="$1" w
  printf '\033[1;36m[reproduce]\033[0m %s:\n' "$label"
  printf '  '
  for w in "${CMD_ARR[@]}"; do printf '%q ' "$w"; done
  printf '\n'
}

# 校验评估集（core/paper 为论文全部评估 workload；其余为具体目录名）
validate_eval_set() {
  local set
  for set in ${1//,/ }; do
    case "$set" in
      core|paper|join_order_full|scan_costs_percentiles|physical_plan|physical_index_plan) ;;
      *) die "未知评估集: $set（可选 core|paper|join_order_full|scan_costs_percentiles|physical_plan|physical_index_plan）" ;;
    esac
  done
}

# 模型在指定库/seed 的 checkpoint 是否存在
#   flat 保存为 <model_dir>_<seed>.txt（即 models/flat/imdb_0.txt）
#   其余模型保存为 <model_dir>/<model>_<seed>.pt
checkpoint_exists() {
  local model="$1" db="$2" seed="$3"
  if [ "$model" = flat ]; then
    [ -s "${MODELS_DIR}/$model/${db}_${seed}.txt" ]
  else
    [ -s "$(model_dir_for "$model" "$db")/${model}_${seed}.pt" ]
  fi
}

# =============================================================================
# 任务执行：读取制表符分隔队列
#   train/retrain: model\tdb\tseed
#   eval:          model\tdb\tseed\tfolder（folder=custom 表示 --test-workload 指定）
# 逐任务生成脚本并并行执行；任一任务失败 => 整体失败（exit 1），但会跑完并保留日志。
# 若指定 --gpus a,b，则按轮转把 CUDA_VISIBLE_DEVICES 注入每个任务脚本。
# =============================================================================
run_task_queue() {
  local mode="$1" queue="$2"
  local tmp idx m d s folder script log task_log_dir rc gpu
  local -a gpu_arr=()
  tmp="$(mktemp -d "$SCRIPT_DIR/.tasks.XXXXXX")"
  idx=0
  task_log_dir="$LOGS_DIR/$mode"
  mkdir -p "$task_log_dir"
  if [ -n "$GPU_IDS_SPEC" ]; then
    for gpu in ${GPU_IDS_SPEC//,/ }; do gpu_arr+=( "$gpu" ); done
  fi
  while IFS=$'\t' read -r m d s folder; do
    [ -n "${m:-}" ] || continue
    local gpu_env=""
    if [ "${#gpu_arr[@]}" -gt 0 ]; then
      gpu="${gpu_arr[$((idx % ${#gpu_arr[@]}))]}"
      gpu_env="export CUDA_VISIBLE_DEVICES=$gpu"
    fi
    case "$mode" in
      train)
        if [ "$DRY_RUN" = 1 ]; then
          build_train_cmd train "$m" "$d" "$s"; print_cmd "train $m / $d (seed=$s)"
        else
          build_train_cmd train "$m" "$d" "$s"
          log="$task_log_dir/TRAIN__${m}__${d}__seed${s}.log"
          script="$tmp/task_$idx.sh"
          emit_task_script "$script" "$log" "$gpu_env"
        fi
        ;;
      retrain)
        if [ "$DRY_RUN" = 1 ]; then
          build_retrain_cmd "$m" "$d" "$s"; print_cmd "retrain $m / $d (seed=$s)"
        else
          build_retrain_cmd "$m" "$d" "$s"
          log="$task_log_dir/RETRAIN__${m}__${d}__seed${s}.log"
          script="$tmp/task_$idx.sh"
          emit_task_script "$script" "$log" "$gpu_env"
        fi
        ;;
      eval)
        if ! collect_eval_workloads "$m" "$d" "$folder"; then
          warn "无评估 workload: $m/$d/$folder"
          continue
        fi
        if [ "${#EVAL_WLS[@]}" -eq 0 ]; then
          warn "无评估 workload: $m/$d/$folder"
          continue
        fi
        if [ "$DRY_RUN" = 1 ]; then
          build_eval_cmd "$m" "$d" "$s" "$folder"; print_cmd "predict $m / $d / $folder (seed=$s)"
        else
          build_eval_cmd "$m" "$d" "$s" "$folder"
          log="$task_log_dir/EVAL__${m}__${d}__${folder}__seed${s}.log"
          script="$tmp/task_$idx.sh"
          emit_task_script "$script" "$log" "$gpu_env"
        fi
        ;;
      *) die "未知任务类型: $mode" ;;
    esac
    idx=$((idx+1))
  done < "$queue"

  if [ "$DRY_RUN" = 1 ]; then
    rm -rf "$tmp"
    return 0
  fi
  if [ "$idx" -eq 0 ]; then
    warn "没有需要执行的任务（可能 checkpoint 缺失或已全部完成）"
    rm -rf "$tmp"
    return 0
  fi
  log "共 $idx 个 $mode 任务，并行度 $PARALLEL_SPEC（日志: $LOGS_DIR/$mode）"
  set +e
  find "$tmp" -name 'task_*.sh' -print0 | xargs -0 -P "$PARALLEL_SPEC" -I{} bash -e {}
  rc=$?
  set -e
  rm -rf "$tmp"
  if [ "$rc" -ne 0 ]; then
    die "部分任务失败（exit=$rc）。日志在 $LOGS_DIR/$mode/，请逐条检查。"
  fi
  log "$mode 全部完成"
}

# =============================================================================
# 公共参数解析
# =============================================================================

# 强制清理：仅删除所选 <model>/<db> 的模型与评估输出（必须显式 --force）
force_clean() {
  local mode="$1" models="$2" dbs="$3" m d f
  [ "$DRY_RUN" = 1 ] && { warn "dry-run：--force 的清理不会执行"; return 0; }
  for m in ${models//,/ }; do
    for d in ${dbs//,/ }; do
      [ -e "$MODELS_DIR/$m/$d" ] && { log "删除 $MODELS_DIR/$m/$d"; rm -rf -- "$MODELS_DIR/$m/$d"; }
      [ -e "$EVAL_DIR/$m/$d" ] && { log "删除 $EVAL_DIR/$m/$d"; rm -rf -- "$EVAL_DIR/$m/$d"; }
      # flat 等 tabular 模型把 checkpoint 存为 models/<m>/<db>_<seed>.txt 等散文件
      # （目录清理覆盖不到），这里按 <db>_ 前缀一并清理，不影响其他库。
      for f in "$MODELS_DIR/$m/${d}_"*; do
        [ -e "$f" ] || continue
        log "删除 $f"
        rm -f -- "$f"
      done
    done
  done
}

# 读取带值参数的值：缺值或误把下一个选项当值时给出明确报错，
# 避免 shift 2 越界触发晦涩的 set -e 崩溃。
arg_value() {
  local flag="$1" value="${2:-}"
  [ -n "$value" ] || die "参数 $flag 需要一个值"
  case "$value" in
    --*) die "参数 $flag 需要一个值（得到选项 $value，可能漏传了值）" ;;
  esac
  printf '%s\n' "$value"
}
# 校验非负整数（--parallel / --num-workers）
is_nonneg_int() {
  case "$1" in ''|*[!0-9]*) return 1 ;; esac
  return 0
}

parse_common_args() {
  MODELS_SPEC=""; DBS_SPEC=""; SEEDS_SPEC=""; DEVICE_SPEC="cpu"
  NWORKERS_ARG=""; PARALLEL_SPEC="1"; PARALLEL_EXPLICIT=0; FORCE=0; DRY_RUN=0
  WANDB_PROJECT_ARG=""; WANDB_NAME_ARG=""; EVAL_SET_ARG=""
  TEST_WORKLOADS=(); GPU_IDS_SPEC=""; WITH_EVAL=0; WITH_RETRAIN=0; OUT_ARCHIVE=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --models|-m)
        MODELS_SPEC="$(arg_value "$@")"; shift 2 ;;
      --dbs|-d)
        DBS_SPEC="$(arg_value "$@")"; shift 2 ;;
      --seeds|-s)
        SEEDS_SPEC="$(arg_value "$@")"; shift 2 ;;
      --device)
        DEVICE_SPEC="$(arg_value "$@")"; shift 2 ;;
      --num-workers)
        NWORKERS_ARG="$(arg_value "$@")"
        is_nonneg_int "$NWORKERS_ARG" || die "--num-workers 需要非负整数（得到 $NWORKERS_ARG）"
        shift 2 ;;
      --parallel|-P)
        PARALLEL_SPEC="$(arg_value "$@")"
        is_nonneg_int "$PARALLEL_SPEC" || die "--parallel 需要非负整数（得到 $PARALLEL_SPEC）"
        PARALLEL_EXPLICIT=1; shift 2 ;;
      --gpus)
        GPU_IDS_SPEC="$(arg_value "$@")"; shift 2 ;;
      --force|-f)      FORCE=1; shift ;;
      --dry-run|-n)    DRY_RUN=1; shift ;;
      --wandb-project)
        WANDB_PROJECT_ARG="$(arg_value "$@")"; shift 2 ;;
      --wandb-name)
        WANDB_NAME_ARG="$(arg_value "$@")"; shift 2 ;;
      --eval-set)
        EVAL_SET_ARG="$(arg_value "$@")"; shift 2 ;;
      --test-workload)
        TEST_WORKLOADS+=( "$(arg_value "$@")" ); shift 2 ;;
      --with-eval)     WITH_EVAL=1; shift ;;
      --train-only)    WITH_EVAL=0; shift ;;
      --with-retrain)  WITH_RETRAIN=1; shift ;;
      --out)
        OUT_ARCHIVE="$(arg_value "$@")"; shift 2 ;;
      *) die "未知参数: $1" ;;
    esac
  done
}

# =============================================================================
# 子命令：train / retrain / eval
# =============================================================================
cmd_train() {
  local models dbs seeds m d s queue
  models="$(valid_models "${MODELS_SPEC:-$ALL_MODELS}")"
  dbs="$(valid_dbs "${DBS_SPEC:-$TARGET_DBS}")"
  seeds="$(valid_seeds "${SEEDS_SPEC:-$DEFAULT_SEEDS}")"
  require_env; require_data; require_prepared
  [ "$FORCE" = 1 ] && force_clean train "$models" "$dbs"
  queue="$(mktemp "$SCRIPT_DIR/.queue.XXXXXX")"
  for m in ${models//,/ }; do
    for d in ${dbs//,/ }; do
      for s in ${seeds//,/ }; do
        if [ "$FORCE" != 1 ] && checkpoint_exists "$m" "$d" "$s"; then
          warn "跳过 $m/$d seed=$s：checkpoint 已存在（--force 可重训）"
          continue
        fi
        printf '%s\t%s\t%s\n' "$m" "$d" "$s" >> "$queue"
      done
    done
  done
  log "训练: models=$models dbs=$dbs seeds=$seeds device=$DEVICE_SPEC parallel=$PARALLEL_SPEC"
  run_task_queue train "$queue"
  rm -f "$queue"
}

cmd_retrain() {
  # 论文 T4 微调：在 imdb 上对 index_retraining + seq_retraining 做 --mode retrain。
  # 需要 base checkpoint 已存在于 models/<model>/imdb/（非 flat 模型）。
  # 官方 exp_retrain_model.py 直接从 retraining_models 目录加载 checkpoint 续训，
  # 因此脚本先把 base checkpoint 复制到 models/retrained/<model>/imdb/ 再执行官方命令。
  local models dbs seeds m d s queue base
  models="$(valid_models "${MODELS_SPEC:-$RETRAIN_DEFAULT_MODELS}")"
  dbs="$(valid_dbs "${DBS_SPEC:-imdb}")"
  seeds="$(valid_seeds "${SEEDS_SPEC:-$DEFAULT_SEEDS}")"
  require_env; require_data; require_prepared
  # 官方 main.py retrain 分支的已知限制：QLoss 模型默认 final_mlp_kwargs=None，
  # 在 main.py:370 直接抛 AttributeError（官方 exp_retrain_model.py 也未启用这些模型）。
  # 仅提示不阻断，命令仍与官方完全一致（让用户看到官方真实行为）。
  local unsupported=""
  for m in ${models//,/ }; do
    is_in_list "$m" "flat,e2e,mscn,query_former" && unsupported="${unsupported:+$unsupported,}$m"
  done
  if [ -n "$unsupported" ]; then
    warn "官方 main.py 的 retrain 分支对 $unsupported 会因 final_mlp_kwargs=None 在 main.py:370 崩溃"
    warn "（官方 exp_retrain_model.py 只启用了 zeroshot）。命令仍按官方生成，可能失败。"
  fi
  for d in ${dbs//,/ }; do
    [ "$d" = imdb ] || die "retrain 仅支持 imdb（官方 T4 实验范围）"
  done
  # 复制 base checkpoint（非 flat 神经模型需要）
  if [ "$DRY_RUN" != 1 ]; then
    for m in ${models//,/ }; do
      for s in ${seeds//,/ }; do
        if [ "$m" = flat ]; then continue; fi
        if ! checkpoint_exists "$m" imdb "$s"; then
          die "retrain 需要 base checkpoint: models/$m/imdb/${m}_$s.pt（请先 train $m imdb）"
        fi
        base="$MODELS_DIR/$m/imdb"
        mkdir -p "$RETRAIN_MODELS_DIR/$m/imdb"
        cp -a "$base/." "$RETRAIN_MODELS_DIR/$m/imdb/"
        log "已复制 base checkpoint: $base -> $RETRAIN_MODELS_DIR/$m/imdb"
      done
    done
  fi
  queue="$(mktemp "$SCRIPT_DIR/.queue.XXXXXX")"
  for m in ${models//,/ }; do
    for d in ${dbs//,/ }; do
      for s in ${seeds//,/ }; do
        printf '%s\t%s\t%s\n' "$m" "$d" "$s" >> "$queue"
      done
    done
  done
  log "retrain: models=$models dbs=$dbs seeds=$seeds device=$DEVICE_SPEC（官方 --mode retrain，epochs+100）"
  run_task_queue retrain "$queue"
  rm -f "$queue"
}

# 论文评估 workload 目录（按库；core/paper 为论文全部 E1-E4）
eval_folders_for() {
  local db="$1" set_csv="$2" out="" f
  for f in ${set_csv//,/ }; do
    case "$f" in
      core|paper)
        if [ "$db" = imdb ]; then
          out="$out join_order_full scan_costs_percentiles physical_plan physical_index_plan"
        else
          out="$out scan_costs_percentiles physical_plan"
        fi ;;
      join_order_full)        [ "$db" = imdb ] && out="$out join_order_full" ;;
      scan_costs_percentiles|physical_plan|physical_index_plan) out="$out $f" ;;
    esac
  done
  out="$(printf '%s\n' $out | awk 'NF && !seen[$0]++' | paste -sd' ' -)"
  printf '%s\n' "$out"
}

cmd_eval() {
  # 官方 --mode predict：按文件夹把该文件夹全部 *.json workload 一次传给
  # --test_workload_runs（与 exp_predict_all.py 的分组方式一致）。
  local models dbs seeds m d s folder queue root
  models="$(valid_models "${MODELS_SPEC:-$ALL_MODELS}")"
  dbs="$(valid_dbs "${DBS_SPEC:-$TARGET_DBS}")"
  seeds="$(valid_seeds "${SEEDS_SPEC:-$DEFAULT_SEEDS}")"
  validate_eval_set "${EVAL_SET_ARG:-core}"
  require_env; require_data; require_prepared
  queue="$(mktemp "$SCRIPT_DIR/.queue.XXXXXX")"
  for m in ${models//,/ }; do
    for d in ${dbs//,/ }; do
      for s in ${seeds//,/ }; do
        if ! checkpoint_exists "$m" "$d" "$s"; then
          warn "跳过 $m/$d seed=$s：缺少 checkpoint"
          continue
        fi
        if [ "${#TEST_WORKLOADS[@]}" -gt 0 ]; then
          printf '%s\t%s\t%s\tcustom\n' "$m" "$d" "$s" >> "$queue"
          continue
        fi
        root="$(input_root_of "$m")"
        for folder in $(eval_folders_for "$d" "${EVAL_SET_ARG:-core}"); do
          [ -n "$(find "$root/$d/$folder" -maxdepth 1 -name '*.json' 2>/dev/null | head -1)" ] \
            || { warn "无 workload 目录 $root/$d/$folder"; continue; }
          printf '%s\t%s\t%s\t%s\n' "$m" "$d" "$s" "$folder" >> "$queue"
        done
      done
    done
  done
  log "评估(predict): models=$models dbs=$dbs seeds=$seeds eval-set=${EVAL_SET_ARG:-core}（官方 predict 默认 cpu）"
  run_task_queue eval "$queue"
  rm -f "$queue"
}

# =============================================================================
# 子命令：setup（环境 + 数据 + 派生统计）
# =============================================================================

version_ge() {
  [ "$(printf '%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]
}
version_le() {
  [ "$(printf '%s\n' "$1" "$2" | sort -V | tail -1)" = "$2" ]
}

ensure_uv() {
  if [ -x "$UV" ]; then
    log "uv 已就绪: $UV"
    return 0
  fi
  need_cmd curl
  mkdir -p "$TOOLS_DIR"
  local installer="$TOOLS_DIR/uv-install.sh"
  log "下载并安装 uv（到 $TOOLS_DIR）..."
  curl -fsSL https://astral.sh/uv/install.sh -o "$installer"
  UV_INSTALL_DIR="$TOOLS_DIR" sh "$installer"
  [ -x "$UV" ] || die "uv 安装失败（$UV 不存在）"
  log "uv 已安装: $UV"
}

# 统一安装入口：有 uv 用 uv pip，否则用 venv 自带 pip
pipi() {
  local py="$1"; shift
  if [ -x "$UV" ]; then
    "$UV" pip install --python "$py" "$@"
  else
    "$py" -m pip install "$@"
  fi
}

setup_venv() {
  # 只复用能真正运行的解释器：断链/失效的 .venv（例如目录被整体拷贝后 venv 内
  # python 指向旧机器的 uv 缓存）会被识别为不可用并自动重建。
  if [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -c 'import sys' >/dev/null 2>&1; then
    log "Python 环境已存在: $VENV_DIR"
    return 0
  fi
  # 移除不可用的旧环境（仅限脚本自己管理的 $SCRIPT_DIR/.venv），避免重建失败
  if [ -e "$VENV_DIR" ]; then
    warn "旧环境不可用（$VENV_DIR），移除并重建"
    rm -rf -- "$VENV_DIR"
  fi
  local sysver=""
  if command -v python3 >/dev/null 2>&1; then
    sysver="$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  fi
  if [ -n "$sysver" ] && version_ge "$sysver" 3.9 && version_le "$sysver" 3.11; then
    log "使用系统 python3 $sysver 创建 venv（对应官方指令 python3 -m venv .venv）"
    if python3 -m venv "$VENV_DIR"; then
      return 0
    fi
    warn "python3 -m venv 创建失败，改用 uv 安装 Python 3.10"
  else
    warn "系统 Python（$sysver）低于 3.9 或不适用；通过 uv 安装 Python 3.10（与官方代码 list[...] 注解兼容）"
  fi
  ensure_uv
  log "通过 uv 安装 Python 3.10 并创建 venv..."
  "$UV" venv --python 3.10 "$VENV_DIR"
}

# 兼容依赖集（v3 已验证可完整跑通官方 main.py）：
#   local = CPU torch + CPU dgl；gpu = cu121 torch + cu121 dgl(cp310)
# 官方 requirements.txt 的 dgl 是 cp312 wheel，除 Python 3.12 外无法安装，
# 因此这里用与 v3 一致的固定版本组合（torch/dgl/numpy 版本与官方声明一致）。
install_compatible_deps() {
  local mode="$1" py="$2" req
  log "安装运行依赖（torch + dgl + 必备包，mode=$mode）..."
  if [ "$mode" = gpu ]; then
    pipi "$py" 'torch==2.3.0' 'torchvision==0.18.0' --index-url https://download.pytorch.org/whl/cu121
    pipi "$py" 'https://data.dgl.ai/wheels/torch-2.3/cu121/dgl-2.4.0%2Bcu121-cp310-cp310-manylinux1_x86_64.whl'
  else
    pipi "$py" 'torch==2.3.0' 'torchvision==0.18.0' --index-url https://download.pytorch.org/whl/cpu
    pipi "$py" 'dgl==1.1.3'
  fi
  req="$(mktemp)"
  cat > "$req" <<'DEPS_EOF'
attrs==25.3.0
filelock==3.16.1
gensim==4.3.3
joblib==1.4.2
lightgbm==4.6.0
loralib==0.1.2
networkx==3.1
numpy==1.24.4
optuna==4.5.0
pandas==2.0.3
psutil==7.0.0
psycopg2-binary==2.9.10
pydantic==2.10.6
python-dotenv==1.0.1
PyYAML==6.0.3
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.13.2
torchdata==0.9.0
tqdm==4.67.1
wandb==0.21.1
osfclient==0.0.5
DEPS_EOF
  pipi "$py" -r "$req"
  rm -f "$req"
}

verify_import() {
  local py; py="$(python_bin)"
  export_node_env
  log "验证官方 main.py 可导入（加载全部模型代码）..."
  # 用 -B 禁止写入 __pycache__，避免污染官方仓库 src/
  ( cd "$SRC_DIR" && PYTHONDONTWRITEBYTECODE=1 "$py" -B -c 'import main; print("main.py 导入成功")' )
}

download_runs() {
  if [ -d "$RUNS_DIR/parsed_plans" ]; then
    log "数据已存在: $RUNS_DIR"
    return 0
  fi
  local py; py="$(python_bin)"
  [ -x "$py" ] || die "需要先完成环境安装"
  export_node_env
  log "通过官方 download_from_osf.py 下载 OSF runs 数据（约 1.4GB，解压后约 10GB，请耐心等待）..."
  ( cd "$SRC_DIR" && "$py" download_from_osf.py --artifacts runs )
  [ -d "$RUNS_DIR/parsed_plans" ] || die "OSF 下载后未找到 $RUNS_DIR/parsed_plans"
}

link_runs() {
  local src="$1" cur
  [ -n "$src" ] || die "--link 需要指定现有 runs 目录"
  [ -d "$src/parsed_plans" ] || die "链接源缺少 parsed_plans: $src"
  src="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"
  mkdir -p "$DATA_DIR"
  if [ -L "$RUNS_DIR" ]; then
    cur="$(readlink -f "$RUNS_DIR")"
    if [ "$cur" != "$src" ]; then
      rm -f "$RUNS_DIR"
      ln -s "$src" "$RUNS_DIR"
    fi
  elif [ -d "$RUNS_DIR" ]; then
    die "$RUNS_DIR 已存在且不是链接，请手动处理（可能已有下载数据）"
  else
    ln -s "$src" "$RUNS_DIR"
  fi
  log "data/runs -> $src（只读复用，不复制任何文件）"
}

cmd_setup() {
  local link_src=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --link) link_src="${2:-}"; shift 2 ;;
      -h|--help) cmd_help setup; return 0 ;;
      *) die "setup 未知参数: $1（支持 --link <runs目录>）" ;;
    esac
  done
  # 1) 官方仓库（只读）
  if [ -d "$SRC_DIR" ]; then
    log "官方仓库已存在: $REPO_DIR"
  else
    need_cmd git
    log "克隆官方仓库 DataManagementLab/lcm-eval ..."
    git clone --depth 1 https://github.com/DataManagementLab/lcm-eval.git "$REPO_DIR"
  fi
  [ -f "$SRC_DIR/main.py" ] || die "官方仓库缺少 src/main.py（$SRC_DIR）"
  # 2) venv + 依赖
  setup_venv
  install_compatible_deps local "$VENV_DIR/bin/python"
  verify_import
  # 3) 数据
  if [ -n "$link_src" ]; then
    link_runs "$link_src"
  else
    download_runs
  fi
  require_data
  # 4) 派生统计（只写 prepared/，data/runs 保持只读）
  log "生成派生统计文件到 prepared/ ..."
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/prepare_stats.py" --runs "$RUNS_DIR" --prepared "$PREPARED_DIR"
  require_prepared
  log "setup 完成。下一步：bash ./reproduce.sh train（本地 CPU）或 ./reproduce.sh gpu-bundle（打包远端 GPU）"
}

# =============================================================================
# 子命令：gpu-bundle / gpu-setup / gpu-run / gpu-pack / import-results
# =============================================================================

# 把本地 data/runs 里本次训练+评估所需的数据选择性复制到 bundle（不复制多余文件）。
bundle_data() {
  local dest="$1" models="$2" dbs="$3" with_retrain="$4" db folder m
  local need_json=0 need_aug=0
  for m in ${models//,/ }; do
    [ "$m" = qppnet ] && need_json=1
    is_in_list "$m" "mscn,e2e,query_former" && need_aug=1
  done
  # parsed_plans：全部 20 库训练 workload（agnostic 训练需要）+ 统计源文件
  mkdir -p "$dest/parsed_plans"
  for db in "${ALL_DB_WORDS[@]}"; do
    mkdir -p "$dest/parsed_plans/$db"
    cp -a "$RUNS_DIR/parsed_plans/$db/workload_100k_s1_c8220.json" "$dest/parsed_plans/$db/"
  done
  cp -a "$RUNS_DIR/parsed_plans/statistics_workload_combined.json" "$dest/parsed_plans/"
  # 目标库的评估/重训目录（parsed_plans）
  for db in ${dbs//,/ }; do
    for folder in join_order_full scan_costs_percentiles physical_plan physical_index_plan; do
      [ -d "$RUNS_DIR/parsed_plans/$db/$folder" ] && cp -a "$RUNS_DIR/parsed_plans/$db/$folder" "$dest/parsed_plans/$db/"
    done
    if [ "$with_retrain" = 1 ]; then
      [ -d "$RUNS_DIR/parsed_plans/$db/index_retraining" ] && cp -a "$RUNS_DIR/parsed_plans/$db/index_retraining" "$dest/parsed_plans/$db/"
      [ -d "$RUNS_DIR/parsed_plans/$db/seq_retraining" ] && cp -a "$RUNS_DIR/parsed_plans/$db/seq_retraining" "$dest/parsed_plans/$db/"
    fi
  done
  # json（qppnet）：整目录复制（含 workload、feature_statistics_combined、评估/重训）
  if [ "$need_json" = 1 ]; then
    mkdir -p "$dest/json"
    for db in ${dbs//,/ }; do
      [ -d "$RUNS_DIR/json/$db" ] && cp -a "$RUNS_DIR/json/$db" "$dest/json/"
    done
  fi
  # augmented_plans_baseline + sentences（mscn/e2e/query_former）
  if [ "$need_aug" = 1 ]; then
    mkdir -p "$dest/augmented_plans_baseline" "$dest/sentences"
    for db in ${dbs//,/ }; do
      [ -d "$RUNS_DIR/augmented_plans_baseline/$db" ] && cp -a "$RUNS_DIR/augmented_plans_baseline/$db" "$dest/augmented_plans_baseline/"
      [ -d "$RUNS_DIR/sentences/$db" ] && cp -a "$RUNS_DIR/sentences/$db" "$dest/sentences/"
    done
  fi
}

cmd_gpu_bundle() {
  parse_common_args "$@"
  local models dbs seeds stamp out root
  models="$(valid_models "${MODELS_SPEC:-$ALL_MODELS}")"
  dbs="$(valid_dbs "${DBS_SPEC:-$TARGET_DBS}")"
  seeds="$(valid_seeds "${SEEDS_SPEC:-$DEFAULT_SEEDS}")"
  require_data; require_prepared
  stamp="$(date +%Y%m%d_%H%M%S)"
  out="${OUT_ARCHIVE:-$OUTGOING_DIR/lcm_gpu_bundle_$stamp.tar.gz}"
  case "$out" in
    /*) ;;
    *) out="$SCRIPT_DIR/$out" ;;
  esac
  mkdir -p "$OUTGOING_DIR"
  [ -e "$out" ] && die "目标已存在，拒绝覆盖: $out"
  build="$(mktemp -d "$SCRIPT_DIR/.gpu-bundle.XXXXXX")"
  root="$build/lcm_gpu_bundle"
  trap 'rm -rf -- "${build:-}"' EXIT
  mkdir -p "$root"
  log "打包 GPU bundle（models=$models dbs=$dbs seeds=$seeds with-eval=$WITH_EVAL with-retrain=$WITH_RETRAIN）..."
  # 官方源码（排除 .git / __pycache__）
  ( cd "$REPO_DIR/.." && tar --exclude='.git' --exclude='__pycache__' -chf - "$(basename "$REPO_DIR")" ) | ( cd "$root" && tar -xf - )
  # 本脚本 + 派生统计脚本
  cp -a "$SCRIPT_DIR/reproduce.sh" "$SCRIPT_DIR/prepare_stats.py" "$root/"
  # 数据（选择性）
  bundle_data "$root/data/runs" "$models" "$dbs" "$WITH_RETRAIN"
  # 派生统计（precomputed）
  mkdir -p "$root/prepared/qpp_stats"
  cp -a "$PREPARED_DIR/statistics_complex_workload_combined.json" "$root/prepared/"
  cp -a "$PREPARED_DIR/statistics_zeroshot_train.json" "$root/prepared/" 2>/dev/null || true
  cp -a "$PREPARED_DIR/qpp_stats/." "$root/prepared/qpp_stats/" 2>/dev/null || true
  # 结果目录（空）
  mkdir -p "$root/models" "$root/evaluation" "$root/logs" "$root/outgoing" "$root/gpu_results"
  # 一键脚本
  local with_eval_arg="" with_retrain_arg=""
  [ "$WITH_EVAL" = 1 ] && with_eval_arg=" --with-eval"
  [ "$WITH_RETRAIN" = 1 ] && with_retrain_arg=" --with-retrain"
  cat > "$root/gpu-run.sh" <<GPUEOF
#!/usr/bin/env bash
set -euo pipefail
exec ./reproduce.sh gpu-run --models "$models" --dbs "$dbs" --seeds "$seeds" --device cuda:0${with_eval_arg}${with_retrain_arg}
GPUEOF
  chmod +x "$root/gpu-run.sh"
  # sha256 清单 + tar.gz
  # 清单不能包含自身：重定向会先创建空的 BUNDLE_SHA256SUMS，若 find 将其纳入，
  # 清单首行会记录空文件哈希，写完后该项必然校验失败。
  ( cd "$build" && find lcm_gpu_bundle -type f ! -name BUNDLE_SHA256SUMS | sort | xargs sha256sum > "$root/BUNDLE_SHA256SUMS" )
  ( cd "$build" && tar -czf "$out" lcm_gpu_bundle )
  # 校验文件必须只记录归档文件名，不能写本机绝对路径；这样上传到远端后
  # 才能在归档所在目录直接执行 sha256sum -c。
  ( cd "$(dirname "$out")" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" )
  rm -rf -- "$build"
  trap - EXIT
  log "GPU bundle 已生成: $out"
  log "SHA256: $(awk '{print $1}' "$out.sha256")"
  log "远端使用：上传 $out（及其 .sha256），然后"
  log "  sha256sum -c lcm_gpu_bundle_*.tar.gz.sha256 && tar -xzf lcm_gpu_bundle_*.tar.gz && cd lcm_gpu_bundle && bash gpu-run.sh"
}

cmd_gpu_setup() {
  ensure_uv
  if [ ! -x "$VENV_GPU_DIR/bin/python" ]; then
    log "创建 GPU venv（Python 3.10）..."
    "$UV" venv --python 3.10 "$VENV_GPU_DIR"
  fi
  install_compatible_deps gpu "$VENV_GPU_DIR/bin/python"
  export DGLBACKEND=pytorch
  log "验证 CUDA ..."
  "$VENV_GPU_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA 不可用"; print("CUDA OK:", torch.cuda.get_device_name(0))'
  log "gpu-setup 完成"
}

cmd_gpu_run() {
  parse_common_args "$@"
  local models dbs seeds
  models="$(valid_models "${MODELS_SPEC:-$ALL_MODELS}")"
  dbs="$(valid_dbs "${DBS_SPEC:-$TARGET_DBS}")"
  seeds="$(valid_seeds "${SEEDS_SPEC:-$DEFAULT_SEEDS}")"
  export LCM_GPU_HOST=1
  if [ ! -x "$VENV_GPU_DIR/bin/python" ]; then
    log "GPU 环境未安装，自动执行 gpu-setup"
    cmd_gpu_setup
  fi
  "$VENV_GPU_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA 不可用"'
  [ "$DEVICE_SPEC" = cpu ] && DEVICE_SPEC="cuda:0"
  if [ "$PARALLEL_EXPLICIT" != 1 ]; then
    if [ -n "$GPU_IDS_SPEC" ]; then
      PARALLEL_SPEC="$(printf '%s\n' ${GPU_IDS_SPEC//,/ } | grep -c . || true)"
      PARALLEL_SPEC="${PARALLEL_SPEC:-1}"
    else
      PARALLEL_SPEC=1
    fi
  fi
  log "GPU 训练开始：models=$models dbs=$dbs seeds=$seeds gpus=${GPU_IDS_SPEC:-0} parallel=$PARALLEL_SPEC"
  cmd_train
  if [ "$WITH_EVAL" = 1 ]; then
    log "GPU 训练完成，开始评估（predict，官方默认 cpu）..."
    cmd_eval
  fi
  if [ "$WITH_RETRAIN" = 1 ]; then
    log "开始 retrain（T4）..."
    cmd_retrain
  fi
  cmd_gpu_pack
}

cmd_gpu_pack() {
  local stamp out list
  stamp="$(date +%Y%m%d_%H%M%S)"
  out="${OUT_ARCHIVE:-$OUTGOING_DIR/lcm_gpu_results_$stamp.tar.gz}"
  case "$out" in
    /*) ;;
    *) out="$SCRIPT_DIR/$out" ;;
  esac
  mkdir -p "$OUTGOING_DIR"
  [ -e "$out" ] && die "拒绝覆盖已有结果包: $out"
  list="$(mktemp "$SCRIPT_DIR/.packlist.XXXXXX")"
  ( cd "$SCRIPT_DIR" && { find models evaluation logs -type f 2>/dev/null || true; } | sort > "$list" )
  if [ ! -s "$list" ]; then
    warn "没有可打包的结果（models/evaluation/logs 为空）"
    rm -f "$list"
    return 1
  fi
  log "打包结果（$(wc -l < "$list") 个文件）-> $out"
  ( cd "$SCRIPT_DIR" && tar -czf "$out" -T "$list" )
  ( cd "$(dirname "$out")" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" )
  rm -f "$list"
  log "结果包: $out"
  log "SHA256: $(awk '{print $1}' "$out.sha256")"
}

cmd_import_results() {
  local archive="${1:-}" tmp
  [ -n "$archive" ] || die "用法: import-results <lcm_gpu_results_*.tar.gz>"
  [ -f "$archive" ] || die "找不到归档: $archive"
  if [ -f "$archive.sha256" ]; then
    log "校验 sha256 ..."
    ( cd "$(dirname "$archive")" && sha256sum -c "$(basename "$archive").sha256" )
  fi
  tmp="$(mktemp -d "$SCRIPT_DIR/.import.XXXXXX")"
  tar -xzf "$archive" -C "$tmp"
  if [ ! -d "$tmp/models" ] && [ ! -d "$tmp/evaluation" ]; then
    rm -rf "$tmp"
    die "归档内容异常（缺少 models/ 或 evaluation/）"
  fi
  log "导入结果到 models/、evaluation/、logs/（--no-clobber，拒绝覆盖已有结果）..."
  if [ -d "$tmp/models" ]; then mkdir -p "$MODELS_DIR"; cp -an "$tmp/models/." "$MODELS_DIR/"; fi
  if [ -d "$tmp/evaluation" ]; then mkdir -p "$EVAL_DIR"; cp -an "$tmp/evaluation/." "$EVAL_DIR/"; fi
  if [ -d "$tmp/logs" ]; then mkdir -p "$LOGS_DIR"; cp -an "$tmp/logs/." "$LOGS_DIR/"; fi
  rm -rf "$tmp"
  log "导入完成。可运行 ./reproduce.sh eval 继续补评估，或直接分析 models/ 与 evaluation/。"
}

# =============================================================================
# 子命令：list-models / help / 主分发
# =============================================================================

cmd_list_models() {
  printf '%-14s %-8s %-22s %s\n' '模型' '类型' '输入表示' '额外参数'
  local m type input extra
  for m in ${ALL_MODELS//,/ }; do
    if model_is_agnostic "$m"; then type="agnostic"; else type="driven"; fi
    case "$m" in
      flat|zeroshot|dace) input="parsed_plans" ;;
      mscn|e2e|query_former) input="augmented_plans_baseline" ;;
      qppnet) input="json" ;;
    esac
    case "$m" in
      mscn|e2e|query_former) extra="+column_statistics +word_embeddings" ;;
      qppnet) extra="+column_statistics" ;;
      zeroshot) extra="+hyperparameter_path" ;;
      *) extra="-" ;;
    esac
    printf '%-14s %-8s %-22s %s\n' "$m" "$type" "$input" "$extra"
  done
}

cmd_help() {
  local topic="${1:-}"
  if [ -n "$topic" ]; then
    case "$topic" in
      setup)
        cat <<'H_EOF'
setup [--link <runs目录>]
  1) 克隆官方仓库 lcm-eval/（已存在则跳过）
  2) 创建 Python 环境（系统 python3 3.9-3.11，否则 uv 安装 Python 3.10）
    并安装与官方一致的 torch 2.3.0 / dgl / 运行依赖
  3) 下载 OSF runs 数据（约 1.4GB）到 data/runs；或用 --link 复用已有 runs（只读软链接）
  4) 生成派生统计文件到 prepared/（官方缺失的文件，data/runs 保持只读）
H_EOF
        ;;
      *) cmd_help ;;
    esac
    return 0
  fi
  cat <<'HELP_EOF'
用法: ./reproduce.sh <子命令> [参数]

setup [--link <runs目录>]
    克隆官方仓库、创建 Python 环境、下载/链接 OSF runs 数据、生成派生统计到 prepared/。

train [--models a,b,c] [--dbs imdb,baseball,tpc_h_pk] [--seeds 0,1,2]
      [--device cpu|cuda:N] [--parallel N] [--gpus 0,1] [--force] [--dry-run]
      [--wandb-project X] [--wandb-name X] [--num-workers N] [--with-eval]
    按官方 exp_train_model.py 生成 --mode train 命令并并行执行；
    加 --with-eval 训练完成后自动按官方 exp_predict_all.py 评估。

eval | predict [--models ...] [--dbs ...] [--seeds ...] [--eval-set core|paper|...]
      [--test-workload <file>] [--parallel N] [--dry-run]
    按官方 exp_predict_all.py 生成 --mode predict（每个文件夹一个命令）。

retrain [--models zeroshot] [--seeds 0,1,2] [--dry-run]
    官方 T4：在 imdb 上 --mode retrain（index/seq retraining），默认 zeroshot
    （对应官方 exp_retrain_model.py as-is 只启用的模型），需先 train 对应模型。

gpu-bundle [--models ...] [--dbs ...] [--seeds ...] [--with-eval] [--with-retrain]
      [--out <file.tar.gz>]
    打包自包含的远端 GPU 训练(+评估) 包到 outgoing/（含官方源码、必要数据、脚本）。

gpu-setup   在远端 GPU 主机安装环境（uv + Python 3.10 + torch cu121 + dgl）。
gpu-run [--models ...] [--dbs ...] [--seeds ...] [--gpus 0,1] [--with-eval]
      [--with-retrain]
    在远端 GPU 主机一键：gpu-setup(如需) -> train -> eval(可选) -> gpu-pack。
gpu-pack [--out <file.tar.gz>]
    把 models/evaluation/logs 打成 tar.gz + sha256 到 outgoing/。
import-results <lcm_gpu_results_*.tar.gz>
    校验 sha256 并导入本地 models/、evaluation/、logs/（拒绝覆盖）。

list-models   列出支持的模型与元数据。
help          显示本帮助。

示例：
  bash reproduce.sh setup --link /root/lcm模型/version3/data/raw/runs
  bash reproduce.sh train --models flat --dbs imdb --seeds 0
  bash reproduce.sh train --models mscn,e2e --device cuda:0 --parallel 2
  bash reproduce.sh eval --models flat --dbs imdb --seeds 0
  bash reproduce.sh gpu-bundle --models all --with-eval
HELP_EOF
}

main() {
  local sub="${1:-help}"
  shift || true
  case "$sub" in
    setup)          cmd_setup "$@" ;;
    train)          parse_common_args "$@"; cmd_train
                    if [ "$WITH_EVAL" = 1 ]; then
                      log "train 完成，按 --with-eval 继续评估..."
                      cmd_eval
                    fi
                    ;;
    eval|predict)   parse_common_args "$@"; cmd_eval ;;
    retrain)        parse_common_args "$@"; cmd_retrain ;;
    gpu-bundle)     cmd_gpu_bundle "$@" ;;
    gpu-setup)      cmd_gpu_setup "$@" ;;
    gpu-run)        cmd_gpu_run "$@" ;;
    gpu-pack)       parse_common_args "$@"; cmd_gpu_pack ;;
    import-results) cmd_import_results "$@" ;;
    list-models)    cmd_list_models ;;
    help|-h|--help) cmd_help ;;
    *) die "未知子命令: $sub（运行 $0 help）" ;;
  esac
}
main "$@"
