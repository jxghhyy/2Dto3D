#!/bin/bash
set -e

# ========================================================
# 2D转3D 完整运行脚本
# 支持灵活配置参数和Plan A-E时序优化组合
# ========================================================

# ---------------- 默认参数配置 ----------------
VIDEO_PATH="../../dataset/shuai/horse.mp4"
OUTPUT=""
OUTDIR="./output_plan"
ENCODER="vits"          # vits, vitb, vitl, vitg
INPUT_SIZE=518          # 深度推理分辨率 (14的倍数)
DIBR_SIZE=294          # DIBR渲染分辨率 (0=同input-size, -1=原分辨率)
MAX_DISPARITY=24        # 最大视差
DEPTH_MODE="inverse"     # metric, inverse
FP16=true               # 启用FP16
LAYOUT="sbs"            # sbs, ou, overlay, anaglyph

# Plan A-E 时序稳定参数 (默认全开推荐配置)
QUANTILE_SMOOTH=0.8     # Plan A: 分位数EMA (0=禁用)
DEPTH_SMOOTH=0.6        # Plan B+C: near空间EMA (0=禁用)
RGB_MOTION_SIGMA=0.1    # Plan B: RGB运动敏感度 (0=禁用自适应)
FLOW_ALIGN=false        # Plan D: 光流对齐 (true/false)
FLOW_HEIGHT=144         # Plan D: 光流计算分辨率
MEDIAN_WINDOW=3         # Plan E: 滑窗中值 (1=禁用, 3/5/7推荐)
HOLE_DILATE_LEFT=2      # 空洞左侧膨胀 (保护前景)
HOLE_DILATE_RIGHT=1     # 空洞右侧膨胀 (消除轮廓线)

# 空洞修补参数
FAST_KERNEL=11           # 补洞卷积核大小 (奇数)
FAST_MAX_ITER=64      # 补洞最大迭代次数

# 边缘敏感补洞参数 (分别尺寸核卷积)
BG_THRESHOLD=0.3        # 背景深度阈值 (只near<该值算背景)
EDGE_KERNEL=5           # 边缘区域用小卷积核
NON_EDGE_KERNEL=11      # 非边缘区域用大卷积核
EDGE_FILL_MODE=0        # 边缘填充模式: 0=边缘小核/非边缘大核, 1=边缘直接填补, 2=混合模式

# 其他参数
PROFILE_TIME=true       # 输出性能统计
PROFILE_OUTPUT=""       # 性能统计保存路径 (默认与视频同目录同名.txt)
VIDEO_ENCODER="h264_nvenc"  # h264_nvenc, libx264
VIDEO_LIST=""           # 视频列表文件
ONLY_PLAN=""            # 只运行指定的Plan配置

# ---------------- 运行模式 ----------------
MODE="custom"           # custom: 自定义配置; all: 测试所有Plan组合; ablations: Plan消融实验

# ========================================================
# 用法帮助
# ========================================================
show_help() {
    cat << EOF
用法: $(basename "$0") [选项]

必要参数:
  -i, --input <路径>     输入视频/目录/txt列表
  -o, --output <路径>    输出视频 (单文件模式)
  --outdir <目录>        输出目录 (批量模式, 默认: $OUTDIR)

模型与分辨率:
  -e, --encoder <型号>   模型: vits/vitb/vitl/vitg (默认: $ENCODER)
  --input-size <数值>    深度推理分辨率 (默认: $INPUT_SIZE, 需14的倍数)
  --dibr-size <数值>     DIBR渲染分辨率 (0=同input-size, -1=原分辨率, 默认: $DIBR_SIZE)

3D效果参数:
  -d, --max-disparity <数值>  最大视差像素 (默认: $MAX_DISPARITY)
  --depth-mode <模式>         metric/inverse (默认: $DEPTH_MODE)
  --layout <模式>             sbs/ou/overlay/anaglyph (默认: $LAYOUT)

Plan A-E 时序稳定 (分别控制):
  --plan-a <数值>       分位数EMA强度 (0=禁用, 0.8推荐)
  --plan-bc <数值>      near空间EMA强度 (0=禁用, 0.6推荐)
  --plan-b-sigma <数值> RGB运动敏感度 (0=禁用自适应, 0.1推荐)
  --plan-d <开关>       光流对齐 (true/false)
  --plan-d-height <数值>光流计算分辨率 (默认: $FLOW_HEIGHT)
  --plan-e <数值>       滑窗中值窗口 (1=禁用, 3推荐)

空洞修补:
  --fast-kernel <数值>   补洞卷积核大小 (奇数, 默认: $FAST_KERNEL)
  --fast-max-iter <数值> 补洞最大迭代次数 (默认: $FAST_MAX_ITER)

边缘敏感补洞 (分别尺寸核卷积):
  --hole-dilate-right <数值> 空洞右侧膨胀像素 (消除轮廓线, 默认: $HOLE_DILATE_RIGHT)
  --bg-threshold <数值>      背景深度阈值 (仅near<该值算背景, 默认: $BG_THRESHOLD)
  --edge-kernel <数值>       边缘区域用小卷积核 (默认: $EDGE_KERNEL)
  --non-edge-kernel <数值>   非边缘区域用大卷积核 (默认: $NON_EDGE_KERNEL)
  --edge-fill-mode <数值>    边缘填充模式: 0=边缘小核/非边缘大核, 1=边缘直接填补, 2=混合模式 (默认: $EDGE_FILL_MODE)

快捷模式:
  --no-plans            禁用所有Plan (纯单帧模式)
  --all-plans           启用所有Plan (推荐配置)
  --only-plan <名称>   只运行指定配置: no_plan/plan_a/plan_bc/plan_d/plan_e/plan_ab/plan_abc_e/plan_abcd/all_plans
  --mode <模式>         运行模式: custom/all/ablations
                        - custom: 使用当前参数配置
                        - all: 循环测试多种Plan组合
                        - ablations: Plan消融实验 (逐个添加/移除)

多视频:
  --video-list <文件>   包含多个视频路径的txt文件 (每行一个视频)
                        或用空格分隔多个视频: -i "v1.mp4 v2.mp4"

其他:
  --fp16 / --no-fp16    开关FP16 (默认: 开)
  --profile / --no-profile  开关性能统计 (默认: 开)
  --profile-output <路径>  性能统计保存位置 (默认与视频同目录同名.txt)
  -h, --help            显示帮助

示例:
  # 基本用法
  $(basename "$0") -i input.mp4 -o output.mp4

  # 禁用所有Plan
  $(basename "$0") -i input.mp4 -o output.mp4 --no-plans

  # 仅用Plan A
  $(basename "$0") -i input.mp4 -o output.mp4 --plan-a 0.8 --plan-bc 0 --plan-e 1

  # 测试所有Plan组合
  $(basename "$0") -i input.mp4 --mode all --outdir ./results_all_plans

EOF
}

# ========================================================
# 解析命令行参数
# ========================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -i|--input) VIDEO_PATH="$2"; shift 2 ;;
            -o|--output) OUTPUT="$2"; shift 2 ;;
            --outdir) OUTDIR="$2"; shift 2 ;;
            -e|--encoder) ENCODER="$2"; shift 2 ;;
            --input-size) INPUT_SIZE="$2"; shift 2 ;;
            --dibr-size) DIBR_SIZE="$2"; shift 2 ;;
            -d|--max-disparity) MAX_DISPARITY="$2"; shift 2 ;;
            --depth-mode) DEPTH_MODE="$2"; shift 2 ;;
            --layout) LAYOUT="$2"; shift 2 ;;

            --plan-a) QUANTILE_SMOOTH="$2"; shift 2 ;;
            --plan-bc) DEPTH_SMOOTH="$2"; shift 2 ;;
            --plan-b-sigma) RGB_MOTION_SIGMA="$2"; shift 2 ;;
            --plan-d)
                case "$2" in
                    true|yes|on|1) FLOW_ALIGN=true ;;
                    false|no|off|0) FLOW_ALIGN=false ;;
                esac
                shift 2
                ;;
            --plan-d-height) FLOW_HEIGHT="$2"; shift 2 ;;
            --plan-e) MEDIAN_WINDOW="$2"; shift 2 ;;

            --fast-kernel) FAST_KERNEL="$2"; shift 2 ;;
            --fast-max-iter) FAST_MAX_ITER="$2"; shift 2 ;;

            --no-plans)
                QUANTILE_SMOOTH=0
                DEPTH_SMOOTH=0
                RGB_MOTION_SIGMA=0
                FLOW_ALIGN=false
                MEDIAN_WINDOW=1
                shift
                ;;
            --all-plans)
                QUANTILE_SMOOTH=0.8
                DEPTH_SMOOTH=0.6
                RGB_MOTION_SIGMA=0.1
                FLOW_ALIGN=true
                MEDIAN_WINDOW=3
                shift
                ;;

            --fp16) FP16=true; shift ;;
            --no-fp16) FP16=false; shift ;;
            --profile) PROFILE_TIME=true; shift ;;
            --no-profile) PROFILE_TIME=false; shift ;;
            --profile-output) PROFILE_OUTPUT="$2"; shift 2 ;;
            --video-list) VIDEO_LIST="$2"; shift 2 ;;
            --only-plan) ONLY_PLAN="$2"; shift 2 ;;

            --mode) MODE="$2"; shift 2 ;;

            -h|--help) show_help; exit 0 ;;
            *) echo "未知参数: $1"; show_help; exit 1 ;;
        esac
    done

    # 如果提供了视频列表文件，读取它
    if [[ -n "$VIDEO_LIST" ]]; then
        if [[ -f "$VIDEO_LIST" ]]; then
            VIDEO_PATH=$(cat "$VIDEO_LIST" | tr '\n' ' ')
        else
            echo "错误: 视频列表文件不存在: $VIDEO_LIST"
            exit 1
        fi
    fi

    if [[ -z "$VIDEO_PATH" ]]; then
        echo "错误: 必须指定输入视频 (-i/--input 或 --video-list)"
        show_help
        exit 1
    fi
}

# ========================================================
# 应用预定义Plan配置
# ========================================================
apply_plan_config() {
    local plan_name="$1"
    case "$plan_name" in
        no_plan)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            ;;
        plan_a)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            ;;
        plan_bc)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.1
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            ;;
        plan_d)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=true
            MEDIAN_WINDOW=1
            ;;
        plan_e)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=false
            MEDIAN_WINDOW=3
            ;;
        plan_ab)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.1
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            ;;
        plan_abc_e)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.1
            FLOW_ALIGN=false
            MEDIAN_WINDOW=3
            ;;
        plan_abcd)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.1
            FLOW_ALIGN=true
            MEDIAN_WINDOW=1
            ;;
        all_plans)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.1
            FLOW_ALIGN=true
            MEDIAN_WINDOW=3
            ;;
        *)
            echo "错误: 未知的Plan配置: $plan_name"
            echo "可用配置: no_plan/plan_a/plan_bc/plan_d/plan_e/plan_ab/plan_abc_e/plan_abcd/all_plans"
            exit 1
            ;;
    esac
}

# ========================================================
# 构建Python命令
# ========================================================
build_cmd() {
    local out_path="$1"
    local profile_path="$2"

    cmd="python mono2stereo.py"
    cmd+=" --video-path \"$VIDEO_PATH\""
    cmd+=" --outdir \"$OUTDIR\""
    [[ -n "$out_path" ]] && cmd+=" --output \"$out_path\""

    cmd+=" --encoder \"$ENCODER\""
    cmd+=" --input-size \"$INPUT_SIZE\""
    cmd+=" --dibr-size \"$DIBR_SIZE\""
    cmd+=" --max-disparity \"$MAX_DISPARITY\""
    cmd+=" --depth-mode \"$DEPTH_MODE\""
    cmd+=" --layout \"$LAYOUT\""

    # Plan A-E
    cmd+=" --quantile-smooth \"$QUANTILE_SMOOTH\""
    cmd+=" --depth-smooth \"$DEPTH_SMOOTH\""
    cmd+=" --rgb-motion-sigma \"$RGB_MOTION_SIGMA\""
    $FLOW_ALIGN && cmd+=" --flow-align --flow-height \"$FLOW_HEIGHT\""
    cmd+=" --median-window \"$MEDIAN_WINDOW\""
    cmd+=" --hole-dilate-left \"$HOLE_DILATE_LEFT\""

    # 空洞修补参数
    cmd+=" --fast-kernel \"$FAST_KERNEL\""
    cmd+=" --fast-max-iter \"$FAST_MAX_ITER\""

    # 其他
    $FP16 && cmd+=" --fp16"
    $PROFILE_TIME && cmd+=" --profile-time"
    [[ -n "$profile_path" ]] && cmd+=" --profile-output \"$profile_path\""

    echo "$cmd"
}

# ========================================================
# 单次运行 (单个视频)
# ========================================================
run_single_for_video() {
    local single_video="$1"
    local single_output="$2"
    local single_profile="$3"
    local video_name=$(basename "$single_video" | sed 's/\.[^.]*$//')
    local video_outdir="$OUTDIR/$video_name"

    # 如果没有指定输出，自动生成 - 每个视频单独子文件夹
    if [[ -z "$single_output" ]]; then
        if [[ -n "$ONLY_PLAN" ]]; then
            single_output="$video_outdir/${ONLY_PLAN}.mp4"
            single_profile="$video_outdir/${ONLY_PLAN}_profile.txt"
        else
            single_output="$video_outdir/${video_name}_3d.mp4"
            single_profile="$video_outdir/${video_name}_profile.txt"
        fi
    fi
    if [[ -z "$single_profile" ]]; then
        if [[ -n "$ONLY_PLAN" ]]; then
            single_profile="$video_outdir/${ONLY_PLAN}_profile.txt"
        else
            single_profile="$video_outdir/${video_name}_profile.txt"
        fi
    fi

    echo "=========================================="
    echo "处理视频: $single_video"
    echo "运行配置:"
    echo "  输出: $single_output"
    echo "  性能统计: $single_profile"
    echo "  模型: $ENCODER, 输入尺寸: $INPUT_SIZE"
    echo "  最大视差: $MAX_DISPARITY, 深度模式: $DEPTH_MODE"
    echo "  Plan A: quantile_smooth=$QUANTILE_SMOOTH"
    echo "  Plan B+C: depth_smooth=$DEPTH_SMOOTH, sigma=$RGB_MOTION_SIGMA"
    echo "  Plan D: $(if $FLOW_ALIGN; then echo "ON (h=$FLOW_HEIGHT)"; else echo "OFF"; fi)"
    echo "  Plan E: median_window=$MEDIAN_WINDOW"
    echo "  空洞修补: kernel=$FAST_KERNEL, max_iter=$FAST_MAX_ITER"
    echo "=========================================="

    mkdir -p "$video_outdir"
    mkdir -p "$(dirname "$single_profile")"

    # 临时设置 VIDEO_PATH 为当前视频
    local SAVED_VIDEO_PATH="$VIDEO_PATH"
    VIDEO_PATH="$single_video"

    cmd=$(build_cmd "$single_output" "$single_profile")
    echo "执行命令:"
    echo "$cmd"
    echo ""
    eval "$cmd"

    # 恢复
    VIDEO_PATH="$SAVED_VIDEO_PATH"
}

# ========================================================
# 单次运行 (入口)
# ========================================================
run_single() {
    local video_count=$(echo "$VIDEO_PATH" | wc -w)

    if [[ "$video_count" -eq 1 && -n "$OUTPUT" ]]; then
        # 单个视频 + 指定输出 - 直接用
        run_single_for_video "$VIDEO_PATH" "$OUTPUT" "$PROFILE_OUTPUT"
    else
        # 多个视频 - 循环处理，自动命名
        local idx=1
        for vid in $VIDEO_PATH; do
            if [[ -f "$vid" ]]; then
                if [[ "$video_count" -gt 1 ]]; then
                    # 多视频模式 - 不传输出路径，让 run_single_for_video 自己生成（带子文件夹）
                    run_single_for_video "$vid" "" ""
                else
                    # 单视频但没指定输出
                    run_single_for_video "$vid" "" ""
                fi
            else
                echo "警告: 视频不存在，跳过: $vid"
            fi
            idx=$((idx+1))
        done
    fi
}

# ========================================================
# 测试所有Plan组合 (单个视频)
# ========================================================
run_all_plan_combinations_for_video() {
    local single_video="$1"
    local video_name=$(basename "$single_video" | sed 's/\.[^.]*$//')
    local video_outdir="$OUTDIR/$video_name"

    echo ""
    echo "=========================================="
    echo "处理视频: $single_video"
    echo "输出目录: $video_outdir"
    echo "=========================================="

    mkdir -p "$video_outdir"

    # 定义各种Plan组合
    declare -a runs=(
        "no_plan|0|0|0|false|1|无Plan"
        "plan_a|0.8|0|0|false|1|仅Plan A"
        "plan_bc|0|0.6|0.1|false|1|仅Plan B+C"
        "plan_d|0|0|0|true|1|仅Plan D"
        "plan_e|0|0|0|false|3|仅Plan E"
        "plan_ab|0.8|0.6|0.1|false|1|Plan A+B+C"
        "plan_abc_e|0.8|0.6|0.1|false|3|Plan A+B+C+E"
        "plan_abcd|0.8|0.6|0.1|true|1|Plan A+B+C+D"
        "all_plans|0.8|0.6|0.1|true|3|全部Plan A+B+C+D+E"
    )

    for run in "${runs[@]}"; do
        IFS='|' read -r name qa ds sigma fw mw desc <<< "$run"

        echo ""
        echo "----------------------------------------"
        echo "[$video_name] 运行: $desc"
        echo "----------------------------------------"

        # 设置参数
        QUANTILE_SMOOTH="$qa"
        DEPTH_SMOOTH="$ds"
        RGB_MOTION_SIGMA="$sigma"
        FLOW_ALIGN="$fw"
        MEDIAN_WINDOW="$mw"

        # 构建输出路径 - 每个视频单独子文件夹
        out_path="$video_outdir/${name}.mp4"
        profile_path="$video_outdir/${name}_profile.txt"

        # 临时设置 VIDEO_PATH 为当前视频
        local SAVED_VIDEO_PATH="$VIDEO_PATH"
        VIDEO_PATH="$single_video"

        cmd=$(build_cmd "$out_path" "$profile_path")
        echo "执行: $cmd"
        eval "$cmd"

        # 恢复
        VIDEO_PATH="$SAVED_VIDEO_PATH"
    done
}

# ========================================================
# 测试所有Plan组合 (入口)
# ========================================================
run_all_plan_combinations() {
    echo "=========================================="
    echo "运行模式: 测试所有Plan组合"
    echo "输出目录: $OUTDIR"
    echo "=========================================="

    mkdir -p "$OUTDIR"

    # 处理多个视频
    for vid in $VIDEO_PATH; do
        if [[ -f "$vid" ]]; then
            run_all_plan_combinations_for_video "$vid"
        else
            echo "警告: 视频不存在，跳过: $vid"
        fi
    done

    echo ""
    echo "=========================================="
    echo "所有Plan组合测试完成！"
    echo "输出目录: $OUTDIR"
    echo "=========================================="
}

# ========================================================
# Plan消融实验 (逐步添加) - 单个视频
# ========================================================
run_ablations_for_video() {
    local single_video="$1"
    local video_name=$(basename "$single_video" | sed 's/\.[^.]*$//')
    local video_outdir="$OUTDIR/$video_name"

    echo ""
    echo "=========================================="
    echo "处理视频: $single_video"
    echo "输出目录: $video_outdir"
    echo "=========================================="

    mkdir -p "$video_outdir"

    # 消融实验: 从无到有逐步添加
    declare -a runs=(
        "01_no_plan|0|0|0|false|1|Baseline (无Plan)"
        "02_add_a|0.8|0|0|false|1|+Plan A"
        "03_add_bc|0.8|0.6|0.1|false|1|+Plan B+C"
        "04_add_e|0.8|0.6|0.1|false|3|+Plan E"
        "05_add_d|0.8|0.6|0.1|true|3|+Plan D (全部)"
    )

    for run in "${runs[@]}"; do
        IFS='|' read -r name qa ds sigma fw mw desc <<< "$run"

        echo ""
        echo "----------------------------------------"
        echo "[$video_name] 运行: $desc"
        echo "----------------------------------------"

        QUANTILE_SMOOTH="$qa"
        DEPTH_SMOOTH="$ds"
        RGB_MOTION_SIGMA="$sigma"
        FLOW_ALIGN="$fw"
        MEDIAN_WINDOW="$mw"

        # 构建输出路径 - 每个视频单独子文件夹
        out_path="$video_outdir/${name}.mp4"
        profile_path="$video_outdir/${name}_profile.txt"

        # 临时设置 VIDEO_PATH 为当前视频
        local SAVED_VIDEO_PATH="$VIDEO_PATH"
        VIDEO_PATH="$single_video"

        cmd=$(build_cmd "$out_path" "$profile_path")
        echo "执行: $cmd"
        eval "$cmd"

        # 恢复
        VIDEO_PATH="$SAVED_VIDEO_PATH"
    done
}

# ========================================================
# Plan消融实验 (逐步添加) - 入口
# ========================================================
run_ablations() {
    echo "=========================================="
    echo "运行模式: Plan消融实验 (逐步添加)"
    echo "输出目录: $OUTDIR"
    echo "=========================================="

    mkdir -p "$OUTDIR"

    # 处理多个视频
    for vid in $VIDEO_PATH; do
        if [[ -f "$vid" ]]; then
            run_ablations_for_video "$vid"
        else
            echo "警告: 视频不存在，跳过: $vid"
        fi
    done

    echo ""
    echo "=========================================="
    echo "消融实验完成！"
    echo "=========================================="
}

# ========================================================
# 主函数
# ========================================================
main() {
    parse_args "$@"

    # 如果指定了 --only-plan，应用配置并强制用 custom 模式
    if [[ -n "$ONLY_PLAN" ]]; then
        echo "应用Plan配置: $ONLY_PLAN"
        apply_plan_config "$ONLY_PLAN"
        MODE="custom"
    fi

    case "$MODE" in
        custom)
            run_single
            ;;
        all)
            run_all_plan_combinations
            ;;
        ablations)
            run_ablations
            ;;
        *)
            echo "未知模式: $MODE"
            exit 1
            ;;
    esac
}

main "$@"
