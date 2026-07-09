#!/bin/bash
set -e

# ========================================================
# 2D转3D 优化版运行脚本
# ✅ 新增: Plan F - 边缘感知深度平滑 (引导滤波/双边滤波)
# ✅ 新增: Plan G - TV-L1梯度驱动空洞修补
# ========================================================

# ---------------- 默认参数配置 ----------------
VIDEO_PATH="../../dataset/shuai/horse.mp4"
OUTPUT=""
OUTDIR="./output_optimized"
ENCODER="vits"          # vits, vitb, vitl, vitg
INPUT_SIZE=518          # 深度推理分辨率 (必须是14的倍数: 518=14×37)
DIBR_SIZE=0             # DIBR渲染分辨率 (0=同input-size, -1=原分辨率)
MAX_DISPARITY=24        # 最大视差
DEPTH_MODE="inverse"     # metric, inverse
FP16=true               # 启用FP16
LAYOUT="sbs"            # sbs, ou, overlay, anaglyph

# Plan A-E 时序稳定参数 (默认全开推荐配置)
QUANTILE_SMOOTH=0.8     # Plan A: 分位数EMA (0=禁用)
DEPTH_SMOOTH=0.6        # Plan B+C: near空间EMA (0=禁用)
RGB_MOTION_SIGMA=0.02   # Plan B: RGB运动敏感度 (0=禁用自适应)
FLOW_ALIGN=true         # Plan D: 光流对齐 (true/false)
FLOW_HEIGHT=144         # Plan D: 光流计算分辨率
MEDIAN_WINDOW=3         # Plan E: 滑窗中值 (1=禁用, 3/5/7推荐)
HOLE_DILATE_LEFT=2      # 空洞左侧膨胀 (保护前景)
HOLE_DILATE_RIGHT=1     # 空洞右侧膨胀 (消除轮廓线)

# ========================================================
# ✨ 新增优化参数 (F + G)
# ========================================================

# ---------- Plan F: 边缘感知深度平滑 ----------
# 消除前景边缘晕轮效应，深度边缘更锐利
EDGE_AWARE_SMOOTH=true  # 启用边缘感知平滑 (true/false)

# 方案1: 引导滤波 (更快，推荐)
GUIDED_FILTER_RADIUS=4  # 引导滤波半径 (>0启用，越大平滑范围越大，推荐3-5)
GUIDED_FILTER_EPS=1e-4  # 引导滤波正则化 (越小边缘保持越强，推荐1e-5 ~ 1e-3)

# 方案2: 双边滤波 (质量更好，稍慢) - 当GUIDED_FILTER_RADIUS=0时启用
SMOOTH_SIGMA_SPATIAL=5.0  # 空间标准差 (越大平滑范围越大)
SMOOTH_SIGMA_COLOR=0.1    # 颜色标准差 (越小颜色相似性要求越高)
SMOOTH_SIGMA_DEPTH=0.05   # 深度标准差 (越小深度相似性要求越高)

# ---------- Plan G: TV-L1梯度驱动空洞修补 ----------
# 沿等照度线扩散，边缘过渡更自然，无块效应
TV_INPAINT=true         # 启用TV梯度修补 (true/false)
TV_LAMBDA=0.1           # TV正则化强度 (0.05-0.2，越大越平滑)
TV_MAX_ITER=50          # TV迭代次数 (30-100，越多越平滑)
TV_TAU=0.125            # 梯度下降步长 (一般不用改)
EDGE_FIRST_INPAINT=true # EdgeConnect风格: 先修边缘，再填内部 (推荐开启)

# ---------- Plan H: 各向异性扩散（导师推荐的梯度方法） ----------
# 沿边缘切线方向扩散，垂直边缘不扩散，边缘保持最好
ANISOTROPIC_INPAINT=false  # 默认关闭（需显式开启）
ANISO_KAPPA=0.05           # 边缘敏感度，越小越保边 (推荐0.02-0.1)
ANISO_MAX_ITER=100         # 扩散迭代次数 (推荐50-200)

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
  --plan-b-sigma <数值> RGB运动敏感度 (0=禁用自适应, 0.02推荐)
  --plan-d <开关>       光流对齐 (true/false)
  --plan-d-height <数值>光流计算分辨率 (默认: $FLOW_HEIGHT)
  --plan-e <数值>       滑窗中值窗口 (1=禁用, 3推荐)

✨ 新增: Plan F - 边缘感知深度平滑 (消除晕轮):
  --[no-]edge-smooth    开关边缘感知平滑 (默认: 开)
  --guided-r <数值>     引导滤波半径 (>0启用, 0=用双边滤波, 默认: $GUIDED_FILTER_RADIUS)
  --guided-eps <数值>   引导滤波正则化 (默认: $GUIDED_FILTER_EPS)
  --bilateral-s <数值>  双边滤波空间sigma (默认: $SMOOTH_SIGMA_SPATIAL)
  --bilateral-c <数值>  双边滤波颜色sigma (默认: $SMOOTH_SIGMA_COLOR)
  --bilateral-d <数值>  双边滤波深度sigma (默认: $SMOOTH_SIGMA_DEPTH)

✨ 新增: Plan G - TV-L1梯度空洞修补 (自然过渡):
  --[no-]tv-inpaint     开关TV梯度修补 (默认: 开)
  --tv-lambda <数值>    TV正则化强度 (0.05-0.2, 默认: $TV_LAMBDA)
  --tv-iter <数值>      TV迭代次数 (30-100, 默认: $TV_MAX_ITER)
  --[no-]edge-first     边缘优先修补 (EdgeConnect风格, 默认: 开)

空洞修补:
  --hole-dilate-left <数值>  空洞左侧膨胀像素 (保护前景, 默认: $HOLE_DILATE_LEFT)
  --hole-dilate-right <数值> 空洞右侧膨胀像素 (消除轮廓线, 默认: $HOLE_DILATE_RIGHT)

快捷模式:
  --no-plans            禁用所有Plan (纯单帧模式)
  --all-plans           启用所有Plan (推荐配置，含F+G)
  --only-plan <名称>   只运行指定配置: no_plan/plan_a/plan_bc/plan_d/plan_e/
                        plan_f/plan_g/plan_fg/plan_abcde/plan_abcdefg/all_plans
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
  # 基本用法 (最佳效果，全部优化开启)
  $(basename "$0") -i input.mp4 -o output.mp4

  # 仅用边缘平滑 + TV修补，禁用时序稳定
  $(basename "$0") -i input.mp4 -o output.mp4 --no-plans --edge-smooth --tv-inpaint

  # TV修补效果微调 (更平滑)
  $(basename "$0") -i input.mp4 -o output.mp4 --tv-lambda 0.2 --tv-iter 100

  # 边缘更锐利 (减小正则化)
  $(basename "$0") -i input.mp4 -o output.mp4 --guided-eps 5e-5

  # 测试所有优化组合 (AB测试)
  $(basename "$0") -i input.mp4 --mode all --outdir ./results_optimized

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

            # ---------- 新增: Plan F 参数 ----------
            --edge-aware-smooth) EDGE_AWARE_SMOOTH=true; shift ;;
            --no-edge-aware-smooth) EDGE_AWARE_SMOOTH=false; shift ;;
            --guided-r) GUIDED_FILTER_RADIUS="$2"; shift 2 ;;
            --guided-eps) GUIDED_FILTER_EPS="$2"; shift 2 ;;
            --bilateral-s) SMOOTH_SIGMA_SPATIAL="$2"; shift 2 ;;
            --bilateral-c) SMOOTH_SIGMA_COLOR="$2"; shift 2 ;;
            --bilateral-d) SMOOTH_SIGMA_DEPTH="$2"; shift 2 ;;

            # ---------- 新增: Plan H - 各向异性扩散（导师推荐的梯度方法） ----------
            --anisotropic-inpaint) ANISOTROPIC_INPAINT=true; shift ;;
            --no-anisotropic-inpaint) ANISOTROPIC_INPAINT=false; shift ;;
            --aniso-kappa) ANISO_KAPPA="$2"; shift 2 ;;
            --aniso-iter) ANISO_MAX_ITER="$2"; shift 2 ;;

            # ---------- 新增: Plan G 参数 ----------
            --tv-inpaint) TV_INPAINT=true; shift ;;
            --no-tv-inpaint) TV_INPAINT=false; shift ;;
            --tv-lambda) TV_LAMBDA="$2"; shift 2 ;;
            --tv-iter) TV_MAX_ITER="$2"; shift 2 ;;
            --edge-first) EDGE_FIRST_INPAINT=true; shift ;;
            --no-edge-first) EDGE_FIRST_INPAINT=false; shift ;;

            --hole-dilate-left) HOLE_DILATE_LEFT="$2"; shift 2 ;;
            --hole-dilate-right) HOLE_DILATE_RIGHT="$2"; shift 2 ;;

            --no-plans)
                QUANTILE_SMOOTH=0
                DEPTH_SMOOTH=0
                RGB_MOTION_SIGMA=0
                FLOW_ALIGN=false
                MEDIAN_WINDOW=1
                EDGE_AWARE_SMOOTH=false
                TV_INPAINT=false
                shift
                ;;
            --all-plans)
                QUANTILE_SMOOTH=0.8
                DEPTH_SMOOTH=0.6
                RGB_MOTION_SIGMA=0.02
                FLOW_ALIGN=true
                MEDIAN_WINDOW=3
                EDGE_AWARE_SMOOTH=true
                TV_INPAINT=true
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
# 应用预定义Plan配置 (扩展版，含F+G)
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
            EDGE_AWARE_SMOOTH=false
            TV_INPAINT=false
            ;;
        plan_a)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            EDGE_AWARE_SMOOTH=false
            TV_INPAINT=false
            ;;
        plan_bc)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.02
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            EDGE_AWARE_SMOOTH=false
            TV_INPAINT=false
            ;;
        plan_d)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=true
            MEDIAN_WINDOW=1
            EDGE_AWARE_SMOOTH=false
            TV_INPAINT=false
            ;;
        plan_e)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=false
            MEDIAN_WINDOW=3
            EDGE_AWARE_SMOOTH=false
            TV_INPAINT=false
            ;;
        plan_f)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            EDGE_AWARE_SMOOTH=true
            TV_INPAINT=false
            ;;
        plan_g)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            EDGE_AWARE_SMOOTH=false
            TV_INPAINT=true
            ;;
        plan_fg)
            QUANTILE_SMOOTH=0
            DEPTH_SMOOTH=0
            RGB_MOTION_SIGMA=0
            FLOW_ALIGN=false
            MEDIAN_WINDOW=1
            EDGE_AWARE_SMOOTH=true
            TV_INPAINT=true
            ;;
        plan_abcde)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.02
            FLOW_ALIGN=true
            MEDIAN_WINDOW=3
            EDGE_AWARE_SMOOTH=false
            TV_INPAINT=false
            ;;
        plan_abcdef)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.02
            FLOW_ALIGN=true
            MEDIAN_WINDOW=3
            EDGE_AWARE_SMOOTH=true
            TV_INPAINT=false
            ;;
        plan_abcdeg)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.02
            FLOW_ALIGN=true
            MEDIAN_WINDOW=3
            EDGE_AWARE_SMOOTH=false
            TV_INPAINT=true
            ;;
        plan_abcdefg|all_plans)
            QUANTILE_SMOOTH=0.8
            DEPTH_SMOOTH=0.6
            RGB_MOTION_SIGMA=0.02
            FLOW_ALIGN=true
            MEDIAN_WINDOW=3
            EDGE_AWARE_SMOOTH=true
            TV_INPAINT=true
            ;;
        *)
            echo "错误: 未知的Plan配置: $plan_name"
            echo "可用配置: no_plan/plan_a/plan_bc/plan_d/plan_e/plan_f/plan_g/plan_fg/plan_abcde/plan_abcdef/plan_abcdeg/plan_abcdefg/all_plans"
            exit 1
            ;;
    esac
}

# ========================================================
# 构建Python命令 (含新增参数)
# ========================================================
build_cmd() {
    local out_path="$1"
    local profile_path="$2"

    cmd="python mono2stereo_optimized.py"
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
    cmd+=" --hole-dilate-right \"$HOLE_DILATE_RIGHT\""

    # ---------- 新增: Plan F - 边缘感知平滑 ----------
    if $EDGE_AWARE_SMOOTH; then
        cmd+=" --edge-aware-smooth"
    else
        cmd+=" --no-edge-aware-smooth"
    fi
    cmd+=" --guided-filter-radius \"$GUIDED_FILTER_RADIUS\""
    cmd+=" --guided-filter-eps \"$GUIDED_FILTER_EPS\""
    if [[ "$GUIDED_FILTER_RADIUS" -eq 0 ]]; then
        cmd+=" --smooth-sigma-spatial \"$SMOOTH_SIGMA_SPATIAL\""
        cmd+=" --smooth-sigma-color \"$SMOOTH_SIGMA_COLOR\""
        cmd+=" --smooth-sigma-depth \"$SMOOTH_SIGMA_DEPTH\""
    fi

    # ---------- 新增: Plan G - TV梯度修补 ----------
    if $TV_INPAINT; then
        cmd+=" --tv-inpaint"
    else
        cmd+=" --no-tv-inpaint"
    fi
    cmd+=" --tv-lambda \"$TV_LAMBDA\""
    cmd+=" --tv-max-iter \"$TV_MAX_ITER\""
    cmd+=" --tv-tau \"$TV_TAU\""
    if $EDGE_FIRST_INPAINT; then
        cmd+=" --edge-first-inpaint"
    else
        cmd+=" --no-edge-first-inpaint"
    fi

    # ---------- 新增: Plan H - 各向异性扩散（梯度方法） ----------
    if $ANISOTROPIC_INPAINT; then
        cmd+=" --anisotropic-inpaint"
    else
        cmd+=" --no-anisotropic-inpaint"
    fi
    cmd+=" --aniso-kappa \"$ANISO_KAPPA\""
    cmd+=" --aniso-max-iter \"$ANISO_MAX_ITER\""

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
            single_output="$video_outdir/${video_name}_optimized.mp4"
            single_profile="$video_outdir/${video_name}_optimized_profile.txt"
        fi
    fi
    if [[ -z "$single_profile" ]]; then
        if [[ -n "$ONLY_PLAN" ]]; then
            single_profile="$video_outdir/${ONLY_PLAN}_profile.txt"
        else
            single_profile="$video_outdir/${video_name}_optimized_profile.txt"
        fi
    fi

    echo "=========================================="
    echo "处理视频: $single_video"
    echo "运行配置 (优化版 F+G):"
    echo "  输出: $single_output"
    echo "  性能统计: $single_profile"
    echo "  模型: $ENCODER, 输入尺寸: $INPUT_SIZE"
    echo "  最大视差: $MAX_DISPARITY, 深度模式: $DEPTH_MODE"
    echo "  Plan A: quantile_smooth=$QUANTILE_SMOOTH"
    echo "  Plan B+C: depth_smooth=$DEPTH_SMOOTH, sigma=$RGB_MOTION_SIGMA"
    echo "  Plan D: $(if $FLOW_ALIGN; then echo "ON (h=$FLOW_HEIGHT)"; else echo "OFF"; fi)"
    echo "  Plan E: median_window=$MEDIAN_WINDOW"
    echo "✨ Plan F: 边缘感知平滑=$(if $EDGE_AWARE_SMOOTH; then echo "ON"; else echo "OFF"; fi)"
    if $EDGE_AWARE_SMOOTH; then
        if [[ "$GUIDED_FILTER_RADIUS" -gt 0 ]]; then
            echo "            引导滤波 r=$GUIDED_FILTER_RADIUS, eps=$GUIDED_FILTER_EPS"
        else
            echo "            双边滤波 s=$SMOOTH_SIGMA_SPATIAL, c=$SMOOTH_SIGMA_COLOR, d=$SMOOTH_SIGMA_DEPTH"
        fi
    fi
    echo "✨ Plan G: TV梯度修补=$(if $TV_INPAINT; then echo "ON"; else echo "OFF"; fi)"
    if $TV_INPAINT; then
        echo "            lambda=$TV_LAMBDA, iter=$TV_MAX_ITER, edge_first=$EDGE_FIRST_INPAINT"
    fi
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
# 测试所有Plan组合 (单个视频) - 扩展版含F+G
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

    # 定义各种Plan组合 (扩展版，含F+G消融)
    declare -a runs=(
        "01_no_plan|0|0|0|false|1|false|false|Baseline (无Plan)"
        "02_plan_a|0.8|0|0|false|1|false|false|+Plan A"
        "03_plan_bc|0.8|0.6|0.02|false|1|false|false|+Plan B+C"
        "04_plan_abcde|0.8|0.6|0.02|true|3|false|false|+Plan D+E (ABCDE)"
        "05_plan_f|0|0|0|false|1|true|false|仅Plan F (边缘平滑)"
        "06_plan_g|0|0|0|false|1|false|true|仅Plan G (TV修补)"
        "07_plan_fg|0|0|0|false|1|true|true|Plan F+G (核心优化)"
        "08_plan_abcdef|0.8|0.6|0.02|true|3|true|false|ABCDE + F"
        "09_plan_abcdeg|0.8|0.6|0.02|true|3|false|true|ABCDE + G"
        "10_all_plans|0.8|0.6|0.02|true|3|true|true|全部优化 ABCDEFG (推荐)"
    )

    for run in "${runs[@]}"; do
        IFS='|' read -r name qa ds sigma fw mw f g desc <<< "$run"

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
        EDGE_AWARE_SMOOTH="$f"
        TV_INPAINT="$g"

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
    echo "运行模式: 测试所有Plan组合 (含F+G优化)"
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
    echo "  重点对比:"
    echo "    - 04_plan_abcde.mp4 (原版)"
    echo "    - 10_all_plans.mp4 (优化版F+G)"
    echo "    - 07_plan_fg.mp4 (仅核心优化，无时序)"
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
    echo "  消融实验: 从无到有逐步添加优化"
    echo "=========================================="

    mkdir -p "$video_outdir"

    # 消融实验: 从无到有逐步添加
    declare -a runs=(
        "01_baseline|0|0|0|false|1|false|false|1. Baseline (无Plan)"
        "02_add_a|0.8|0|0|false|1|false|false|2. +Plan A (分位数)"
        "03_add_bc|0.8|0.6|0.02|false|1|false|false|3. +Plan B+C (EMA)"
        "04_add_de|0.8|0.6|0.02|true|3|false|false|4. +Plan D+E (光流+中值)"
        "05_add_f|0.8|0.6|0.02|true|3|true|false|5. +Plan F (边缘平滑) ✨"
        "06_add_g|0.8|0.6|0.02|true|3|true|true|6. +Plan G (TV修补) ✨"
    )

    for run in "${runs[@]}"; do
        IFS='|' read -r name qa ds sigma fw mw f g desc <<< "$run"

        echo ""
        echo "----------------------------------------"
        echo "[$video_name] 运行: $desc"
        echo "----------------------------------------"

        QUANTILE_SMOOTH="$qa"
        DEPTH_SMOOTH="$ds"
        RGB_MOTION_SIGMA="$sigma"
        FLOW_ALIGN="$fw"
        MEDIAN_WINDOW="$mw"
        EDGE_AWARE_SMOOTH="$f"
        TV_INPAINT="$g"

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
    echo "运行模式: Plan消融实验 (逐步添加，含F+G)"
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
    echo "输出目录: $OUTDIR"
    echo "  按顺序查看效果提升:"
    echo "    01 → 04: 原版时序优化"
    echo "    04 → 05: 边缘平滑带来的提升"
    echo "    05 → 06: TV修补带来的提升"
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
