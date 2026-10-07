"""Reusable EEG research illustrations consolidated from four scripts.

Plot functions return their results without calling plt.show(). Call plt.show()
explicitly after composing figures. Connectivity examples require this project's
utils and feature_fusion modules; the other plots use NumPy, pandas and Matplotlib.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc
from matplotlib.colors import ListedColormap

__all__ = [
    "plot_electrode_selection", "plot_seed_electrode_selection",
    "plot_connectivity_fusion", "compute_phase_gate", "plot_phase_gate",
    "plot_classification_performance", "plot_example_classification_performance",
]

def plot_electrode_selection(
    df_all: pd.DataFrame,
    df_subset: pd.DataFrame | None = None,
    *,
    channel_col: str = "channel",
    x_col: str = "x",
    y_col: str = "y",
    order_col: str = "order",
    figsize: tuple[float, float] = (9, 9),
    electrode_size: float = 0.080,
    head_radius: float = 1.0,
    padding: float = 0.12,
    selected_color: str = "#398B7E",
    unselected_color: str = "white",
    edge_color: str = "black",
    selected_text_color: str = "white",
    unselected_text_color: str = "black",
    head_linewidth: float = 2.0,
    electrode_linewidth: float = 1.4,
    font_size: float = 10,
    show_guides: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    绘制 EEG 全电极及子电极高亮示意图。

    Parameters
    ----------
    df_all : pd.DataFrame
        全部电极信息，至少包含 channel、x、y 三列。

    df_subset : pd.DataFrame | None
        需要高亮显示的子电极集合。
        可以只包含 channel 列，也可以包含完整电极信息。
        如果为 None，则所有电极均绘制为未选中状态。

    channel_col : str
        通道名称列。

    x_col : str
        左右方向坐标列。

    y_col : str
        前后方向坐标列。数值越大，绘图位置越靠上。

    order_col : str
        可选的排序列。

    figsize : tuple
        图像尺寸。

    electrode_size : float
        电极圆半径，使用归一化后的绘图坐标。

    head_radius : float
        头部圆形轮廓半径。

    padding : float
        最外层电极与头部轮廓之间的留白。

    selected_color : str
        子集电极的填充颜色。

    unselected_color : str
        未选中电极的填充颜色。

    edge_color : str
        电极和头部轮廓颜色。

    selected_text_color : str
        子集电极名称颜色。

    unselected_text_color : str
        未选中电极名称颜色。

    head_linewidth : float
        头部轮廓线宽。

    electrode_linewidth : float
        电极边框线宽。

    font_size : float
        电极名称字号。

    show_guides : bool
        是否显示内部参考圆以及水平、垂直辅助线。

    title : str | None
        图标题。

    ax : plt.Axes | None
        可选的 matplotlib 坐标轴。

    Returns
    -------
    fig : matplotlib.figure.Figure
        图像对象。

    ax : matplotlib.axes.Axes
        坐标轴对象。

    plot_df : pd.DataFrame
        包含 plot_x、plot_y 和 is_selected 的绘图数据。
    """

    # =========================================================
    # 1. 检查输入列
    # =========================================================
    required_columns = {channel_col, x_col, y_col}
    missing_columns = required_columns.difference(df_all.columns)

    if missing_columns:
        raise ValueError(
            f"df_all 缺少必要列：{sorted(missing_columns)}"
        )

    data = df_all.copy()

    # =========================================================
    # 2. 清理全部电极数据
    # =========================================================
    data = data.dropna(
        subset=[channel_col, x_col, y_col]
    ).copy()

    data[channel_col] = (
        data[channel_col]
        .astype(str)
        .str.strip()
    )

    data[x_col] = pd.to_numeric(
        data[x_col],
        errors="coerce",
    )

    data[y_col] = pd.to_numeric(
        data[y_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[x_col, y_col]
    ).copy()

    if data.empty:
        raise ValueError("df_all 中没有有效电极数据")

    # 检查重复通道
    duplicated_mask = data[channel_col].duplicated(
        keep=False
    )

    if duplicated_mask.any():
        duplicated_channels = sorted(
            data.loc[
                duplicated_mask,
                channel_col,
            ].unique().tolist()
        )

        raise ValueError(
            "df_all 中存在重复通道："
            f"{duplicated_channels}"
        )

    # 根据 order 排序
    if order_col in data.columns:
        data = data.sort_values(
            order_col
        ).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    # =========================================================
    # 3. 获取需要高亮的通道
    # =========================================================
    if df_subset is None:
        selected_channels: set[str] = set()

    else:
        if channel_col not in df_subset.columns:
            raise ValueError(
                f"df_subset 必须包含列：{channel_col!r}"
            )

        selected_channels = set(
            df_subset[channel_col]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        all_channels = set(
            data[channel_col].tolist()
        )

        unknown_channels = (
            selected_channels - all_channels
        )

        if unknown_channels:
            raise ValueError(
                "df_subset 中存在 df_all 未包含的通道："
                f"{sorted(unknown_channels)}"
            )

    data["is_selected"] = data[channel_col].isin(
        selected_channels
    )

    # =========================================================
    # 4. 计算绘图坐标
    # =========================================================
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    # 使用坐标范围中点作为中心
    x_center = (
        np.nanmax(x) + np.nanmin(x)
    ) / 2.0

    y_center = (
        np.nanmax(y) + np.nanmin(y)
    ) / 2.0

    x_centered = x - x_center
    y_centered = y - y_center

    # 使用统一比例缩放，避免横纵方向失真
    max_extent = max(
        np.nanmax(np.abs(x_centered)),
        np.nanmax(np.abs(y_centered)),
    )

    if not np.isfinite(max_extent) or max_extent == 0:
        raise ValueError(
            "电极坐标范围为 0，无法计算绘图位置"
        )

    available_radius = head_radius - padding

    if available_radius <= 0:
        raise ValueError(
            "padding 必须小于 head_radius"
        )

    data["plot_x"] = (
        x_centered / max_extent * available_radius
    )

    data["plot_y"] = (
        y_centered / max_extent * available_radius
    )

    # 确保所有电极都位于头部圆形轮廓内部
    radial_distance = np.sqrt(
        data["plot_x"].to_numpy() ** 2
        + data["plot_y"].to_numpy() ** 2
    )

    max_radial_distance = np.nanmax(
        radial_distance
    )

    if max_radial_distance > available_radius:
        shrink_ratio = (
            available_radius / max_radial_distance
        )

        data["plot_x"] *= shrink_ratio
        data["plot_y"] *= shrink_ratio

    # =========================================================
    # 5. 创建画布
    # =========================================================
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize
        )
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")

    # =========================================================
    # 6. 绘制头部轮廓
    # =========================================================
    head = Circle(
        xy=(0, 0),
        radius=head_radius,
        facecolor="white",
        edgecolor=edge_color,
        linewidth=head_linewidth,
        zorder=1,
    )

    ax.add_patch(head)

    # =========================================================
    # 7. 绘制辅助线
    # =========================================================
    if show_guides:
        guide_radius = head_radius * 0.82

        inner_circle = Circle(
            xy=(0, 0),
            radius=guide_radius,
            facecolor="none",
            edgecolor="0.55",
            linewidth=0.7,
            linestyle=":",
            zorder=2,
        )

        ax.add_patch(inner_circle)

        ax.plot(
            [-head_radius, head_radius],
            [0, 0],
            color="0.55",
            linewidth=0.7,
            linestyle=":",
            zorder=2,
        )

        ax.plot(
            [0, 0],
            [-head_radius, head_radius],
            color="0.55",
            linewidth=0.7,
            linestyle=":",
            zorder=2,
        )

    # =========================================================
    # 8. 绘制鼻子
    # =========================================================
    nose_width = head_radius * 0.20
    nose_height = head_radius * 0.10

    nose = Polygon(
        [
            (
                -nose_width / 2,
                head_radius + 0.02,
            ),
            (
                0,
                head_radius + nose_height,
            ),
            (
                nose_width / 2,
                head_radius + 0.02,
            ),
        ],
        closed=False,
        fill=False,
        edgecolor=edge_color,
        linewidth=head_linewidth,
        joinstyle="miter",
        zorder=3,
    )

    ax.add_patch(nose)

    # =========================================================
    # 9. 绘制耳朵
    # =========================================================
    ear_width = head_radius * 0.15
    ear_height = head_radius * 0.25

    left_ear = Arc(
        xy=(-head_radius, 0),
        width=ear_width,
        height=ear_height,
        theta1=90,
        theta2=270,
        linewidth=head_linewidth,
        color=edge_color,
        zorder=3,
    )

    right_ear = Arc(
        xy=(head_radius, 0),
        width=ear_width,
        height=ear_height,
        theta1=-90,
        theta2=90,
        linewidth=head_linewidth,
        color=edge_color,
        zorder=3,
    )

    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # 耳朵与头部之间的连接线
    ax.plot(
        [-head_radius, -head_radius],
        [-ear_height / 2, ear_height / 2],
        color=edge_color,
        linewidth=head_linewidth,
        zorder=3,
    )

    ax.plot(
        [head_radius, head_radius],
        [-ear_height / 2, ear_height / 2],
        color=edge_color,
        linewidth=head_linewidth,
        zorder=3,
    )

    # =========================================================
    # 10. 绘制电极
    #
    # 临时列不再使用下划线开头，因此 itertuples 可以安全访问。
    # =========================================================
    for channel, px, py, selected in data[
        [channel_col, "plot_x", "plot_y", "is_selected"]
    ].itertuples(index=False, name=None):
        channel = str(channel)

        if selected:
            facecolor = selected_color
            text_color = selected_text_color
        else:
            facecolor = unselected_color
            text_color = unselected_text_color

        electrode = Circle(
            xy=(px, py),
            radius=electrode_size,
            facecolor=facecolor,
            edgecolor=edge_color,
            linewidth=electrode_linewidth,
            zorder=5,
        )

        ax.add_patch(electrode)

        ax.text(
            px,
            py,
            channel,
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color=text_color,
            zorder=6,
        )

    # =========================================================
    # 11. 设置显示范围
    # =========================================================
    horizontal_limit = (
        head_radius + ear_width + 0.08
    )

    upper_limit = (
        head_radius + nose_height + 0.08
    )

    lower_limit = (
        -head_radius - 0.08
    )

    ax.set_xlim(
        -horizontal_limit,
        horizontal_limit,
    )

    ax.set_ylim(
        lower_limit,
        upper_limit,
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=14,
            pad=15,
        )

    fig.tight_layout()

    return fig, ax, data

def plot_seed_electrode_selection(channel_count=16, *, dataset="seed", **plot_options):
    """Load a distribution and plot an original 4/8/16/32/62-channel subset.

    Subsets use the original one-based row positions, converted to iloc positions.
    Returns (figure, axes, plotting_dataframe).
    """
    from utils import utils_feature_loading

    selections = {
        62: list(range(1, 63)),
        32: [1,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,53,55,59,60,61],
        16: [1,3,8,10,12,24,26,28,30,32,44,46,48,59,60,61],
        8: [1,3,26,30,44,48,59,61],
        4: [1,3,44,48],
    }
    if channel_count not in selections:
        raise ValueError("channel_count must be one of 4, 8, 16, 32, 62")
    data = utils_feature_loading.read_distribution(dataset)
    subset = data.iloc[[index - 1 for index in selections[channel_count]]]
    plot_options.setdefault("title", "Selected EEG electrodes")
    return plot_electrode_selection(data, subset, **plot_options)


def plot_connectivity_fusion(
    *, dataset="seed", recording="avg_sub1ex1_sub5ex3", band="alpha",
    phase_feature="plv", additional_phase_feature=None, gating_params=None,
    show_colorbar=False,
):
    """Load and illustrate PCC, phase connectivity and five fusion variants.

    Returns a mapping of names to plotted matrices. Set additional_phase_feature
    to "pli" to request an actual PLI example; the original script loaded PLV
    twice under different names. Loaded arrays are copied before masking.
    """
    from utils import utils_feature_loading, utils_interaction
    import feature_fusion

    def load(feature):
        matrix = np.array(utils_feature_loading.read_features(
            dataset, recording, feature)[band], dtype=float, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Connectivity features must be square 2D matrices")
        np.fill_diagonal(matrix, np.nan)
        return matrix

    pcc, phase = load("pcc"), load(phase_feature)
    if pcc.shape != phase.shape:
        raise ValueError("PCC and phase connectivity shapes must match")
    params = dict(fusion_type="sigmoid_gating", k=10.0, percentile=25,
                  power=1, normalization_basis=False,
                  normalization_modifier=False, scale=(0, 1))
    if gating_params is not None:
        params.update(gating_params)
    matrices = {"PCC": pcc, phase_feature.upper(): phase}
    if additional_phase_feature is not None:
        matrices[additional_phase_feature.upper()] = load(additional_phase_feature)
    finite_mask = np.isfinite(pcc)
    if not finite_mask.any():
        raise ValueError("PCC must contain finite off-diagonal entries")
    finite_values = pcc[finite_mask]
    normalized_pcc = np.full_like(pcc, np.nan)
    normalized_pcc[finite_mask] = (
        (finite_values - finite_values.min())
        / max(float(finite_values.max() - finite_values.min()), 1e-8)
    )
    matrices.update({
        "Additive": normalized_pcc + phase,
        "Multiplicative": pcc * phase,
        "Diagonal splicing": feature_fusion.feature_fusion_diagonal_blocking(pcc.copy(), phase.copy()),
        "Triangle splicing": feature_fusion.feature_fusion_triangle_blocking(pcc.copy(), phase.copy()),
        "Sigmoid gating": feature_fusion.feature_fusion_sigmoid_gating(pcc.copy(), phase.copy(), params),
    })
    positive_cmap = ListedColormap(
        plt.get_cmap("RdBu_r")(np.linspace(0.5, 1.0, 128)),
        name="RdBu_r_positive_half")
    positive_names = {phase_feature.upper(), "Additive"}
    if additional_phase_feature is not None:
        positive_names.add(additional_phase_feature.upper())
    for name, matrix in matrices.items():
        np.fill_diagonal(matrix, np.nan)
        utils_interaction.draw_projection(
            matrix, name, "", "", show_colorbar=show_colorbar,
            cmap=positive_cmap if name in positive_names else "RdBu_r")
    return matrices


def compute_phase_gate(modifier, *, k=200.0, tau=0.3, mode="heaviside"):
    """Compute a signed sigmoid or strict-threshold Heaviside gate.

    Heaviside is the original script's effective output. At exactly +/-tau,
    strict comparisons preserve the original boundary behavior.
    """
    if not np.isfinite(k) or k <= 0 or not np.isfinite(tau) or tau < 0:
        raise ValueError("k must be positive and tau nonnegative, both finite")
    values = np.asarray(modifier, dtype=float)
    if mode == "heaviside":
        return (values > tau).astype(float) + (values > -tau).astype(float) - 1.0
    if mode == "sigmoid":
        # Stable logistic evaluations avoid exponential overflow for large k.
        return (np.exp(-np.logaddexp(0, -k * (values - tau)))
                - np.exp(-np.logaddexp(0, k * (values + tau))))
    raise ValueError("mode must be 'sigmoid' or 'heaviside'")


def plot_phase_gate(*, k=200.0, tau=0.3, mode="heaviside", modifier=None, ax=None):
    """Plot a phase gate; return (figure, axes, modifier, gate)."""
    values = np.linspace(-1, 1, 1000) if modifier is None else np.asarray(modifier, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("modifier must be a nonempty 1D sequence")
    gate = compute_phase_gate(values, k=k, tau=tau, mode=mode)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(values, gate, linewidth=2, label="alpha")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(tau, color="r", linestyle=":", label=r"$\tau$")
    ax.axvline(-tau, color="r", linestyle=":")
    ax.set(xlabel="fn_modifier", ylabel="alpha",
           title=f"{mode.capitalize()} phase gate (k={k}, tau={tau})")
    ax.grid(True)
    ax.legend()
    ax.figure.tight_layout()
    return ax.figure, ax, values, gate


def plot_classification_performance(
    nrrs: Sequence[float], mean_acc: Mapping[str, Sequence[float]],
    std_acc: Mapping[str, Sequence[float]], *, methods=None, ax=None,
    title="Classification Performance Across Methods and NRRs",
):
    """Plot supplied means with +/- one SD across recordings; return fig, ax.

    This function visualizes summaries; it does not perform cross-validation.
    """
    nrrs = np.asarray(nrrs, dtype=float)
    if nrrs.ndim != 1 or nrrs.size == 0 or not np.isfinite(nrrs).all():
        raise ValueError("nrrs must be a nonempty finite 1D sequence")
    methods = list(mean_acc) if methods is None else list(methods)
    if not methods:
        raise ValueError("At least one method is required")
    series = []
    for method in methods:
        y = np.asarray(mean_acc[method], dtype=float)
        s = np.asarray(std_acc[method], dtype=float)
        if y.shape != nrrs.shape or s.shape != nrrs.shape:
            raise ValueError(f"{method}: means and SDs must match nrrs")
        if not np.isfinite(y).all() or not np.isfinite(s).all() or (s < 0).any():
            raise ValueError(f"{method}: values must be finite and SDs nonnegative")
        series.append((method, y, s))
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    for method, y, s in series:
        line, = ax.plot(nrrs, y, marker="o", linewidth=2, label=method)
        ax.fill_between(nrrs, y - s, y + s, alpha=0.15, color=line.get_color())
    ax.set(xlabel="Node Retention Rate (NRR, %)", ylabel="Accuracy (%)", title=title)
    ax.set_xticks(nrrs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    ax.figure.tight_layout()
    return ax.figure, ax


def plot_example_classification_performance(*, ax=None):
    """Plot the original placeholder results, explicitly labeled as example data."""
    nrrs = np.array([100, 50, 40, 30, 20, 10])
    methods = ["PCC", "PLV", "Additive", "Multiplicative", "Splicing", "PG-AC"]
    mean_acc = {
        "PCC":            np.array([82.1, 80.4, 79.6, 78.8, 76.9, 73.5]),
        "PLV":            np.array([83.4, 82.0, 81.2, 80.3, 78.7, 75.4]),
        "Additive":       np.array([83.0, 81.5, 80.8, 79.9, 77.8, 74.6]),
        "Multiplicative": np.array([83.8, 82.4, 81.7, 80.9, 79.3, 76.1]),
        "Splicing":       np.array([82.7, 81.2, 80.3, 79.1, 77.0, 73.8]),
        "PG-AC":          np.array([85.2, 84.6, 84.0, 83.1, 81.7, 79.8]),
    }
    std_acc = {
        "PCC":            np.array([2.1, 2.4, 2.3, 2.6, 2.8, 3.1]),
        "PLV":            np.array([2.0, 2.1, 2.2, 2.4, 2.6, 2.9]),
        "Additive":       np.array([2.2, 2.3, 2.4, 2.5, 2.7, 3.0]),
        "Multiplicative": np.array([2.0, 2.2, 2.1, 2.3, 2.5, 2.7]),
        "Splicing":       np.array([2.3, 2.5, 2.6, 2.7, 2.9, 3.2]),
        "PG-AC":          np.array([1.8, 1.9, 2.0, 2.1, 2.3, 2.5]),
    }
    return plot_classification_performance(
        nrrs, mean_acc, std_acc, methods=methods, ax=ax,
        title="Example Classification Performance (Placeholder Data)")

if __name__ == "__main__":
    # 1. Electrode selection using supplied data
    df_all = pd.DataFrame({
        "channel": ["Fp1", "Fp2", "C3", "C4", "O1", "O2"],
        "x": [-0.4, 0.4, -0.7, 0.7, -0.4, 0.4],
        "y": [0.8, 0.8, 0.0, 0.0, -0.8, -0.8],
    })
    df_subset = df_all[df_all["channel"].isin(["C3", "C4"])]

    fig, ax, plot_df = plot_electrode_selection(
        df_all, df_subset, title="Example electrode selection"
    )

    # 2. Electrode selection using the project's SEED distribution
    fig, ax, plot_df = plot_seed_electrode_selection(
        channel_count=16, dataset="seed"
    )

    # 3. Connectivity and fusion variants:
    # PCC, PLV, additive, multiplicative, diagonal splicing,
    # triangle splicing, and sigmoid gating
    matrices = plot_connectivity_fusion(
        dataset="seed",
        recording="avg_sub1ex1_sub5ex3",
        band="alpha",
        phase_feature="plv",
        show_colorbar=False,
    )

    # 4. Compute both phase gates directly
    modifier = np.linspace(-1, 1, 1000)
    sigmoid_gate = compute_phase_gate(
        modifier, k=200, tau=0.3, mode="sigmoid"
    )
    heaviside_gate = compute_phase_gate(
        modifier, tau=0.3, mode="heaviside"
    )

    # 5. Plot both phase gates side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_phase_gate(mode="sigmoid", k=200, tau=0.3, ax=axes[0])
    plot_phase_gate(mode="heaviside", tau=0.3, ax=axes[1])

    # 6. Classification performance using supplied summaries
    # These values are illustrative placeholders.
    plot_classification_performance(
        nrrs=[100, 50, 20],
        mean_acc={
            "PCC": [82.1, 80.4, 76.9],
            "PG-AC": [85.2, 84.6, 81.7],
        },
        std_acc={
            "PCC": [2.1, 2.4, 2.8],
            "PG-AC": [1.8, 1.9, 2.3],
        },
        title="Example Performance (Placeholder Data)",
    )

    # 7. Original six-method placeholder example
    plot_example_classification_performance()

    plt.show()