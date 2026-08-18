# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 02:31:55 2026

@author: 18307
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc


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
    for row in data.itertuples(index=False):
        px = float(row.plot_x)
        py = float(row.plot_y)
        selected = bool(row.is_selected)

        # channel_col 默认是 channel。
        # 为兼容自定义列名，这里使用 getattr。
        channel = str(
            getattr(row, channel_col)
        )

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

# %%
ch_index_62 = list(range(1,63))
ch_index_32 = [1,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,53,55,59,60,61]
ch_index_16 = [1,3,8,10,12,24,26,28,30,32,44,46,48,59,60,61]
ch_index_8 = [1,3,26,30,44,48,59,61]
ch_index_4 = [1,3,44,48]

ch_index_62 = [ch - 1 for ch in ch_index_62]
ch_index_32 = [ch - 1 for ch in ch_index_32]
ch_index_16 = [ch - 1 for ch in ch_index_16]
ch_index_8 = [ch - 1 for ch in ch_index_8]
ch_index_4 = [ch - 1 for ch in ch_index_4]

from utils import utils_feature_loading

df_all = utils_feature_loading.read_distribution("seed")
df_subset = df_all.iloc[ch_index_16]

# %%
fig, ax, plot_df = plot_electrode_selection(
    df_all=df_all,
    df_subset=df_subset,
    title="Selected EEG electrodes",
)

plt.show()