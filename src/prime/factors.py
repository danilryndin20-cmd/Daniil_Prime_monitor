"""Факторное разложение (семинар 6): цепные подстановки и каскадная диаграмма.

Договорённость курса: сначала «сколько» (количество), потом «почём» (цена).
Сумма влияний всегда равна изменению показателя — это проверяет assert.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd


def chain(n0: float, p0: float, n1: float, p1: float) -> tuple[float, float]:
    """Цепные подстановки для R = N × P: сначала количество, потом цена."""
    by_count = (n1 - n0) * p0          # количество меняется при старой цене
    by_price = n1 * (p1 - p0)          # цена меняется при уже новом количестве
    assert abs(by_count + by_price - (n1 * p1 - n0 * p0)) < 0.01, "сумма влияний не равна изменению"
    return by_count, by_price


def contrib(x0: dict, x1: dict, order: list[str]) -> dict:
    """Влияние каждого фактора произведения: меняем их по одному в порядке order."""
    cur, out = dict(x0), {}
    for f in order:
        before = math.prod(cur.values())
        cur[f] = x1[f]
        out[f] = math.prod(cur.values()) - before
    total = math.prod(x1.values()) - math.prod(x0.values())
    assert abs(sum(out.values()) - total) < 0.01, "сумма влияний не равна изменению"
    return out


def waterfall(deltas: pd.Series, title: str, unit: str = "тыс. руб.") -> plt.Axes:
    """Каскад изменений: по столбику на фактор и «итого» в конце. Ось включает ноль."""
    before = deltas.cumsum() - deltas      # уровень, на котором закончился предыдущий столбик
    colors = ["#C1272D" if v < 0 else "#2A9D8F" for v in deltas]

    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    bars = ax.bar(deltas.index, deltas, bottom=before, color=colors, zorder=2)
    total = ax.bar("итого", deltas.sum(), color="#6B7280", zorder=2)
    for group in (bars, total):
        ax.bar_label(group, fmt=lambda v: f"{v:+,.0f}".replace(",", " ").replace("-", "−"), padding=3)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title(title, loc="left")
    ax.set_ylabel(unit)
    ax.use_sticky_edges = False
    ax.margins(y=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB")
    return ax
