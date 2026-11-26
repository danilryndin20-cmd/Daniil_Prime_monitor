"""Расчёты для пилотов (A/B-тестов): семинары С7 и С8.

Импорты — только стандартная библиотека, поэтому модуль работает и в основном
окружении prime-monitor (pandas 3), и в окружении pilot/ (HypEx, pandas 2.2).
"""

import math
from statistics import NormalDist

z = NormalDist().inv_cdf


# --- С7: сколько клиентов нужно и какой эффект заметим -----------------------

def n_per_group(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                comparisons: int = 1) -> int:
    """Клиентов на КАЖДУЮ группу, чтобы заметить эффект mde с заданной мощностью.

    Поправка Бонферрони на число сравнений с контролем: alpha делится на comparisons.
    """
    za = z(1 - alpha / (2 * comparisons))   # двусторонний критерий
    zb = z(power)                            # квантиль, отвечающий за мощность
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2)


def mde_for_n(sigma: float, n: int, alpha: float = 0.05, power: float = 0.8,
              comparisons: int = 1) -> float:
    """Минимальный эффект, который заметит пилот с n клиентами на группу.

    Обратная к n_per_group: MDE = (z_{1-α/2k} + z_{мощность}) · σ · √(2/n).
    """
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return (za + zb) * sigma * math.sqrt(2 / n)


# --- С8: эффект с интервалом, SRM, CUPED -------------------------------------

def diff_ci(t, c, alpha: float = 0.05) -> dict:
    """Разность средних test − control, её (1 − alpha) интервал и p-значение.

    Стандартная ошибка — по дисперсиям каждой группы отдельно (как у Уэлча),
    квантиль — нормальный. t и c — pandas.Series.
    """
    d = t.mean() - c.mean()
    se = (t.var() / len(t) + c.var() / len(c)) ** 0.5
    q = NormalDist().inv_cdf(1 - alpha / 2)
    p = 2 * (1 - NormalDist().cdf(abs(d / se)))
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "p": p}


def srm_pvalue(sizes) -> float:
    """p-значение χ² согласия с равными долями для 2 или 3 групп (без scipy).

    При k − 1 степенях свободы: две группы — erfc(√(χ²/2)), три — exp(−χ²/2).
    """
    sizes = list(sizes)
    n = sum(sizes)
    exp = n / len(sizes)
    chi2 = sum((s - exp) ** 2 / exp for s in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    if len(sizes) == 3:
        return math.exp(-chi2 / 2)
    raise ValueError("для 4 и более групп — scipy.stats.chisquare")


def cuped(y, x):
    """Метрика после поправки CUPED и theta; y и x — pandas.Series одной длины.

    x обязан быть измерен ДО назначения групп, иначе поправка съест эффект.
    """
    theta = y.cov(x) / x.var()
    return y - theta * (x - x.mean()), theta
