import math
from typing import Dict, Tuple

def sigmoid(x: float) -> float:
    # Численно устойчивая логистическая функция
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def logit_and_proba(coef: Dict[str, float], x: Dict[str, float]) -> Tuple[float, float]:
    """Возвращает (логит, вероятность) по словарям коэффициентов и значений признаков."""
    s = coef.get("intercept", 0.0)
    for k, v in x.items():
        if k == "intercept":
            continue
        beta = coef.get(k, 0.0)
        s += beta * float(v)
    return s, sigmoid(s)

def contributions(coef: Dict[str, float], x: Dict[str, float]) -> Dict[str, float]:
    """Вклад каждого признака в логиты (beta * x)."""
    out = {}
    for k, v in x.items():
        if k == "intercept":
            continue
        out[k] = coef.get(k, 0.0) * float(v)
    return out
