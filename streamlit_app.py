import json
import math
import pandas as pd
import streamlit as st

# --------------------------
# Настройки страницы
# --------------------------
st.set_page_config(
    page_title="Риск осложнений после лапароскопической герниопластики",
    page_icon="🩺",
    layout="centered",
)

# --------------------------
# Утилиты
# --------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def logit_and_proba(coef: dict, x: dict, intercept: float = 0.0):
    s = float(intercept)
    for k, v in x.items():
        s += float(coef.get(k, 0.0)) * float(v)
    return s, sigmoid(s)

def contributions(coef: dict, x: dict) -> pd.DataFrame:
    rows = []
    for k, v in x.items():
        beta = float(coef.get(k, 0.0))
        xv = float(v)
        rows.append({"Признак": k, "β": beta, "x": xv, "β·x": beta * xv})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("β·x", ascending=False)
    return df

def batch_predict(df: pd.DataFrame, mapping: dict, coef: dict, intercept: float = 0.0) -> pd.DataFrame:
    work = df.copy()
    # Начинаем с константы
    logit = pd.Series(float(intercept), index=work.index, dtype="float64")
    # Складываем вклад каждого признака (NaN -> 0)
    for feat, col in mapping.items():
        vals = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
        logit = logit + vals * float(coef.get(feat, 0.0))
    res = work.copy()
    res["logit"] = logit
    res["probability"] = res["logit"].apply(sigmoid)
    return res

# --------------------------
# Конфиги моделей (встроенные)
# Можно редактировать прямо в UI в режиме "Коэффициенты/конфиг"
# --------------------------
DEFAULT_MODELS = {
    "Серома": {
        "name": "Риск серомы (ранний ПОП)",
        "intercept": 1.669,
        "threshold": 0.30,
        "features": [
            {"key": "intervention_type", "label": "Тип вмешательства (0/1)", "type": "int", "help": "0=база, 1=альтернатива"},
            {"key": "prior_hernia_surgery", "label": "Грыжесечение в анамнезе (0/1)", "type": "int"},
            {"key": "asa", "label": "ASA (число)", "type": "number"},
            {"key": "bmi", "label": "ИМТ (число)", "type": "number"}
        ],
        "coef": {
            # Внимание: b для типа вмешательства по таблице может быть -0.977, проверьте по исходнику
            "intervention_type": -0.977,
            "prior_hernia_surgery": 2.018,
            "asa": -1.418,
            "bmi": -0.007
        }
    },
    "Боль": {
        "name": "Риск боли (ранний ПОП)",
        "intercept": -62.457,
        "threshold": 0.30,
        "features": [
            {"key": "age", "label": "Возраст, лет", "type": "int"},
            {"key": "obesity", "label": "Ожирение (0/1)", "type": "int"},
            {"key": "hernia_type_binary", "label": "Тип грыжи (0/1)", "type": "int"},
            {"key": "diabetes", "label": "Сахарный диабет (0/1)", "type": "int"},
            {"key": "asthma", "label": "Бронхиальная астма (0/1)", "type": "int"},
            {"key": "hypertension", "label": "Гипертоническая болезнь (0/1)", "type": "int"},
            {"key": "ctd", "label": "Заболевания соединительной ткани (0/1)", "type": "int"},
            {"key": "cvi", "label": "Хроническая венозная недостаточность (0/1)", "type": "int"},
            {"key": "hemorrhoids", "label": "Геморрой (0/1)", "type": "int"},
            {"key": "asa_high", "label": "ASA ≥ 3 (0/1)", "type": "int"},
            {"key": "intervention_type", "label": "Тип вмешательства (0/1)", "type": "int"},
            {"key": "prior_operation", "label": "Оперативные вмешательства в анамнезе (любые) (0/1)", "type": "int"},
            {"key": "prior_hernia_surgery", "label": "Грыжесечение в анамнезе (0/1)", "type": "int"},
            {"key": "duration_long", "label": "Длительная операция (0/1)", "type": "int"},
            {"key": "fixation_method", "label": "Метод фиксации (0/1)", "type": "int"}
        ],
        "coef": {
            "age": 0.055,
            "obesity": 1.541,
            "hernia_type_binary": -0.930,
            "diabetes": 3.486,
            "asthma": -1.277,
            "hypertension": 3.290,
            "ctd": 20.762,
            "cvi": 2.897,
            "hemorrhoids": -20.295,
            "asa_high": 3.495,
            "intervention_type": 6.063,
            "prior_operation": -3.389,
            "prior_hernia_surgery": 2.069,
            "duration_long": 2.605,
            "fixation_method": -0.956
        }
    }
}

# --------------------------
# Работа с текущей моделью в сессии
# --------------------------
def get_model_cfg() -> dict:
    if "model_cfg" not in st.session_state:
        st.session_state["model_cfg"] = DEFAULT_MODELS["Серома"]
    return st.session_state["model_cfg"]

def set_model_cfg(cfg: dict):
    st.session_state["model_cfg"] = cfg

# --------------------------
# Рендер блоков UI
# --------------------------
def render_single_calc(cfg: dict):
    st.subheader(cfg.get("name", "Модель"))
    coef = cfg["coef"]
    intercept = float(cfg.get("intercept", 0.0))
    fields = cfg["features"]

    st.caption("Введите числовые значения признаков в той же кодировке, что использовалась при расчете коэффициентов (обычно 0/1 для бинарных).")

    values = {}
    cols = st.columns(2)
    for i, f in enumerate(fields):
        col = cols[i % 2]
        t = f.get("type", "number")
        label = f.get("label", f["key"])
        help_ = f.get("help")
        if t == "int":
            values[f["key"]] = col.number_input(label, value=0, step=1, format="%d", help=help_)
        else:
            values[f["key"]] = col.number_input(label, value=0.0, step=0.1, help=help_)

    logit, p = logit_and_proba(coef, values, intercept)

    st.write("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Логит", f"{logit:.3f}")
    c2.metric("Вероятность", f"{p*100:.1f}%")
    thr = float(cfg.get("threshold", 0.30))
    risk_label = "Высокий риск" if p >= thr else "Низкий/умеренный риск"
    c3.metric("Класс риска", risk_label)

    with st.expander("Вклад признаков (β·x)", expanded=False):
        df = contributions(coef, values)
        st.dataframe(df, hide_index=True, use_container_width=True)

def render_batch_calc(cfg: dict):
    st.subheader(cfg.get("name", "Модель") + " — пакетный расчёт по CSV")
    st.caption("Загрузите CSV, затем сопоставьте колонки с признаками модели. Значения должны быть числовыми (0/1 или число).")

    file = st.file_uploader("CSV-файл", type=["csv"])
    if not file:
        st.info("Подсказка: назовите колонки так же, как ключи признаков в модели — тогда сопоставление подставится автоматически.")
        return

    df = pd.read_csv(file)
    st.write("Предпросмотр:")
    st.dataframe(df.head(), use_container_width=True)

    fields = cfg["features"]
    coef = cfg["coef"]
    intercept = float(cfg.get("intercept", 0.0))

    st.write("Сопоставление колонок CSV с признаками модели:")
    mapping = {}
    cols = st.columns(2)
    for i, f in enumerate(fields):
        col = cols[i % 2]
        key = f["key"]
        # авто-совпадение по имени
        default_idx = 0
        options = ["<не выбрано>"] + list(df.columns)
        for j, c in enumerate(df.columns, start=1):
            if c.strip().lower() == key.strip().lower():
                default_idx = j
                break
        sel = col.selectbox(f"{key}", options=options, index=default_idx)
        if sel != "<не выбрано>":
            mapping[key] = sel

    missing = [f["key"] for f in fields if f["key"] not in mapping]
    if missing:
        st.warning("Не сопоставлены признаки: " + ", ".join(missing) + ". Можно продолжить (несопоставленные будут считаться 0), или сопоставить все.")
    proceed = st.checkbox("Продолжить расчёт даже если не все признаки сопоставлены", value=(len(missing) == 0))

    if proceed and st.button("Рассчитать"):
        # Заполним отсутствующие сопоставления фиктивно нулевыми столбцами
        tmp = df.copy()
        for m in missing:
            tmp[f"__zero__{m}"] = 0.0
            mapping[m] = f"__zero__{m}"

        res = batch_predict(tmp, mapping, coef, intercept)
        st.success("Готово. Ниже первые строки результата.")
        st.dataframe(res.head(), use_container_width=True)

        csv_bytes = res.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Скачать результат CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

def render_config_editor(cfg: dict):
    st.subheader("Коэффициенты / конфиг модели")
    st.caption("Можно отредактировать текущий JSON-конфиг или загрузить свой.")

    uploaded = st.file_uploader("Загрузить JSON", type=["json"])
    if uploaded:
        try:
            new_cfg = json.load(uploaded)
            set_model_cfg(new_cfg)
            st.success("Новый конфиг загружен и применён.")
            cfg = new_cfg
        except Exception as e:
            st.error(f"Ошибка чтения JSON: {e}")

    text = st.text_area(
        "Текущий конфиг (редактируемый JSON):",
        value=json.dumps(cfg, ensure_ascii=False, indent=2),
        height=420,
    )
    if st.button("Применить изменения"):
        try:
            new_cfg = json.loads(text)
            set_model_cfg(new_cfg)
            st.success("Изменения применены.")
        except Exception as e:
            st.error(f"Ошибка парсинга JSON: {e}")

    st.download_button(
        "Скачать текущий JSON",
        data=json.dumps(get_model_cfg(), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="model_config.json",
        mime="application/json",
    )

# --------------------------
# Главная функция
# --------------------------
def main():
    st.title("Прогноз осложнений после лапароскопической герниопластики")
    st.caption("Исследовательский инструмент. Модель логистической регрессии задаётся коэффициентами β, свободным членом и списком признаков.")

    with st.sidebar:
        st.header("Модель")
        choice = st.selectbox("Выберите преднастройку", list(DEFAULT_MODELS.keys()))
        if st.button("Загрузить преднастройку"):
            set_model_cfg(DEFAULT_MODELS[choice])

        cfg = get_model_cfg()
        st.write("Текущая модель:", cfg.get("name", "—"))

        mode = st.radio("Режим работы", ["Один пациент", "CSV пакет", "Коэффициенты/конфиг"])

    cfg = get_model_cfg()
    if mode == "Один пациент":
        render_single_calc(cfg)
    elif mode == "CSV пакет":
        render_batch_calc(cfg)
    else:
        render_config_editor(cfg)

    st.divider()
    st.caption(
        "Формула: p = 1 / (1 + exp(-(b0 + Σ β_i·x_i))). "
        "Для бинарных признаков используйте 0/1; для количественных — числовые значения. "
        "Коэффициенты для преднастроек взяты из предоставленных таблиц и могут требовать уточнения."
    )

if __name__ == "__main__":
    if "model_cfg" not in st.session_state:
        set_model_cfg(DEFAULT_MODELS["Серома"])
    main()
