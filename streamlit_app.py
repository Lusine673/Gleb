import streamlit as st
import pandas as pd

from utils import logit_and_proba, contributions
from models import seroma as m_seroma
from models import pain as m_pain

st.set_page_config(page_title="Риск осложнений после лапароскопической герниопластики",
                   page_icon="🩺",
                   layout="centered")

def render_form(fields):
    values = {}
    for f in fields:
        key = f["key"]
        label = f["label"]
        typ = f["type"]

        if typ == "number":
            values[key] = st.number_input(
                label,
                value=float(f.get("default", 0)),
                min_value=float(f.get("min", -1e9)),
                max_value=float(f.get("max", 1e9)),
                step=float(f.get("step", 1)),
                key=key
            )
        elif typ == "checkbox":
            values[key] = 1 if st.checkbox(label, value=f.get("default", False), key=key) else 0
        elif typ == "select":
            opts = f.get("options", {})
            # Отображаем подписи, возвращаем код
            label_to_code = opts
            label_selected = st.selectbox(label, list(label_to_code.keys()), key=key)
            values[key] = label_to_code[label_selected]
        else:
            st.warning(f"Неизвестный тип поля: {typ}")
    return values

def render_model_block(name, fields, coef, default_threshold=0.30):
    st.subheader(name)

    with st.expander("Примечания к вводу", expanded=False):
        st.write(
            "- Для чекбоксов 1 = «есть признак», 0 = «нет».\n"
            "- Для выпадающих списков указан код, который идет в модель (базовая категория = 0).\n"
            "- Проверьте кодировку категорий согласно исходной работе."
        )

    values = render_form(fields)

    # Расчет
    lg, p = logit_and_proba(coef, values)
    contr = contributions(coef, values)
    thr = default_threshold

    st.markdown("—")
    cols = st.columns(3)
    cols[0].metric("Логит", f"{lg:.3f}")
    cols[1].metric("Вероятность", f"{p*100:.1f}%")
    risk_label = "Высокий риск" if p >= thr else "Низкий/умеренный риск"
    cols[2].metric("Класс риска", risk_label)

    with st.expander("Вклад признаков (β·x)", expanded=False):
        df = pd.DataFrame(
            [{"Признак": k, "β": coef.get(k, 0.0), "x": values[k], "β·x": v} for k, v in contr.items()]
        ).sort_values("β·x", ascending=False)
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.caption(f"Порог визуализации риска: {int(thr*100)}% (можно изменить в коде модели).")

st.title("Прогноз осложнений после лапароскопической герниопластики")
st.write("Приложение рассчитывает риск серомы и риск боли в раннем послеоперационном периоде по данным логистических регрессий из вашего исследования. Пожалуйста, перепроверьте кодировку категориальных переменных и коэффициенты.")

tab1, tab2 = st.tabs(["Серома", "Боль"])

with tab1:
    render_model_block(m_seroma.MODEL_NAME, m_seroma.FIELDS, m_seroma.COEF, m_seroma.DEFAULT_THRESHOLD)

with tab2:
    render_model_block(m_pain.MODEL_NAME, m_pain.FIELDS, m_pain.COEF, m_pain.DEFAULT_THRESHOLD)

st.divider()
st.caption("Не является медицинским изделием. Результаты предназначены только для исследовательских целей и не заменяют клиническое решение.")
