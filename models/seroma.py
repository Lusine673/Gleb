# Модель: риск серомы в раннем ПОП (Таблица 9)
# Формула: logit = b0 + b1*Тип вмешательства + b2*Грыжесечение в анамнезе + b3*ASA + b4*ИМТ

MODEL_NAME = "Риск серомы (ранний ПОП)"
# ВАЖНО: уточните кодировку "Тип вмешательства".
# Ниже предположено: 0=TEP, 1=TAPP. Измените при необходимости.
FIELDS = [
    dict(key="intervention_type", label="Тип вмешательства", type="select",
         options={"TEP": 0, "TAPP": 1}),
    dict(key="prior_hernia_surgery", label="Грыжесечение в анамнезе", type="checkbox"),
    dict(key="asa", label="Оценка ASA (1–4)", type="number", min=1, max=4, step=1, default=2),
    dict(key="bmi", label="ИМТ (кг/м²)", type="number", min=10.0, max=60.0, step=0.1, default=26.0),
]

# Коэффициенты B из таблицы (проверьте с первоисточником!)
COEF = {
    "intercept": 1.669,
    # Судя по Exp(B)=0.377, B ≈ -0.977 (а не -0.0977). Проверьте оригинал!
    "intervention_type": -0.977,
    "prior_hernia_surgery": 2.018,
    "asa": -1.418,
    "bmi": -0.007,
}

# Порог для цветовой индикации риска в UI
DEFAULT_THRESHOLD = 0.30
