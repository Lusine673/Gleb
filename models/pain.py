# Модель: риск боли в раннем ПОП (Таблица 14)
# В этой таблице большинство предикторов, вероятно, бинарные (0/1).
# Явно проверьте кодировку по вашей диссертации/статье и поправьте словарь COEF ниже.

MODEL_NAME = "Риск боли (ранний ПОП)"

# TODO: при необходимости переименуйте и уточните кодировку категорий.
FIELDS = [
    dict(key="age", label="Возраст (лет)", type="number", min=16, max=100, step=1, default=45),
    dict(key="obesity", label="Ожирение (ИМТ ≥ 30)", type="checkbox"),
    dict(key="hernia_type_binary", label="Тип грыжи (дихотомия)", type="select",
         # Пример кодировки: 0 = «базовый тип», 1 = «альтернативный тип». Уточните смысл.
         options={"Базовый": 0, "Альтернативный": 1}),
    dict(key="diabetes", label="Сахарный диабет", type="checkbox"),
    dict(key="asthma", label="Бронхиальная астма", type="checkbox"),
    dict(key="hypertension", label="Гипертоническая болезнь", type="checkbox"),
    dict(key="ctd", label="Заболевания соединительной ткани", type="checkbox"),
    dict(key="cvi", label="Хроническая венозная недостаточность (ХВН)", type="checkbox"),
    dict(key="hemorrhoids", label="Геморрой", type="checkbox"),
    dict(key="asa_high", label="ASA ≥ 3", type="checkbox"),
    dict(key="intervention_type", label="Тип вмешательства (TAPP/TEP)", type="select",
         options={"TEP (база)": 0, "TAPP": 1}),
    dict(key="prior_operation", label="Оперативные вмешательства в анамнезе (любые)", type="checkbox"),
    dict(key="prior_hernia_surgery", label="Грыжесечение в анамнезе", type="checkbox"),
    dict(key="duration_long", label="Длительность операции (длинная, порог по протоколу)", type="checkbox"),
    dict(key="fixation_method", label="Метод фиксации (кат. признак)", type="select",
         # Пример: 0 = самофиксирующаяся/клей, 1 = такеры. Уточните.
         options={"Самофикс./клей (база)": 0, "Такеры": 1}),
]

# Коэффициенты B из таблицы (перепроверьте!)
# Интерсепт очень большой по модулю — так в источнике. Без «включенных» факторов базовый риск ≈ 0.
COEF = {
    "intercept": -62.457,
    "age": 0.055,               # Предположительно линейно по годам. Проверьте!
    "obesity": 1.541,
    "hernia_type_binary": -0.930,
    "diabetes": 3.486,
    "asthma": -1.277,
    "hypertension": 3.290,
    "ctd": 20.762,              # Очень крупный; вероятно редкое событие/нестабильная оценка.
    "cvi": 2.897,
    "hemorrhoids": -20.295,     # Нестабильно; узкие данные.
    "asa_high": 3.495,
    "intervention_type": 6.063,
    "prior_operation": -3.389,
    "prior_hernia_surgery": 2.069,
    "duration_long": 2.605,
    "fixation_method": -0.956,
}

DEFAULT_THRESHOLD = 0.30
