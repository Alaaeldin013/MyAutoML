
def detect_task(y):
    n_unique = y.nunique()
    n_total = len(y)
    ratio = n_unique / n_total

    # 🧠 Rule 1: Object / category → classification
    if y.dtype == "object" or str(y.dtype) == "category":
        return "classification"

    # 🧠 Rule 2: Boolean → classification
    if y.dtype == "bool":
        return "classification"

    # 🧠 Rule 3: Few unique values → classification
    if n_unique <= 15:
        return "classification"

    # 🧠 Rule 4: Low ratio of unique values → classification
    if ratio < 0.05:
        return "classification"

    # Otherwise → regression
    return "regression"