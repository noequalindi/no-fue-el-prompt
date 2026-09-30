"""Benchmark 'sliced': no alcanza con un accuracy global.

Estas funciones cortan el resultado por género y por congruencia con el
estereotipo (columna `stereotype`: 1 = congruente con el estereotipo
ocupacional, -1 = incongruente), que es donde suele esconderse el sesgo.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def slice_report(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    rows = []

    def add(name, subset):
        if len(subset) == 0:
            return
        rows.append(
            {
                "corte": name,
                "n": len(subset),
                "accuracy": accuracy_score(subset[y_true_col], subset[y_pred_col]),
                "f1_macro": f1_score(
                    subset[y_true_col], subset[y_pred_col], average="macro"
                ),
            }
        )

    add("overall", df)
    for gender, sub in df.groupby("gender"):
        add(f"gender={gender}", sub)
    if "stereotype" in df.columns:
        add("stereotype (congruente)", df[df["stereotype"] == 1])
        add("anti-stereotype (incongruente)", df[df["stereotype"] == -1])

    report = pd.DataFrame(rows).set_index("corte")

    if {"gender=male", "gender=female"} <= set(report.index):
        report.loc["Δ gender (|male - female|)", "accuracy"] = abs(
            report.loc["gender=male", "accuracy"] - report.loc["gender=female", "accuracy"]
        )
    if {"stereotype (congruente)", "anti-stereotype (incongruente)"} <= set(report.index):
        report.loc["Δ stereotype", "accuracy"] = abs(
            report.loc["stereotype (congruente)", "accuracy"]
            - report.loc["anti-stereotype (incongruente)", "accuracy"]
        )
    return report


def profession_breakdown(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, min_support: int = 15
) -> pd.DataFrame:
    """Accuracy por profesión, para encontrar las más castigadas por el sesgo."""
    rows = []
    for profession, sub in df.groupby("profession"):
        if len(sub) < min_support:
            continue
        rows.append(
            {
                "profession": profession,
                "n": len(sub),
                "accuracy": accuracy_score(sub[y_true_col], sub[y_pred_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)


def summarize(df: pd.DataFrame, pred_col: str) -> dict:
    """Resumen de una fila para la tabla final comparativa entre modelos."""
    report = slice_report(df, "label", pred_col)
    return {
        "accuracy": report.loc["overall", "accuracy"],
        "Δ gender": report.loc["Δ gender (|male - female|)", "accuracy"]
        if "Δ gender (|male - female|)" in report.index
        else np.nan,
        "Δ stereotype": report.loc["Δ stereotype", "accuracy"]
        if "Δ stereotype" in report.index
        else np.nan,
    }
