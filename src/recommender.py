from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


ELEMENTS = ["avatar", "badges", "score", "timer", "progress", "ranking"]


def _norm_label(x: str) -> str:
    # normalise index/colonnes (espaces, guillemets, casse)
    s = str(x).strip().strip('"').strip("'")
    s_low = s.lower()
    # uniformiser amotvar / amotVar / AMOTVar
    if s_low == "amotvar" or s_low == "amotvar ":
        return "amotVar"
    if s_low == "mivar":
        return "MIVar"
    if s_low == "mevar":
        return "MEVar"
    if s_low == "amoti":
        return "amotI"
    if s_low == "mi":
        return "MI"
    if s_low == "me":
        return "ME"
    return s  # garder tel quel sinon


def load_pls_matrix(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", index_col=0)
    df.index = [_norm_label(i) for i in df.index]
    df.columns = [_norm_label(c) for c in df.columns]
    return df


def mask_by_pvalue(coefs: pd.DataFrame, pvals: pd.DataFrame, alpha: float) -> pd.DataFrame:
    # aligne indices/colonnes puis met à 0 si p>=alpha
    coefs = coefs.copy()
    pvals = pvals.copy()
    coefs = coefs.loc[pvals.index, pvals.columns]
    return coefs.where(pvals < alpha, other=0.0)


def predict_variations(coefs_sig: pd.DataFrame, profile: pd.Series) -> pd.Series:
    """
    coefs_sig: DataFrame rows = [MIVar, MEVar, amotVar], cols = variables d'entrée
    profile: Series index = mêmes variables d'entrée
    """
    # alignement sur variables d'entrée
    common = coefs_sig.columns.intersection(profile.index)
    if len(common) == 0:
        raise ValueError("Aucune variable commune entre profil et matrices PLS.")
    M = coefs_sig[common].astype(float).values  # (3, d)
    x = profile[common].astype(float).values    # (d,)
    y = M @ x                                  # (3,)
    out = pd.Series(y, index=coefs_sig.index)
    # garantir la présence des 3 clés standard
    for k in ["MIVar", "MEVar", "amotVar"]:
        if k not in out.index:
            out[k] = 0.0
    return out[["MIVar", "MEVar", "amotVar"]]


def objective_score(variations: pd.Series, w_mi=1.0, w_me=1.0, w_amot=1.0) -> float:
    # objectif TP : +MIvar +MEvar -amotVar [TP]
    return float(w_mi * variations["MIVar"] + w_me * variations["MEVar"] - w_amot * variations["amotVar"])


@dataclass
class AffinityResult:
    element: str
    score: float
    pred_MIVar: float
    pred_MEVar: float
    pred_amotVar: float


def affinity_vector(pls_folder: str | Path, profile: pd.Series, alpha: float) -> pd.DataFrame:
    """
    Calcule le vecteur d’affinité (6 éléments) pour un profil donné et un dossier PLS (Hexad ou Motivation).
    """
    pls_folder = Path(pls_folder)
    rows = []

    for el in ELEMENTS:
        coefs_path = pls_folder / f"{el}PathCoefs.csv"
        pvals_path = pls_folder / f"{el}pVals.csv"

        coefs = load_pls_matrix(coefs_path)
        pvals = load_pls_matrix(pvals_path)

        coefs_sig = mask_by_pvalue(coefs, pvals, alpha=alpha)
        pred = predict_variations(coefs_sig, profile)
        score = objective_score(pred)

        rows.append(
            AffinityResult(
                element=el,
                score=score,
                pred_MIVar=float(pred["MIVar"]),
                pred_MEVar=float(pred["MEVar"]),
                pred_amotVar=float(pred["amotVar"]),
            ).__dict__
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if np.isclose(s.max(), s.min()):
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def combine_affinities(hexad_vec: pd.DataFrame, motiv_vec: pd.DataFrame, w_hexad: float = 0.5, w_motiv: float = 0.5) -> pd.DataFrame:
    """
    Combine les 2 vecteurs (Hexad + Motivation) en normalisant les scores puis moyenne pondérée.
    """
    df = hexad_vec[["element", "score"]].merge(
        motiv_vec[["element", "score"]],
        on="element",
        suffixes=("_hexad", "_motiv")
    )
    df["s_hexad"] = minmax(df["score_hexad"])
    df["s_motiv"] = minmax(df["score_motiv"])
    df["score_final"] = w_hexad * df["s_hexad"] + w_motiv * df["s_motiv"]
    return df.sort_values("score_final", ascending=False).reset_index(drop=True)


def load_students(user_stats_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(user_stats_csv, sep=";")
    df.columns = df.columns.str.strip()
    return df


def extract_hexad_profile(student_row: pd.Series) -> pd.Series:
    cols = ["achiever", "player", "socialiser", "freeSpirit", "disruptor", "philanthropist"]
    return student_row[cols].astype(float)


def extract_motivation_initial_profile(student_row: pd.Series) -> pd.Series:
    # agrégation cohérente avec le TP: MI (3 sous-dim), ME (3 sous-dim), amotI [TP]
    mico = float(student_row["micoI"])
    miac = float(student_row["miacI"])
    mist = float(student_row["mistI"])
    meid = float(student_row["meidI"])
    mein = float(student_row["meinI"])
    mere = float(student_row["mereI"])
    amotI = float(student_row["amotI"])

    MI = (mico + miac + mist) / 3.0
    ME = (meid + mein + mere) / 3.0
    return pd.Series({"MI": MI, "ME": ME, "amotI": amotI})
