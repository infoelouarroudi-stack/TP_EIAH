from pathlib import Path
import argparse

from src.recommender import (
    load_students,
    extract_hexad_profile,
    extract_motivation_initial_profile,
    affinity_vector,
    combine_affinities,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="Identifiant élève (colonne User), ex: elevebf01")
    parser.add_argument("--alpha", type=float, default=0.1, help="Seuil p-value (par défaut 0.1)")
    parser.add_argument("--data_dir", default="data", help="Dossier data/")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = load_students(data_dir / "userStats.csv")

    if args.user not in set(df["User"]):
        raise SystemExit(f"User inconnu: {args.user}")

    row = df.loc[df["User"] == args.user].iloc[0]

    hexad_profile = extract_hexad_profile(row)
    motiv_profile = extract_motivation_initial_profile(row)

    vec_hexad = affinity_vector(data_dir / "Hexad", hexad_profile, alpha=args.alpha)
    vec_motiv = affinity_vector(data_dir / "Motivation", motiv_profile, alpha=args.alpha)

    final = combine_affinities(vec_hexad, vec_motiv, w_hexad=0.5, w_motiv=0.5)

    print("\n=== Vecteur affinité HEXAD ===")
    print(vec_hexad)

    print("\n=== Vecteur affinité MOTIVATION ===")
    print(vec_motiv)

    print("\n=== Combinaison finale ===")
    print(final)

    reco = final.loc[0, "element"]
    print("\nRecommandation finale:", reco)

    # sauvegarde (pratique pour rendre)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    vec_hexad.to_csv(out_dir / f"{args.user}_affinity_hexad.csv", index=False)
    vec_motiv.to_csv(out_dir / f"{args.user}_affinity_motivation.csv", index=False)
    final.to_csv(out_dir / f"{args.user}_affinity_combined.csv", index=False)

if __name__ == "__main__":
    main()
