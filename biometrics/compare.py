"""
Zwraca podobieństwo kosinusowe i werdykt MATCH / NO MATCH.
"""
from __future__ import annotations

import argparse
import sys

from biometrics.engine import extract_single_embedding, load_image
from biometrics.matching import DEFAULT_THRESHOLD, cosine_similarity


def main() -> int:
    parser = argparse.ArgumentParser(description="Porównanie dwóch twarzy.")
    parser.add_argument("image_a", help="ścieżka do pierwszego zdjęcia")
    parser.add_argument("image_b", help="ścieżka do drugiego zdjęcia")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"próg dopasowania (domyślnie {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    try:
        emb_a = extract_single_embedding(load_image(args.image_a))
        emb_b = extract_single_embedding(load_image(args.image_b))
    except (ValueError, FileNotFoundError) as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    score = cosine_similarity(emb_a, emb_b)
    verdict = "MATCH" if score >= args.threshold else "NO MATCH"
    print(f"Podobieństwo: {score:.4f} (próg {args.threshold})  ->  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
