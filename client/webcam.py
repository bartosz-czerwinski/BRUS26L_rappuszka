"""Demo rozpoznawania twarzy na żywo z kamery (klient kioskowy, Faza 1).

Lokalne demo BEZ serwera — pokazuje samą biometrię. Komunikacja z serwerem
(enroll/identify przez API) dochodzi w kolejnych fazach.

Sterowanie (okno musi być aktywne):
    r        - zarejestruj twarz z kadru jako wzorzec (referencję)
    c        - wyczyść wzorzec
    q / ESC  - wyjście

Uruchomienie (z katalogu projektu):
    $env:PYTHONPATH = (Get-Location).Path
    python -m client.webcam
    # opcjonalnie: --camera 1  (inny indeks kamery), --threshold 0.4

Uwaga: napisy nakładane na obraz są bez polskich znaków — czcionka OpenCV
ich nie renderuje. Komunikaty w konsoli są pełne.
"""
from __future__ import annotations

import argparse

import cv2

from biometrics.engine import analyze
from biometrics.matching import DEFAULT_THRESHOLD, cosine_similarity

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo rozpoznawania twarzy z kamery.")
    parser.add_argument("--camera", type=int, default=0, help="indeks kamery (domyślnie 0)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"próg dopasowania (domyślnie {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    # CAP_DSHOW przyspiesza otwieranie kamery na Windows.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Nie udało się otworzyć kamery o indeksie {args.camera}.")
        print("Spróbuj innego indeksu: python -m client.webcam --camera 1")
        return 1

    reference = None
    print("Sterowanie: [r] rejestruj wzorzec, [c] wyczysc, [q]/[ESC] wyjscie")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Nie udało się odczytać klatki z kamery.")
            break

        faces = analyze(frame)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            color, label = GREEN, "twarz"
            if reference is not None:
                score = cosine_similarity(reference, face.normed_embedding)
                match = score >= args.threshold
                color = GREEN if match else RED
                label = f"{'MATCH' if match else 'NO MATCH'} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), FONT, 0.6, color, 2)

        hint = "wzorzec USTAWIONY" if reference is not None else "brak wzorca - nacisnij 'r'"
        cv2.putText(frame, hint, (10, 30), FONT, 0.7, YELLOW, 2)
        cv2.imshow("BRUS - demo biometrii", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            reference = None
            print("Wyczyszczono wzorzec.")
        if key == ord("r"):
            if len(faces) == 1:
                reference = faces[0].normed_embedding
                print("Zarejestrowano wzorzec.")
            elif len(faces) == 0:
                print("Nie wykryto twarzy - sprobuj ponownie.")
            else:
                print(f"Wykryto {len(faces)} twarzy - w kadrze ma byc dokladnie jedna.")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
