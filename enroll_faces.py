#!/usr/bin/env python3
import os, json, sys
import numpy as np
import face_recognition as fr

PEOPLE_DIR = "people"
DB_PATH = "faces_db.npz"

def main():
    names, encs = [], []
    for person in sorted(os.listdir(PEOPLE_DIR)):
        pdir = os.path.join(PEOPLE_DIR, person)
        if not os.path.isdir(pdir): 
            continue
        collected = 0
        for fn in os.listdir(pdir):
            if not fn.lower().endswith((".jpg",".jpeg",".png")): 
                continue
            img = fr.load_image_file(os.path.join(pdir, fn))
            locs = fr.face_locations(img, model="hog")  # быстрее на rpi
            if not locs: 
                continue
            enc = fr.face_encodings(img, known_face_locations=locs)
            if enc:
                encs.append(enc[0])
                names.append(person)
                collected += 1
        print(f"{person}: {collected} образцов")
    if not encs:
        print("Нет валидных фото.")
        sys.exit(1)
    np.savez(DB_PATH, names=np.array(names), encs=np.array(encs))
    print(f"OK: {DB_PATH} создан.")

if __name__ == "__main__":
    main()
