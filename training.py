import numpy as np
import cv2
from pathlib import Path

test_data_path = Path("Faces_FA_FB/fa_H")

for file in test_data_path.iterdir():
    if file.is_file():
        print(file.name)

