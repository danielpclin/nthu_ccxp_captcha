import os
import pathlib

import requests

from codes import codes, validate_codes, test_codes


def get_images(password, code, directory):
    os.makedirs(pathlib.Path(directory) / code, exist_ok=True)
    with requests.Session() as s:
        for j in range(1000):
            if j % 200 == 0:
                print(f"Getting image {j}....")
            with open(pathlib.Path(directory) / code / f"{j}.png", 'wb') as temp_file:
                r = s.get(f"https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/auth_img.php?pwdstr={password}")
                temp_file.write(r.content)


def main(password_codes, directory='dataset'):
    i = 1
    for password, code in password_codes:
        print(f"Getting images from https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/auth_img.php?pwdstr={password}, Code is {code}. ({i}/{len(password_codes)})")
        while True:
            try:
                get_images(password, code, directory)
                break
            except Exception:
                pass
        i += 1


if __name__ == "__main__":
    main(codes, 'dataset')
    main(validate_codes, 'validate')
    main(test_codes, 'test')
