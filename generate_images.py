import os
import pathlib

import requests

from codes import ccxp_codes, ccxp_oauth_codes, ccxp_validate_codes, ccxp_oauth_validate_codes


def get_ccxp_images_with_code(password, code, directory):
    os.makedirs(pathlib.Path(directory) / code, exist_ok=True)
    with requests.Session() as s:
        for j in range(1000):
            if j % 200 == 0:
                print(f"Getting image {j}....")
            with open(pathlib.Path(directory) / code / f"{j}.png", 'wb') as temp_file:
                r = s.get(f"https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/auth_img.php?pwdstr={password}")
                temp_file.write(r.content)


def get_ccxp_images(password_codes, directory='dataset'):
    i = 1
    for password, code in password_codes:
        print(f"Getting images from https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/auth_img.php?pwdstr={password}, Code is {code}. ({i}/{len(password_codes)})")
        while True:
            try:
                get_ccxp_images_with_code(password, code, directory)
                break
            except Exception:
                pass
        i += 1


def get_ccxp_oauth_images_with_code(id_str, code, directory):
    os.makedirs(pathlib.Path(directory) / code, exist_ok=True)
    with requests.Session() as s:
        s.cookies.set("PHPSESSID", "p7vn499ldjl80mq59geq4t0khl", domain="oauth.ccxp.nthu.edu.tw")
        for j in range(1000):
            if j % 200 == 0:
                print(f"Getting image {j}....")
            with open(pathlib.Path(directory) / code / f"{j}.png", 'wb') as temp_file:
                r = s.get(f"https://oauth.ccxp.nthu.edu.tw/v1.1/captchaimg.php?id={id_str}")
                temp_file.write(r.content)


def get_ccxp_oauth_images(password_codes, directory='dataset'):
    i = 1
    for id_str, code in password_codes:
        print(f"Getting images from {id_str}, Code is {code}. ({i}/{len(password_codes)})")
        while True:
            try:
                get_ccxp_oauth_images_with_code(id_str, code, directory)
                break
            except Exception:
                pass
        i += 1


def main():
    get_ccxp_images(ccxp_codes, 'dataset/ccxp/dataset')
    get_ccxp_images(ccxp_validate_codes, 'dataset/ccxp/validate')
    # get_ccxp_oauth_images(ccxp_oauth_codes, 'dataset/oauth/dataset')
    # get_ccxp_oauth_images(ccxp_oauth_validate_codes, 'dataset/oauth/validate')


if __name__ == "__main__":
    main()
