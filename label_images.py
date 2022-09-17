from io import BytesIO
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from PIL import Image


def label_ccxp(total=198):
    result = []
    try:
        with requests.Session() as s:
            for i in range(total):
                r = s.get("https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/")
                soup = BeautifulSoup(r.text, features="html.parser")
                image_url = soup.find('input', {'name': 'passwd2'}).parent.find('img')['src']
                pass_str = parse_qs(urlparse(image_url).query)['pwdstr'][0]
                r = s.get(f"https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/{image_url}")
                im = Image.open(BytesIO(r.content))
                plt.imshow(im)
                plt.show()
                code = input(f"Enter code ({i+1}/{total}): ")
                result.append((pass_str, code))
    except KeyboardInterrupt:
        pass
    finally:
        print(result)


def label_ccxp_oauth(total=50):
    result = []
    try:
        with requests.Session() as s:
            s.cookies.set("PHPSESSID", "p7vn499ldjl80mq59geq4t0khl", domain="oauth.ccxp.nthu.edu.tw")
            for i in range(total):
                r = s.get("https://oauth.ccxp.nthu.edu.tw/v1.1/authorize.php?client_id=elearn&response_type=code")
                soup = BeautifulSoup(r.text, features="html.parser")
                image_url = soup.find('img', {'id': 'captcha_image'})['src']
                id_str = parse_qs(urlparse(image_url).query)['id'][0]
                r = s.get(f"https://oauth.ccxp.nthu.edu.tw/v1.1/captchaimg.php?id={id_str}")
                im = Image.open(BytesIO(r.content))
                plt.imshow(im)
                plt.show()
                code = input(f"Enter code ({i+1}/{total}): ")
                result.append((id_str, code))
    except KeyboardInterrupt:
        pass
    finally:
        print(result)


def main():
    # label_ccxp()
    label_ccxp_oauth()


if __name__ == "__main__":
    main()
