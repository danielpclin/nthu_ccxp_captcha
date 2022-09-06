from io import BytesIO
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from PIL import Image


def main():
    result = []
    try:
        with requests.Session() as s:
            for _ in range(198):
                r = s.get("https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/")
                soup = BeautifulSoup(r.text, features="html.parser")
                image_url = soup.find('input', {'name': 'passwd2'}).parent.find('img')['src']
                pass_str = parse_qs(urlparse(image_url).query)['pwdstr'][0]
                r = s.get(f"https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/{image_url}")
                im = Image.open(BytesIO(r.content))
                plt.imshow(im)
                plt.show()
                code = input("Enter code: ")
                result.append((pass_str, code))
    except KeyboardInterrupt:
        pass
    finally:
        print(result)


if __name__ == "__main__":
    main()
