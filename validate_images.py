import pathlib
from PIL import Image
import shutil


def check_directory(directory):
    bad_dirs = set()
    for filename in pathlib.Path(directory).glob('*/*.png'):
        try:
            img = Image.open(filename)
            img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            bad_dirs.add(filename.parent)
    print(bad_dirs)
    for d in bad_dirs:
        shutil.rmtree(d)


def main():
    check_directory('dataset/ccxp/dataset')
    check_directory('dataset/ccxp/validate')
    # check_directory('dataset/oauth/dataset')
    # check_directory('dataset/oauth/validate')
    pass


if __name__ == "__main__":
    main()
