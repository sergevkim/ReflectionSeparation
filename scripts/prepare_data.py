import requests
import tarfile

from bs4 import BeautifulSoup
from pathlib import Path
from tqdm.auto import tqdm

from utils.args import prepare_data_parse_args


def prepare_subject_images_dir(where, dir_name, tar_name):
    target_path = Path("{}/{}".format(where, dir_name))    # ./data/subject_images
    path_tmp = Path("{}/tmp".format(where))                # ./data/tmp
    path_tmp_images = Path("{}/tmp/Images".format(where))  # ./data/tmp/Images
    target_path.mkdir(parents=True, exist_ok=True)
    path_tmp.mkdir(parents=True, exist_ok=True)

    tar = tarfile.open(tar_name, 'r:')
    tar.extractall(path_tmp)
    tar.close()

    for directory in tqdm(path_tmp_images.glob("*")):
        directory_path = Path(directory)

        for filename in directory_path.glob("*"):
            filename_path = Path(filename)
            filename_path_new = "{}/{}".format(target_path, filename_path.name)
            filename_path.replace(filename_path_new)

        directory_path.rmdir()

    path_tmp_images.rmdir()
    path_tmp.rmdir()


def prepare_astigma_images_dir(where, dir_name, url):
    target_path = Path("{}/{}".format(where, dir_name))  # ./data/astigma_images
    target_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', {'class': 'v'})

    for link in tqdm(links):
        image_id = link.get('href')[1:]
        image_filename = "{}/{}/{}.jpg".format(where, dir_name, image_id)
        image_file = open(image_filename, 'wb')

        image_url = 'https://www.hel-looks.com/big-photos/{}.jpg'.format(image_id)
        image_response = requests.get(url, allow_redirects=True)
        image_file.write(image_response.content)


def main():
    args = prepare_data_parse_args()

    data_path = Path(args.where)
    data_path.mkdir(parents=True, exist_ok=True)

    prepare_subject_images_dir(where=args.where,
                               dir_name=args.subject_images_dir_name,
                               tar_name=args.tar_name)
    prepare_astigma_images_dir(where=args.where,
                               dir_name=args.astigma_images_dir_name,
                               url=args.url)

    return 0


if __name__ == "__main__":
    main()

