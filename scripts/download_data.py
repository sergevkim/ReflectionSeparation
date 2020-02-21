import os
import tarfile
import requests

from bs4 import BeautifulSoup
from tqdm.auto import tqdm


def get_outdoor_img(dir_name, img_id):
    template = "https://www.hel-looks/big-photos/{}.jpg"
    url = template.format(img_id)

    response = requests.get(url, allow_redirects=True)

    template = "data/{}/{}.jpg"
    reader = open(template.format(dir_name, img_id), 'wb')
    reader.write(response.content)


def move_file(filename, from_dir, to_dir):
    os.replace("{}/{}".format(from_dir, filename),
               "{}/{}".format(to_dir, filename))


def make_outdoor(dir_name, img_id="#20190810_13"):
    os.mkdir("data/{}".format(dir_name))

    template = "https://www.hel-looks.com/archive/{}"
    url = template.format(img_id)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', {'class': 'v'})

    for link in tqdm(links):
        img_id = link.get('href')[1:]
        get_outdoor_img()


def make_indoor(dir_name, tar_name="indoorCVPR_09.tar"):
    os.mkdir("data/{}".format(dir_name))

    tar = tarfile.open(tar_name, 'r:')
    tar.extractall("data/indoor_tmp")
    tar.close()

    for current_dir, dirs, files in os.walk("data/indoor_tmp"):
        for f in tqdm(files):
            move_file(f, current_dir, "data/{}".format(dir_name))
        if not (current_dir == "data/indoor_tmp" or current_dir == "data/indoor_tmp/Images"):
            os.rmdir(current_dir)


    os.rmdir("data/indoor_tmp/Images") # add it in a cycle later
    os.rmdir("data/indoor_tmp")


def do_all():
    os.mkdir("data")
    make_indoor("indoor_raw")
    make_outdoor("outdoor_raw")


do_all()

