from google_images_download import google_images_download


def download_and_save_images(query_term, num_images, output_dir):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": query_term, "chromedriver": "/usr/bin/chromedriver", "limit": num_images, "output_directory": output_dir}
    response.download(arguments)


if __name__ == '__main__':
    common_cat_breeds = ['MaineCoon', 'PersianCat', 'AmericanShorthair', 'SiameseCat']
    for cat in common_cat_breeds:
        download_and_save_images(cat, 500, 'data/')
