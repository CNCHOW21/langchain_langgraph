import csv

import datasets


_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1KhI9WDqbx64nUTQT_FsUqU6mu6AkEc3m&export=download"
_DEV_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1C_-pJhh38ItfyF2-u2tTmRx2OI5zr-I9&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1v5KRU1pIYUf13ebTLyfSQFRbiEO-DM8U&export=download"



class Weibo(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=None,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['like', 'disgust', 'happiness', 'sadness', 'anger', 'surprise', 'fear', 'none']),
                }
            ),
            homepage=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        dev_path = dl_manager.download_and_extract(_DEV_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                if id_ == 0:
                    continue
                label, text = row
                label = int(label)
                yield id_, {"text": text, "label": label}