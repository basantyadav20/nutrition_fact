# coding=utf-8

import json
import os

import datasets

from PIL import Image
import numpy as np

logger = datasets.logging.get_logger(__name__)



def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


class nutfactConfig(datasets.BuilderConfig):
    """BuilderConfig for nutfact"""

    def __init__(self, **kwargs):
        """BuilderConfig for nutfact.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(nutfactConfig, self).__init__(**kwargs)


class nutfact(datasets.GeneratorBasedBuilder):
    """nutfact dataset."""

    BUILDER_CONFIGS = [
        nutfactConfig(name="nutfact", version=datasets.Version("1.0.0"), description="nutfact dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADING", "I-HEADING","B-SERVE_SIZE", "I-SERVE_SIZE","B-%_DAILY","I-%_DAILY","B-ENERGY","I-ENERGY", "B-NUTRIENT", "I-NUTRIENT",
                                   "B-MICRONUTRIENT", "I-MICRONUTRIENT", "B-VALUE", "I-VALUE"]
                        )
                    ),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://drive.google.com/uc?export=download&id=1YRTklb_IUFJ2caeaxcu6Hvp3TvmFYPmQ",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract("https://drive.google.com/uc?export=download&id=1YRTklb_IUFJ2caeaxcu6Hvp3TvmFYPmQ")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            words = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["Nutrition_table"]:
                words_example, label = item["words"], item["label"]
                words_example = [w for w in words_example if w["text"].strip() != ""]
                if len(words_example) == 0:
                    continue
                if label == "gen_info":
                    for w in words_example:
                        words.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(w["box"], size))
                else:
                    words.append(words_example[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(words_example[0]["box"], size))
                    for w in words_example[1:]:
                        words.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(w["box"], size))

            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path}