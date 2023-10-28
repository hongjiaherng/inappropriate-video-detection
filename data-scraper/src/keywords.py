import os
import pandas as pd
import tqdm
from googletrans import Translator

LANGS = ["en", "es", "pt", "hi", "ko", "zh-cn", "ms"]


def generate_keywords(root_path: str):
    """
    Generates keywords_all.json from keywords_en.json

    List of languages:
    - en: English
    - es: Spanish
    - pt: Portuguese
    - hi: Hindi
    - ko: Korean
    - zh-cn: Simplified Chinese
    - ms: Malay
    """

    # Load keywords_en.json into a list
    df_keywords = (
        pd.read_json(
            os.path.join(root_path, "data/meta/keywords_en.json"), orient="values"
        )
        .rename(columns={0: "en"})
        .set_index("en", drop=False)
        .sort_index()
    )

    translator = Translator()
    for lang in tqdm.tqdm(
        LANGS, desc="Translating 19 keywords to other languages", unit="language"
    ):
        if lang == "en":
            continue

        df_keywords[lang] = df_keywords["en"].apply(
            lambda x: translator.translate(x, dest=lang).text
        )

    # Save to keywords_all.json
    df_keywords.to_json(
        os.path.join(root_path, "data/meta/keywords_all.json"),
        orient="index",
        indent=2,
        force_ascii=False,
    )
