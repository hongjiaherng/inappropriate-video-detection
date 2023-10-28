import os
import json
import tqdm
import requests
import time
import pandas as pd
from TikTokApi import TikTokApi
from bs4 import BeautifulSoup
from urllib.request import urlopen

import scraper


class TikTokScraper(scraper.Scraper):
    def __init__(self, root_path: str, n_videos_per_keyword_per_lang: int = 3):
        self.n_videos_per_keyword_per_lang = n_videos_per_keyword_per_lang
        self.cookies_path = os.path.abspath(os.path.join(root_path, "cookies.json"))
        self.ssstik_params_path = os.path.abspath(
            os.path.join(root_path, "ssstik_params.json")
        )
        self.meta_dirpath = os.path.abspath(os.path.join(root_path, "data/meta"))
        self.video_dirpath = os.path.abspath(os.path.join(root_path, "data/videos"))
        self.keywords_fpath = os.path.join(self.meta_dirpath, "keywords_all.json")
        self.full_meta_fpath = os.path.join(self.meta_dirpath, "tiktok_full_meta.json")
        self.fine_meta_fpath = os.path.join(self.meta_dirpath, "tiktok_fine_meta.json")
        self.video_meta_fpath = os.path.join(self.video_dirpath, "shorts.json")
        self.cookies_kv = None
        self.ssstik_params = None
        self.en2others_map = None
        self.other2en_map = None

        os.makedirs(self.meta_dirpath, exist_ok=True)
        os.makedirs(self.video_dirpath, exist_ok=True)

        try:
            with open(self.keywords_fpath, "r", encoding="utf-8") as f:
                self.en2others_map = json.load(f)
                self.other2en_map = {
                    otherKeyword: {"lang": lang, "en": enKeyword}
                    for enKeyword, otherKeywords in self.en2others_map.items()
                    for lang, otherKeyword in otherKeywords.items()
                }
        except:
            raise FileNotFoundError(
                f"`{os.path.basename(self.keywords_fpath)}` is not found at `{self.meta_dirpath}`. Please run `python main.py --task generate_keywords` first to generate the file."
            )

    def fetch_metadata(self):
        """
        Fetch metadata of each keyword search using TikTok API.
        """
        tiktok_meta, keywords_pending = self._get_metadata_progress()

        try:
            for cur_keyword in tqdm.tqdm(
                keywords_pending,
                desc="Fetching videos metadata of each keyword",
                unit="keyword",
            ):
                en_keyword = self.other2en_map[cur_keyword]["en"]
                cur_lang = self.other2en_map[cur_keyword]["lang"]
                search_items = self._search(cur_keyword)

                if en_keyword not in tiktok_meta:
                    tiktok_meta[en_keyword] = {}

                tiktok_meta[en_keyword][cur_keyword] = {
                    "lang": cur_lang,
                    "items": search_items,
                }
        except Exception as e:
            raise e
        finally:
            if len(tiktok_meta) > 0:
                with open(self.full_meta_fpath, "w", encoding="utf-8") as f:
                    json.dump(tiktok_meta, f, indent=2, ensure_ascii=False)

    def prepare_metadata(self):
        tiktok_meta, keywords_pending = self._get_metadata_progress()

        if len(keywords_pending) > 0:
            raise Exception(
                f"There are still {len(keywords_pending)} pending keywords to search for metadata. Please ensure that `fetch_metadata` is fully completed before running this function."
            )

        df_fine_meta = pd.DataFrame(
            columns=[
                "id",
                "title",
                "description",
                "url",
                "lang",
                "searchKeyword",
                "enKeyword",
                "orderNum",
            ]
        )
        n_keywords = sum(len(keyword_dict) for keyword_dict in tiktok_meta.values())
        with tqdm.tqdm(
            desc="Preparing metadata",
            unit="keyword",
            total=n_keywords,
        ) as pbar:
            # Skip if tiktok_fine_meta.json already exists
            if os.path.exists(self.fine_meta_fpath):
                pbar.update(n_keywords)
                return

            for keyword_en in tiktok_meta:
                for keyword_other, meta_other in tiktok_meta[keyword_en].items():
                    for order_num, item in enumerate(meta_other["items"]):
                        try:
                            video_meta = {
                                "id": item["id"],
                                "title": "",
                                "description": str(item["desc"]).strip(),
                                "url": f"https://www.tiktok.com/@{item['author']['uniqueId']}/video/{item['id']}",
                                "lang": meta_other["lang"],
                                "searchKeyword": keyword_other,
                                "enKeyword": keyword_en,
                                "orderNum": order_num,
                            }

                            # This takes 95% of the time of this function
                            if not self._shorter_than_60s(item["video"]["duration"]):
                                raise Exception("Longer than 60s.")

                            df_fine_meta.loc[len(df_fine_meta)] = video_meta
                        except:
                            #  Handle missing key
                            continue

                    pbar.update(1)

        # remove duplicates (based on id)
        df_fine_meta = df_fine_meta.drop_duplicates(subset=["id"], keep="first")

        # keep only the first n videos (default: 3) for each keyword
        df_fine_meta = (
            df_fine_meta.sort_values(by=["searchKeyword", "orderNum"])
            .groupby("searchKeyword")
            .head(self.n_videos_per_keyword_per_lang)
            .drop(columns=["orderNum"])
        )

        with open(self.fine_meta_fpath, "w", encoding="utf-8") as f_out:
            json.dump(
                df_fine_meta.to_dict(orient="records"),
                f_out,
                indent=2,
                ensure_ascii=False,
            )

    def download_videos(self):
        if not os.path.exists(self.video_meta_fpath):
            df_summary = pd.read_json(self.fine_meta_fpath, dtype={"id": str})
            df_summary["source"] = "tiktok"
            with open(self.video_meta_fpath, "w", encoding="utf-8") as f_out:
                json.dump(
                    df_summary.to_dict("records"), f_out, indent=2, ensure_ascii=False
                )
        else:
            df_summary = pd.read_json(self.video_meta_fpath, dtype={"id": str})

            if "tiktok" not in df_summary["source"].unique():
                df_new = pd.read_json(self.fine_meta_fpath, dtype={"id": str})
                df_new["source"] = "tiktok"
                df_summary = pd.concat([df_summary, df_new], ignore_index=True)

                with open(self.video_meta_fpath, "w", encoding="utf-8") as f_out:
                    json.dump(
                        df_summary.to_dict("records"),
                        f_out,
                        indent=2,
                        ensure_ascii=False,
                    )

        df_summary = df_summary.loc[df_summary["source"] == "tiktok"].drop(
            "source", axis=1
        )

        df_summary["downloaded"] = df_summary.apply(
            lambda row: os.path.exists(
                os.path.join(self.video_dirpath, row["enKeyword"], f"{row['id']}.mp4")
            ),
            axis=1,
        )

        to_download = df_summary.loc[~df_summary["downloaded"]]
        for id, keyword, url in tqdm.tqdm(
            to_download[["id", "enKeyword", "url"]].values,
            desc="Downloading videos",
            unit="video",
        ):
            self._download_video(url, keyword, id)
            time.sleep(5)  # sleep for 5s to avoid rate limit

    def verify_download(self):
        # To check:
        # - check shorts.json and see if everything in the file is downloaded (cbeck the matching of videos and shorts.json) (integrity check)
        # - check stats:
        #   - how many videos per keyword

        # Integrity check: check if all videos in shorts.json are downloaded into the respective folder
        df_summary = pd.read_json(self.video_meta_fpath)
        df_summary = df_summary.loc[df_summary["source"] == "tiktok"]
        df_summary["downloaded"] = df_summary.apply(
            lambda row: os.path.exists(
                os.path.join(self.video_dirpath, row["enKeyword"], f"{row['id']}.mp4")
            ),
            axis=1,
        )
        n_missing = len(df_summary) - df_summary["downloaded"].sum()

        print(
            "Integrity check:",
            f"All {len(df_summary)} videos are downloaded successfully."
            if n_missing == 0
            else f"{n_missing} out of {len(df_summary)} videos are not downloaded properly.",
        )

        # Stats for all downloaded videos
        df_by_keyword = (
            df_summary["enKeyword"].loc[df_summary["downloaded"]].value_counts()
        )
        df_by_keyword["TOTAL"] = df_by_keyword.sum()
        df_by_lang = df_summary["lang"].loc[df_summary["downloaded"]].value_counts()
        df_by_lang["TOTAL"] = df_by_lang.sum()

        print("\n================")
        print("Download Summary")
        print("================")
        print(df_by_keyword.to_string(), end="\n\n")
        print(df_by_lang.to_string())

        return df_by_keyword, df_by_lang

    def execute(self):
        self.fetch_metadata()
        self.prepare_metadata()
        self.download_videos()
        self.verify_download()

    def _init_client(self):
        try:
            if not self.cookies_kv:
                # Load cookies
                with open(self.cookies_path) as f:
                    cookies = json.load(f)
                self.cookies_kv = {
                    cookie["name"]: cookie["value"] for cookie in cookies
                }

            # Initialize TikTok API client
            tt_client = TikTokApi()
            tt_client._get_cookies = lambda **kwargs: self.cookies_kv
            return tt_client

        except FileNotFoundError:
            raise FileNotFoundError(
                f"`{os.path.basename(self.cookies_path)}` is not found at `{os.path.dirname(self.cookies_path)}`. Please ensure you download your own cookies on TikTok and put in the directory."
            )
        except:
            raise Exception("Error initializing TikTok API client.")

    def _get_metadata_progress(self):
        keywords_all = set(
            keyword
            for keyword_dict in self.en2others_map.values()
            for keyword in keyword_dict.values()
        )
        try:
            with open(self.full_meta_fpath, "r", encoding="utf-8") as f:
                tiktok_meta = json.load(f)

            keywords_done = set(
                keyword
                for keywords_dict in tiktok_meta.values()
                for keyword in keywords_dict.keys()
            )
        except:
            tiktok_meta = {}
            keywords_done = set()

        keywords_pending = keywords_all - keywords_done

        return tiktok_meta, keywords_pending

    def _search(self, keyword: str):
        try:
            # TikTokApi needs to be re-initialized for each search otherwise won't work.
            tt_client = self._init_client()

            with tt_client as api:
                items_iter = api.search.videos(
                    search_term=keyword, count=self.n_videos_per_keyword_per_lang
                )
                items = [item.as_dict for item in items_iter]

            return items

        except Exception as e:
            raise e

    def _shorter_than_60s(self, duration: int):
        return duration < 60

    def _download_video(self, url: str, en_keyword: str, id: str):
        """
        Download the video: https://ssstik.io/en
        Generate the download script: https://curlconverter.com/
        """
        try:
            if not self.ssstik_params:
                with open(self.ssstik_params_path, "r", encoding="utf-8") as f:
                    raw_ssstik_params = json.load(f)
                    self.ssstik_params = {
                        "cookies": raw_ssstik_params["cookies"],
                        "headers": raw_ssstik_params["headers"],
                        "params": raw_ssstik_params["queries"],
                        "data": raw_ssstik_params["data"],
                    }

                    # remove cookie from headers
                    self.ssstik_params["headers"].pop("cookie", None)
                    self.ssstik_params["data"]["id"] = ""

            cookies = self.ssstik_params["cookies"]
            headers = self.ssstik_params["headers"]
            params = self.ssstik_params["params"]
            data = self.ssstik_params["data"]
            data["id"] = url  # update id with the current url

            response = requests.post(
                "https://ssstik.io/abc",
                params=params,
                cookies=cookies,
                headers=headers,
                data=data,
            )
            download_soup = BeautifulSoup(response.text, "html.parser")
            download_url = download_soup.find("a")["href"]
            mp4File = urlopen(download_url)

            output_fp = os.path.join(self.video_dirpath, en_keyword, f"{id}.mp4")
            os.makedirs(os.path.dirname(output_fp), exist_ok=True)
            with open(output_fp, "wb") as output:
                while True:
                    data = mp4File.read(4096)
                    if data:
                        output.write(data)
                    else:
                        break

            return True

        except TypeError as e:
            print(
                f"Failed to download `{id}` from `{url}` under keyword `{en_keyword}` due to rate limit! Sleeping for 5s then retry..."
            )
            time.sleep(5)  # sleep for 5s then retry
            status = self._download_video(url, en_keyword, id)
            if status:
                print(
                    f"Successfully downloaded `{id}` from `{url}` under keyword `{en_keyword}`!"
                )

        except Exception as e:
            raise Exception(
                f"Error downloading video from `{url}` due to unexpected error: {e}"
            )
