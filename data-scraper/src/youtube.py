import os
import dotenv
import json
import tqdm
import requests
import yt_dlp as youtube_dl
import googleapiclient.discovery
import pandas as pd

import scraper


class YoutubeScraper(scraper.Scraper):
    def __init__(self, root_path: str, n_videos_per_keyword_per_lang: int = 3):
        dotenv.load_dotenv(os.path.join(root_path, ".env"))
        self.YT_DATA_API_KEYS = [
            os.getenv("YOUTUBE_DATA_API_KEY_1"),
            os.getenv("YOUTUBE_DATA_API_KEY_2"),
        ]
        self.n_videos_per_keyword_per_lang = n_videos_per_keyword_per_lang
        self.meta_dirpath = os.path.abspath(os.path.join(root_path, "data/meta"))
        self.video_dirpath = os.path.abspath(os.path.join(root_path, "data/videos"))
        self.keywords_fpath = os.path.join(self.meta_dirpath, "keywords_all.json")
        self.full_meta_fpath = os.path.join(self.meta_dirpath, "youtube_full_meta.json")
        self.fine_meta_fpath = os.path.join(self.meta_dirpath, "youtube_fine_meta.json")
        self.video_meta_fpath = os.path.join(self.video_dirpath, "shorts.json")
        self.yt_client = None
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
        Fetch metadata of each keyword search using YouTube Data API v3.

        What this does:
        - search n videos (default: 3) for each keyword in each language using YouTube Data API v3
        - produce <root>/data/meta/youtube_full_meta.json
        """
        # Get current progress
        youtube_meta, keywords_pending = self._get_metadata_progress()

        try:
            for cur_keyword in tqdm.tqdm(
                keywords_pending,
                desc="Fetching videos metadata of each keyword",
                unit="keyword",
            ):
                en_keyword = self.other2en_map[cur_keyword]["en"]
                cur_lang = self.other2en_map[cur_keyword]["lang"]

                search_items = self._search(cur_keyword)

                if en_keyword not in youtube_meta:
                    youtube_meta[en_keyword] = {}

                youtube_meta[en_keyword][cur_keyword] = {
                    "lang": cur_lang,
                    "items": search_items,
                }

        except Exception as e:
            raise e

        finally:
            if len(youtube_meta) > 0:
                with open(self.full_meta_fpath, "w", encoding="utf-8") as f:
                    json.dump(youtube_meta, f, indent=2, ensure_ascii=False)

    def prepare_metadata(self):
        """
        Run this function only if `fetch_metadata` is fully completed

        What this does:
        - remove non-shorts videos (conventional youtube video, playlist, etc.)
        - remove duplicate videos
        - keep only the first n videos (default: 3) for each keyword
        - produce <root>/data/meta/youtube_meta.json
        """
        youtube_meta, keywords_pending = self._get_metadata_progress()

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
        n_keywords = sum(len(keyword_dict) for keyword_dict in youtube_meta.values())
        with tqdm.tqdm(
            desc="Preparing metadata",
            unit="keyword",
            total=n_keywords,
        ) as pbar:
            # Skip if youtube_fine_meta.json already exists
            if os.path.exists(self.fine_meta_fpath):
                pbar.update(n_keywords)
                return

            for keyword_en in youtube_meta:
                for keyword_other, meta_other in youtube_meta[keyword_en].items():
                    for order_num, item in enumerate(meta_other["items"]):
                        try:
                            video_meta = {
                                "id": item["id"]["videoId"],
                                "title": str(item["snippet"]["title"]).strip(),
                                "description": str(
                                    item["snippet"]["description"]
                                ).strip(),
                                "url": f"https://www.youtube.com/shorts/{item['id']['videoId']}",
                                "lang": meta_other["lang"],
                                "searchKeyword": keyword_other,
                                "enKeyword": keyword_en,
                                "orderNum": order_num,
                            }

                            # This takes 95% of the time of this function
                            if not self._is_shorts(item["id"]["videoId"]):
                                raise Exception("Not a shorts video.")

                            df_fine_meta.loc[len(df_fine_meta)] = video_meta
                        except:
                            #  Handle missing key
                            continue

                    pbar.update(1)

        # remove duplicates (based on video id)
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
                df_fine_meta.to_dict("records"), f_out, indent=2, ensure_ascii=False
            )

    def download_videos(self):
        if not os.path.exists(self.video_meta_fpath):
            df_summary = pd.read_json(self.fine_meta_fpath, dtype={"id": str})
            df_summary["source"] = "youtube"
            with open(self.video_meta_fpath, "w", encoding="utf-8") as f_out:
                json.dump(
                    df_summary.to_dict("records"), f_out, indent=2, ensure_ascii=False
                )
        else:
            df_summary = pd.read_json(self.video_meta_fpath, dtype={"id": str})

            if "youtube" not in df_summary["source"].unique():
                df_new = pd.read_json(self.fine_meta_fpath, dtype={"id": str})
                df_new["source"] = "youtube"
                df_summary = pd.concat([df_summary, df_new], ignore_index=True)

                with open(self.video_meta_fpath, "w", encoding="utf-8") as f_out:
                    json.dump(
                        df_summary.to_dict("records"),
                        f_out,
                        indent=2,
                        ensure_ascii=False,
                    )

        df_summary = df_summary.loc[df_summary["source"] == "youtube"].drop(
            "source", axis=1
        )

        df_summary["downloaded"] = df_summary.apply(
            lambda row: os.path.exists(
                os.path.join(self.video_dirpath, row["enKeyword"], f"{row['id']}.mp4")
            ),
            axis=1,
        )

        to_download = df_summary.loc[~df_summary["downloaded"]]
        for keyword, sub_df in tqdm.tqdm(
            to_download.groupby("enKeyword"),
            desc="Downloading videos for each keyword",
            unit="keyword",
        ):
            ydl_opts = {
                "format": "mp4",
                "outtmpl": os.path.join(self.video_dirpath, keyword, "%(id)s.%(ext)s"),
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(sub_df["url"].tolist())

    def verify_download(self):
        # To check:
        # - check shorts.json and see if everything in the file is downloaded (cbeck the matching of videos and shorts.json) (integrity check)
        # - check stats:
        #   - how many videos per keyword

        # Integrity check: check if all videos in shorts.json are downloaded into the respective folder
        df_summary = pd.read_json(self.video_meta_fpath)
        df_summary = df_summary.loc[df_summary["source"] == "youtube"]
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
            api_secret_key = self.YT_DATA_API_KEYS.pop()
            self.yt_client = googleapiclient.discovery.build(
                serviceName="youtube",
                version="v3",
                developerKey=api_secret_key,
            )

        except:
            raise IndexError(
                "No more API keys with sufficient quota available. Please try again later (24h later) or add more API keys."
            )

    def _search(self, keyword: str):
        try:
            if not self.yt_client:
                self._init_client()

            query = f"#shorts {keyword}"
            req = self.yt_client.search().list(
                part="snippet", maxResults=self.n_videos_per_keyword_per_lang, q=query
            )
            res = req.execute()
            items = res.get("items", [])
            return items

        except googleapiclient.discovery.HttpError as e:
            self.yt_client = None
            return self._search(keyword)
        except IndexError as e:
            self.yt_client = None
            raise e
        except:
            self.yt_client = None
            raise Exception("Unknown error occurred while searching.")

    def _get_metadata_progress(self):
        keywords_all = set(
            keyword
            for keyword_dict in self.en2others_map.values()
            for keyword in keyword_dict.values()
        )
        try:
            with open(self.full_meta_fpath, "r", encoding="utf-8") as f:
                youtube_meta = json.load(f)

            keywords_done = set(
                keyword
                for keywords_dict in youtube_meta.values()
                for keyword in keywords_dict.keys()
            )
        except:
            youtube_meta = {}
            keywords_done = set()

        keywords_pending = keywords_all - keywords_done

        return youtube_meta, keywords_pending

    def _is_shorts(self, video_id):
        """
        Check if a video is a shorts video. Video will get redirect to a normal YouTube video if it is not a shorts video.
        Non-shorts video will get redirected to https://www.youtube.com/watch?v={video_id}; status code 303 will be returned.
        """
        url = f"https://www.youtube.com/shorts/{video_id}"
        res = requests.head(url)
        return res.status_code == 200
