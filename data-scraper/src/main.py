import os
import argparse

import keywords
import youtube
import tiktok

ROOT_PATH = os.path.abspath(os.path.join(__file__, "../.."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["generate_keywords", "scrape_youtube", "scrape_tiktok"],
        default="generate_keywords_all",
    )

    args = parser.parse_args()

    if args.task == "generate_keywords":
        keywords.generate_keywords(root_path=ROOT_PATH)

    elif args.task == "scrape_youtube":
        scraper = youtube.YoutubeScraper(
            root_path=ROOT_PATH, n_videos_per_keyword_per_lang=3
        )
        scraper.execute()

    elif args.task == "scrape_tiktok":
        scraper = tiktok.TikTokScraper(
            root_path=ROOT_PATH, n_videos_per_keyword_per_lang=3
        )
        scraper.execute()

    else:
        raise ValueError(f"Invalid task: {args.task}")
