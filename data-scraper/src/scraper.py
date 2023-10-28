from typing import Protocol


class Scraper(Protocol):
    def fetch_metadata(self):
        pass

    def prepare_metadata(self):
        pass

    def download_videos(self):
        pass

    def execute(self):
        pass
