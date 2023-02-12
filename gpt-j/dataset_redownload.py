from datasets import load_dataset, load_metric, load_from_disk, Dataset, concatenate_datasets

data_server = "https://api.plan4.house/static/"
ds = load_dataset("json", data_files={'train': f"{data_server}aihub_book_qna.zip"}, download_mode='force_redownload', ignore_verifications=True)
ds = load_dataset("json", data_files={'train': f"{data_server}aihub_news_qna.zip"}, download_mode='force_redownload', ignore_verifications=True)
