import subprocess

SCRATCH_MODEL_NAME = "FT_Epoch_latest.pt"
MODEL_PATH = "/tmp/models/scratch_models"
scratch_model_file_download_url = "https://www.dropbox.com/s/5jencqq4h59fbtb/FT_Epoch_latest.pt"


def download_file(download_url, destination):
    command = "wget -nc -O " + destination + " " + download_url
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
