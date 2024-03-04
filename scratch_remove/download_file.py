import subprocess


def download_file(download_url, destination):
    command = "wget -nc -O " + destination + " " + download_url
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
