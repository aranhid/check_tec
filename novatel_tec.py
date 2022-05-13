from datetime import datetime, timedelta
import os
import sys
import gzip
import shlex
import shutil
import subprocess
from ftplib import FTP_TLS, error_temp


def get_brdc_path(dir, date):
    nav_file_gz = download_nav_file(date.strftime("%y"), date.strftime("%j"), dir)
    nav_file = nav_file_gz[:-3]
    ungz(nav_file_gz, nav_file)
    return nav_file


def ungz(gzip_path, ungzip_path):
    if not os.path.exists(ungzip_path) or os.path.getsize(os.path.join(ungzip_path)) == 0:
        with gzip.open(gzip_path, 'r') as f_in, open(ungzip_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_nav_file(short_year, day, path):
    year = '20' + short_year

    email = 'aranhid@yandex.ru' #sys.argv[1]
    directory = f'gnss/data/daily/{year}/{day}/{short_year}p' #sys.argv[2]
    filename = f'BRDC00IGS_R_{year}{day}0000_01D_MN.rnx.gz' #sys.argv[3]
    file_path = os.path.join(path, filename)

    try:
        if not os.path.exists(file_path[:-3]):
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                ftps = FTP_TLS(host = 'gdc.cddis.eosdis.nasa.gov')
                ftps.login(user='anonymous', passwd=email)
                ftps.prot_p()
                ftps.cwd(directory)
                ftps.retrbinary("RETR " + filename, open(file_path, 'wb').write)
    
    except error_temp as e:
        print(e)
        download_nav_file(short_year, day, path)
    
    return file_path


file_prefix = "ISNO"
file_extensions = ['.21O', '.22O']
files_path = "R:\\NovAtel\\Converted"
brdc_path = "R:\\BRDC"
image_path = "R:\\NovAtel\\Images"

interval = 30
poli_degree_phase = 7
poli_degree_range = 15
std_mult_range = 3.5
std_mult_phase = 3.5
min_win_size = 20
max_win_size = 100
cutoff = 10

stard_date = datetime(2022, 3, 14).date()
end_date = datetime.now().date()
date = stard_date

date_list = {}
files = os.listdir(files_path)

while date != end_date:
    file_name = f'{file_prefix}_{date.strftime("%y%b%d")}'
    file_name = file_name.upper()
    prev_date = date - timedelta(days=1)
    prev_filename = f'{file_prefix}_{prev_date.strftime("%y%b%d")}_2359'
    prev_filename = prev_filename.upper()
    print(file_name)

    day_files = []

    for file in files:
        if file.startswith(file_name) or file.startswith(prev_filename):
            for ext in file_extensions:
                if file.endswith(ext):
                    day_files.append(os.path.join(files_path, file))

    if len(day_files):
        day_files = sorted(day_files)
        for f in list(day_files[-5:]):
            splited = f.split("\\")
            if splited[-1][13:15] == '23' or splited[-1][13:17] == '2259':
                day_files.remove(f)
        
        date_list[date.strftime("%y%b%d").upper()] = day_files

    date += timedelta(days=1)

for key in date_list.keys():
    day_files = date_list[key]

    day = datetime.strptime(key, "%y%b%d")
    brdc_file = get_brdc_path(brdc_path, day)
    day_dir = os.path.join(image_path, key)
    if not os.path.exists(day_dir):
        os.mkdir(day_dir)

    images_dir = os.path.join(day_dir, "tec")
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    my_arg = f'{sys.executable} check_tec.py --files {" ".join(day_files)} --interval {interval} --poli-degree-phase {poli_degree_phase} --poli-degree-range {poli_degree_range} --std-mult-range {std_mult_range} --std-mult-phase {std_mult_phase} --min-win-size {min_win_size} --max-win-size {max_win_size} --plot-dir {images_dir} --nav-file {brdc_file} --cutoff {cutoff}'
    
    cmd = shlex.split(my_arg, posix=False)
    process = subprocess.run(cmd)