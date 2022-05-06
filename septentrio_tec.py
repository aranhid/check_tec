import os
import sys
import gzip
import shlex
import shutil
import subprocess
from ftplib import FTP_TLS, error_temp


def ungz(gzip_path, ungzip_path):
    if not os.path.exists(ungzip_path) or os.path.getsize(os.path.join(ungzip_path)) == 0:
        with gzip.open(gzip_path, 'r') as f_in, open(ungzip_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_nav_file(short_year, day, path):
    year = '20' + short_year

    email = 'aranhid@yandex.ru' #sys.argv[1]
    directory = f'gnss/data/daily/{year}/{day}/{short_year}p' #sys.argv[2]
    filename = f'BRDC00IGS_R_{year}{day}0000_01D_MN.rnx.gz' #sys.argv[3]

    try:
        if not os.path.exists(os.path.join(path, filename)) or os.path.getsize(os.path.join(path, filename)) == 0:
            ftps = FTP_TLS(host = 'gdc.cddis.eosdis.nasa.gov')
            ftps.login(user='anonymous', passwd=email)
            ftps.prot_p()
            ftps.cwd(directory)
            ftps.retrbinary("RETR " + filename, open(os.path.join(path, filename), 'wb').write)
    
    except error_temp as e:
        print(e)
        download_nav_file(short_year, day, path)
    
    return filename


file_extensions = ['.21O', '.22O']
path = "R:\\Septentrio"
dirs = os.listdir(path)
interval = 30
poli_degree = 15
std_mult_range = 3
std_mult_phase = 3
cutoff = 10

for dir in dirs:
    new_path = os.path.join(path, dir)
    print(new_path)
    nav_file_gz = download_nav_file(dir[0:2], dir[2:], new_path)
    nav_file_gz = os.path.join(new_path, nav_file_gz)
    nav_file = nav_file_gz[:-3]
    ungz(nav_file_gz, nav_file)
    files = os.listdir(new_path)
    files_to_check = []
    for file in files:
        for ext in file_extensions:
            if file.endswith(ext):
                files_to_check.append(os.path.join(new_path, file))

    plot_dir = os.path.join(new_path, 'tec')
    my_arg = f'{sys.executable} check_tec.py --files {" ".join(files_to_check)} --interval {interval} --poli-degree {poli_degree} --std-mult-range {std_mult_range} --std-mult-phase {std_mult_phase} --plot-dir {plot_dir} --nav-file {nav_file} --cutoff {cutoff}'

    cmd = shlex.split(my_arg, posix=False)
    process = subprocess.run(cmd)