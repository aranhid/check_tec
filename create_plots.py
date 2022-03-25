import os
import shlex
import subprocess
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='path to directory')
    parser.add_argument('--extension', type=str, help='extension like .21o')
    parser.add_argument('--interval', type=float, help='interval of RINEX file, in seconds')
    parser.add_argument('--window-size', type=float, help='size for the rolling window to check gaps, in seconds')
    parser.add_argument('--max-gap-num', type=int, help='maximum number of gaps in the rolling window')
    parser.add_argument('--nav-file', type=str, help='path to NAV file')
    parser.add_argument('--year', type=int, help='Year like 2022')
    parser.add_argument('--doy', type=int, help='Day of year like 103')
    parser.add_argument('--cutoff', type=float, help='Cutoff for elevation')
    args = parser.parse_args()
    directory = args.directory
    file_extension = args.extension
    for r, d, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                path = os.path.join(directory, file)
                image_file = path.split('.')[0] + '.png'
                print(path, image_file)
                my_arg = f'{sys.executable} check_availability.py --files {path} --interval {args.interval} --window-size {args.window_size} --max-gap-num {args.max_gap_num} --plot-file {image_file} --nav-file {args.nav_file} --year {args.year} --doy {args.doy} --cutoff {args.cutoff}'
                # print(my_arg)
                cmd = shlex.split(my_arg, posix=False)
                # print(cmd)
                process = subprocess.run(cmd)
