# Launch examples
## Check availability

`python check_availability.py --files "C:\abak3080.21o" --interval 30 --window-size 3600 --max-gap-num 5 --plot-show --plot-file "C:\abak3080.png" --nav "C:\BRDC00IGS_R_20213080000_01D_MN.rnx" --cutoff 10`

## TEC Noise Check

`python tec_noise_checker.py --files "C:\arkh3080.21o" --interval 30 --poli-degree 10 --std-mult 2 --nav "C:\BRDC00IGS_R_20213080000_01D_MN.rnx" --cutoff 10 --plot-dir "C:\dir_for_plots"`