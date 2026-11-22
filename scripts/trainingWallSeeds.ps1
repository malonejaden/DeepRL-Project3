# Sequential training across target seeds 22, 33, 44, 55 in PowerShell
python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition baseline --seed 22
python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition baseline --seed 33
python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition baseline --seed 44
python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition baseline --seed 55
