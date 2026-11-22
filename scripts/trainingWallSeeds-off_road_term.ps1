# (Seed 11 is already trained and saved under results/off_road_term/seed_11)
python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition off_road_term --terminate_on_off_road --seed 22
python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition off_road_term --terminate_on_off_road --seed 33
python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition off_road_term --terminate_on_off_road --seed 44
python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition off_road_term --terminate_on_off_road --seed 55
