# $1 should be the desired gpu core to run on
python run_f_g_experiments_no_subnets_lingonly.py "$1"
python run_f_g_experiments_two_subnets.py "$1"
# python run_f_g_experiments_no_subnets.py "$1"