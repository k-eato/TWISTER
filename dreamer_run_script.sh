export CARLA_ROOT="/home/keaton30/geigh/TWISTER/carla-0-9-15"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
bash train.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane --dreamerv3.run.steps 200000
