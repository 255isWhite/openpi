#!/bin/zsh

echo "Starting parallel evaluation for libero_90..."

# 假设你的结果路径是 data/libero_ac5/results
RESULTS_DIR="data/pi0/libero_ac10/results"
mkdir -p $RESULTS_DIR

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 0-6 on port 8000
python examples/libero/main_pi0.py --args.task-id-start 0 --args.task-id-end 6 --args.port 8000 &
# 6-12 on port 8001
python examples/libero/main_pi0.py --args.task-id-start 6 --args.task-id-end 12 --args.port 8001 &
# 12-18 on port 8002
python examples/libero/main_pi0.py --args.task-id-start 12 --args.task-id-end 18 --args.port 8002 &
# 18-24 on port 8003
python examples/libero/main_pi0.py --args.task-id-start 18 --args.task-id-end 24 --args.port 8003 &
# 24-30 on port 8004
python examples/libero/main_pi0.py --args.task-id-start 24 --args.task-id-end 30 --args.port 8004 &
# 30-36 on port 8005
python examples/libero/main_pi0.py --args.task-id-start 30 --args.task-id-end 36 --args.port 8005 &
# 36-42 on port 8006
python examples/libero/main_pi0.py --args.task-id-start 36 --args.task-id-end 42 --args.port 8006 &
# 42-48 on port 8007
python examples/libero/main_pi0.py --args.task-id-start 42 --args.task-id-end 48 --args.port 8007 &
# 48-54 on port 8008
python examples/libero/main_pi0.py --args.task-id-start 48 --args.task-id-end 54 --args.port 8008 &
# 54-60 on port 8009
python examples/libero/main_pi0.py --args.task-id-start 54 --args.task-id-end 60 --args.port 8009 &
# 60-66 on port 8010
python examples/libero/main_pi0.py --args.task-id-start 60 --args.task-id-end 66 --args.port 8010 &
# 66-72 on port 8011
python examples/libero/main_pi0.py --args.task-id-start 66 --args.task-id-end 72 --args.port 8011 &
# 72-78 on port 8012
python examples/libero/main_pi0.py --args.task-id-start 72 --args.task-id-end 78 --args.port 8012 &
# 78-84 on port 8013
python examples/libero/main_pi0.py --args.task-id-start 78 --args.task-id-end 84 --args.port 8013 &
# 84-90 on port 8014
python examples/libero/main_pi0.py --args.task-id-start 84 --args.task-id-end 90 --args.port 8014 &


# 'wait' 命令会等待所有后台作业 (&) 完成
wait

echo "All evaluation processes finished."
echo "Results are in $RESULTS_DIR"
echo "Now, run the merge script to get the final summary."