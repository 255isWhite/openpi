#!/bin/zsh

echo "Starting server for libero_90..."


source .venv/bin/activate
conda activate pi05

export CUDA_VISIBLE_DEVICES=0
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8000 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8001 &

export CUDA_VISIBLE_DEVICES=1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8002 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8003 &

export CUDA_VISIBLE_DEVICES=2
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8004 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8005 &

export CUDA_VISIBLE_DEVICES=3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8006 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8007 &

export CUDA_VISIBLE_DEVICES=4
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8008 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8009 &

export CUDA_VISIBLE_DEVICES=5
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8010 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8011 &

export CUDA_VISIBLE_DEVICES=6
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8012 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8013 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run scripts/serve_policy.py --env LIBERO --port 8014 &

# 'wait' 命令会等待所有后台作业 (&) 完成
wait

echo "All server processes finished."