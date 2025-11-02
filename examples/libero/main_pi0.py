import collections
import dataclasses
import logging
import math
import pathlib
import json  # <-- 导入 json 模块
import datetime  # <-- 导入 datetime 模块

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8003
    resize_size: int = 224
    replan_steps: int = 10

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_90"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"data/pi0/libero_ac{replan_steps}/videos"  # Path to save videos
    results_out_path: str = f"data/pi0/libero_ac{replan_steps}/results"  # <-- 路径保持不变

    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # <-- 新增：并行化参数 -->
    #################################################################################################################
    task_id_start: int = 0
    """此进程要运行的起始任务 ID (包含)"""
    task_id_end: int = -1
    """此进程要运行的结束任务 ID (不包含)。-1 表示运行到任务套件的末尾。"""


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} (共 {num_tasks_in_suite} 个任务)")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # --- 修改：设置 JSON 结果文件 ---
    results_dir = pathlib.Path(args.results_out_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # <-- 新增：解析任务范围 -->
    start_idx = args.task_id_start
    end_idx = args.task_id_end
    if end_idx == -1 or end_idx > num_tasks_in_suite:
        end_idx = num_tasks_in_suite # 如果是 -1 或太大，则设置为最大任务数

    # <-- 修改：文件名包含任务范围，避免冲突 -->
    results_filename = f"results_{args.task_suite_name}_{start_idx}_to_{end_idx}_{timestamp}.json"
    results_filepath = results_dir / results_filename

    # 初始化 JSON 文件，写入元数据
    initial_data = {
        "metadata": {
            "args": dataclasses.asdict(args),
            "timestamp": timestamp,
            "task_suite": args.task_suite_name,
            "task_range_processed": [start_idx, end_idx], # <-- 新增：记录本文件处理的范围
        },
        "results_per_task": {},  # 存储每个任务的结果
        "summary": {},  # 存储最终总结
    }
    try:
        with open(results_filepath, "w") as f:
            json.dump(initial_data, f, indent=4)
        logging.info(f"结果将保存至: {results_filepath}") # <-- 修改了日志
    except Exception as e:
        logging.error(f"无法初始化 JSON 结果文件: {e}")
        return
    # --- JSON 设置结束 ---

    # ... (max_steps 的逻辑不变) ...
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    
    # <-- 修改：主循环只迭代指定的任务范围 -->
    logging.info(f"本进程将处理 Task ID 从 {start_idx} 到 {end_idx - 1}")
    task_ids_to_run = range(start_idx, end_idx)

    for task_id in tqdm.tqdm(task_ids_to_run):
        # ... (内部的 "Get task", "Get initial states" 逻辑不变) ...
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # ... (内部的 "Start episodes" 循环 (for episode_idx in ...) 完全不变) ...
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description} (ID: {task_id})") # <-- 增加了 ID
            
            # ... (try...except... 循环逻辑不变) ...
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    # ... (img, wrist_img, element, action_chunk 逻辑不变) ...
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)
                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            
            # ... (视频保存逻辑不变) ...
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # ... (任务结束后的 JSON 写入逻辑不变，现在它是安全的，因为写入的是独立文件) ...
        task_success_rate = 0.0
        if task_episodes > 0:
            task_success_rate = float(task_successes) / float(task_episodes)
        
        try:
            with open(results_filepath, "r") as f:
                results_data = json.load(f)
                
            # results_data["results_per_task"][task_description] = {
            #     "task_id": task_id,
            #     "success_rate": task_success_rate,
            #     "successes": task_successes,
            #     "episodes_tried": task_episodes,
            # }
            
            results_data["results_per_task"][f"{task_id}_{task_description}"] = {
                "task_id": task_id,
                "success_rate": task_success_rate,
                "successes": task_successes,
                "episodes_tried": task_episodes,
            }
            
            
            with open(results_filepath, "w") as f:
                json.dump(results_data, f, indent=4)
            logging.info(f"已更新任务 {task_id} ({task_description}) 的结果到 JSON。")
        except Exception as e:
            logging.error(f"写入任务 {task_id} 的 JSON 结果时失败: {e}")

        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # --- 最终总结写入 ---
    # (这部分逻辑不变，但现在它只总结此进程处理的任务范围)
    total_success_rate = 0.0
    if total_episodes > 0:
        total_success_rate = float(total_successes) / float(total_episodes)

    logging.info(f"此进程的总成功率: {total_success_rate}")
    logging.info(f"此进程的总 episodes: {total_episodes}")

    try:
        with open(results_filepath, "r") as f:
            results_data = json.load(f)
        
        results_data["summary"] = {
            "total_success_rate": total_success_rate,
            "total_successes": total_successes,
            "total_episodes": total_episodes,
            "note": f"Summary for tasks {start_idx} to {end_idx - 1}" # <-- 新增
        }
        with open(results_filepath, "w") as f:
            json.dump(results_data, f, indent=4)
        
        logging.info(f"此进程的最终总结已写入 {results_filepath}")

    except Exception as e:
        logging.error(f"写入最终 JSON 总结时失败: {e}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 将你的脚本文件名从 "your_script_name.py" 改为实际的文件名
    # tyro.cli(eval_libero)
    # 假设你的文件名是 eval_parallel.py
    tyro.cli(eval_libero)