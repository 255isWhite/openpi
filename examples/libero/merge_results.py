import json
import pathlib
import argparse
import glob
import datetime
import os

def merge_results(results_dir: str, output_filename: str = "summary_FINAL_By_SuccessRate.json"):
    # 确保路径是绝对路径
    results_dir_path = pathlib.Path(results_dir).resolve() 
    search_path = str(results_dir_path / "results_*.json")
    json_files = glob.glob(search_path)
    
    if not json_files:
        print(f"在 {results_dir_path} 中未找到 'results_*.json' 文件。")
        return

    print(f"找到了 {len(json_files)} 个结果文件。正在合并并按成功率排序 (高到低)...")

    all_task_results = {}
    total_successes = 0
    total_episodes = 0
    all_metadata = []

    for f_path_str in json_files:
        f_path = pathlib.Path(f_path_str)
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
            
            # 1. 合并 "results_per_task"
            all_task_results.update(data.get("results_per_task", {}))
            
            # 2. 累加 "summary"
            summary = data.get("summary", {})
            total_successes += summary.get("total_successes", 0)
            total_episodes += summary.get("total_episodes", 0)
            
            # 3. 收集元数据 (简化)
            meta_data_to_store = data.get("metadata", {}).copy()
            if "args" in meta_data_to_store:
                meta_data_to_store["args_simplified"] = {
                    "port": meta_data_to_store["args"].get("port"),
                    "task_range_processed": meta_data_to_store.get("task_range_processed")
                }
                del meta_data_to_store["args"]
            all_metadata.append(meta_data_to_store)

            print(f"  ... 已合并 {f_path.name}")

        except Exception as e:
            print(f"合并文件 {f_path.name} 时出错: {e}")

    # --- 核心修改：按成功率排序（高到低） ---
    
    # 1. 将字典转换为 (task_name, task_details) 元组列表
    task_items = all_task_results.items()
    
    # 2. 使用 sorted() 函数，根据 task_details (item[1]) 中的 'success_rate' 进行排序
    # 设置 reverse=True 实现从高到低排序 (Descending)
    sorted_task_items = sorted(
        task_items, 
        key=lambda item: item[1]['success_rate'],
        reverse=True  # <-- 关键：设置为 True 实现从高到低排序
    )
    
    # 3. 将排序后的列表重新转换回字典（Python 3.7+ 字典保持插入顺序）
    sorted_task_results = dict(sorted_task_items)
    
    # --- 排序结束 ---

    # 4. 计算最终的总体成功率
    final_success_rate = 0.0
    if total_episodes > 0:
        final_success_rate = float(total_successes) / float(total_episodes)

    # 5. 构建最终的 JSON 对象
    final_summary = {
        "metadata": {
            "merge_timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "sorting_key": "success_rate (Descending)", # <-- 记录排序方式
            "source_files_count": len(json_files),
            "total_tasks_processed": len(sorted_task_results),
            "source_files_metadata_snippets": all_metadata,
        },
        "results_per_task": sorted_task_results, # <-- 使用排序后的结果
        "summary": {
            "total_success_rate": final_success_rate,
            "total_successes": total_successes,
            "total_episodes": total_episodes,
            "average_success_rate_per_task": final_success_rate,
        }
    }

    # 6. 写入最终文件
    output_path = results_dir_path / output_filename
    try:
        with open(output_path, "w") as f:
            json.dump(final_summary, f, indent=4)
        print(f"\n🎉 合并完成！最终总结已保存到: {output_path}")
        print(f"总成功率 (Total Success Rate): {final_success_rate * 100:.2f}% ({total_successes} / {total_episodes})")
    except Exception as e:
        print(f"写入最终总结时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LIBERO evaluation JSON results.")
    parser.add_argument(
        "results_dir", 
        type=str, 
        help="包含所有部分 JSON 结果文件 (results_*.json) 的目录。"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="summary_FINAL_By_SuccessRate.json", # <-- 默认输出文件名已更新
        help="最终输出的 JSON 文件名。"
    )
    args = parser.parse_args()
    
    merge_results(args.results_dir, args.out)