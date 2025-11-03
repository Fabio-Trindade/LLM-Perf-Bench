import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Markdown report from model CSV results.")
    parser.add_argument("--csv_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Path to save results (directory)")
    parser.add_argument("--experiments", nargs="+", required=True, help="List of experiments (key) to compare")
    parser.add_argument("--experiment_is_group", action="store_true")
    parser.add_argument("--type", choices=["single_prompt", "load"], required=True)
    parser.add_argument("--e2e_goal", type=float, default=None, help="Goal threshold for End-to-End latency in seconds")
    return parser.parse_args()

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_tbt(val):
        if pd.isna(val):
            return []
        if isinstance(val, str):
            try:
                return [float(x) for x in json.loads(val)]
            except Exception:
                return []
        return val

    df["tbt_times"] = df["tbt_times"].apply(parse_tbt)
    return df

def aggregate_metrics(df):
    grouped = []
    
    for (key, prompt_size, model_alias), group in df.groupby(["experiment_key", "prompt_size", "model_alias"]):
        agg = {}
        agg["experiment_key"] = key
        agg["prompt_size"] = prompt_size
        agg["model_alias"] = model_alias
        agg["num_decoded_tokens"] = group["num_decoded_tokens"].mean()
        agg["e2e_latency"] = group["e2e_latency"].mean()
        agg["ttft"] = group["ttft"].mean()
        agg["total_decode_time"] = group["total_decode_time"].mean()
        agg["req_per_sec"] = group["req_per_sec"].mean() if "req_per_sec" in group else np.nan

        tbt_lists = group["tbt_times"].tolist()

        if len(tbt_lists) > 0:
            tbt_array = np.array(tbt_lists)
            mean_tbt_curve = np.nanmean(tbt_array, axis=0)

            median_tbt_value = np.nanmedian(mean_tbt_curve)
            p99_tbt_value = np.nanpercentile(mean_tbt_curve, 99)

            agg["tbt_times"] = mean_tbt_curve.tolist()
            agg["tbt_median"] = median_tbt_value
            agg["tbt_p99"] = p99_tbt_value
        else:
            agg["tbt_times"] = []
            agg["tbt_median"] = np.nan
            agg["tbt_p99"] = np.nan

        prompt_size_mean = group["prompt_size"].mean()
        decode_size_mean = group["num_decoded_tokens"].mean()
        e2e_latency_mean = group["e2e_latency"].mean()
        ttft_mean = group["ttft"].mean()
        total_decode_time_mean = group["total_decode_time"].mean()

        agg["total_throughput"] = (
            (prompt_size_mean + decode_size_mean) / e2e_latency_mean
            if e2e_latency_mean > 0 else 0.0
        )
        
        agg["prefill_throughput"] = (
            prompt_size_mean / ttft_mean
            if ttft_mean > 0 else 0.0
        )

        agg["decode_throughput"] = (
            decode_size_mean / total_decode_time_mean
            if total_decode_time_mean > 0 else 0.0
        )

        grouped.append(agg)

    agg_df = pd.DataFrame(grouped)
    return agg_df

def generate_single_prompt_plots(df, output_dir, experiments):
    metrics = [
        ("ttft", "Time To First Token (s)"),
        ("tbt_median", "Median Time-between-tokens Latency (s)"),
        ("tbt_p99", "P99 Time-between-tokens Latency (s)"),
        ("total_throughput", "Total Throughput (tokens/s)"),
        ("e2e_latency", "End-to-End Latency (s)"),
        ("prefill_throughput", "Prefill Throughput (tokens/s)"),
        ("decode_throughput", "Decode Throughput (tokens/s)")
    ]

    plots = []
    for metric, title in metrics:
        plt.figure(figsize=(8,5))
        for exp in experiments:
            subset = df[df["experiment_key"] == exp]
            for model in subset["model_alias"].unique():
                model_data = subset[subset["model_alias"] == model]
                x = model_data["prompt_size"]
                y = model_data[metric]
                plt.plot(x, y, label=f"{exp} - {model}", marker='o')

        plt.xlabel("Prompt Size (tokens)")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        filename = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plots.append(filename)

    return plots

def compute_pareto_frontier(points):
    if len(points) == 0:
        return []
    
    points = np.array(points)
    sorted_indices = np.argsort(points[:, 0])
    pareto_indices = []
    min_y = float('inf')
    
    for idx in sorted_indices:
        if points[idx, 1] < min_y:
            min_y = points[idx, 1]
            pareto_indices.append(idx)
    
    return pareto_indices



def generate_latency_vs_req_per_sec_plots(df, output_dir, experiments):

    plots = []
    
    columns = ["e2e_latency", "e2e_latency", "ttft", "ttft", "tbt_times", "tbt_times"]
    percentiles = [50, 99, 50, 99, 50, 99]
    
    titles = [
        "End-to-End Latency Median (s)",
        "End-to-End Latency P99 (s)",
        "Time To First Token Median (s)",
        "Time To First Token P99 (s)",
        "Time Between Tokens Median (s)",
        "Time Between Tokens P99 (s)"
    ]
    
    y_labels = [
        "End-to-End Latency Median (s)",
        "End-to-End Latency P99 (s)",
        "Time To First Token Median (s)",
        "Time To First Token P99 (s)",
        "Time Between Tokens Median (s)",
        "Time Between Tokens P99 (s)"
    ]
    
    image_names = [
        "e2e_median_latency_vs_req_per_sec.png",
        "e2e_p99_latency_vs_req_per_sec.png",
        "ttft_median_vs_req_per_sec.png",
        "ttft_p99_vs_req_per_sec.png",
        "tbt_median_vs_req_per_sec.png",
        "tbt_p99_vs_req_per_sec.png"
    ]
    
    y_is_list_flags = [False, False, False, False, True, True]
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    for i, (column, percentile, title, y_label, image_name, is_list) in enumerate(zip(
        columns, percentiles, titles, y_labels, image_names, y_is_list_flags
    )):
        plt.figure(figsize=(10, 6))
        
        for exp_idx, exp in enumerate(experiments):
            subset = df[df["experiment_key"] == exp]
            if subset.empty:
                print(f"Warning: No data found for experiment '{exp}'")
                continue
            
            models = subset["model_alias"].unique()
            if len(models) != 1:
                print(f"Warning: Expected exactly one model for experiment {exp}, but found {models}")
                model = models[0]
            else:
                model = models[0]
            
            x_vals = []
            y_vals = []
            x_label = ""
            for req_per_sec in sorted(subset["req_per_sec"].unique()):
                group = subset[subset["req_per_sec"] == req_per_sec]
                if group.empty:
                    continue
                min_time = group["initial_time"].min()
                max_time = group["final_time"].max()
                total_time = max_time - min_time
                throughput = len(group)/total_time
                if not is_list:
                    y_value = np.percentile(group[column], percentile)
                    x_vals.append(req_per_sec)
                    x_label = "Throughput (req/sec)"
                else:
                    tbt_times = group["tbt_times"].explode()
                    y_value = np.percentile(tbt_times, percentile) if not tbt_times.empty else np.nan
                    x_vals.append(req_per_sec)
                    x_label = "Request/s"

                
                y_vals.append(y_value)
            
            if len(x_vals) > 0:
                color = colors[exp_idx % len(colors)]
                
                # Plot dos pontos
                plt.plot(x_vals, y_vals, marker='o', color=color, 
                        label=f"{exp} - {model}", linestyle='-')
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{title} vs {x_label}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        filename = os.path.join(output_dir, image_name)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plots.append(filename)
    
    return plots

def generate_table_markdown(df, experiments):
    rows = []
    for exp in experiments:
        subset = df[df["experiment_key"] == exp]
        for _, row in subset.iterrows():
            rows.append([
                row["model_alias"],
                int(row["prompt_size"]),
                int(row["num_decoded_tokens"]),
                round(row["e2e_latency"], 4),
                round(row["total_throughput"], 4),
                round(row["ttft"], 4),
                round(row["prefill_throughput"], 4),
                round(row["total_decode_time"], 4),
                round(row["decode_throughput"], 4),
                round(row["tbt_median"], 4),
                round(row["tbt_p99"], 4),
            ])
    
    headers = [
        "Model", "Prompt Size", "Decode Size", 
        "E2E (s)", "Total Throughput (tok/s)", "TTFT (s)", 
        "Prefill Throughput (tok/s)", "Decode Time (s)", 
        "Decode Throughput (tok/s)", "TBT median (s)", "TBT 99th percentile (s)"
    ]
    
    rows_sorted = sorted(rows, key=lambda x: (x[0], x[1]))
    
    if rows_sorted:
        max_throughput = max(r[4] for r in rows_sorted)
        for r in rows_sorted:
            if r[4] == max_throughput:
                r[0] = f"**{r[0]}**"  

    md_table = tabulate(rows_sorted, headers, tablefmt="github")
    return md_table

def filter_by_single_goal(df, goal_name, threshold):
    if goal_name == "goal_ttft":
        filtered_df = df[df["ttft"] < threshold]
    elif goal_name == "e2e_goal":
        filtered_df = df[df["e2e_latency"] < threshold]
    elif goal_name == "goal_decode":
        filtered_df = df[df["total_decode_time"] < threshold]
    else:
        return None
    return filtered_df

def generate_markdown_report(plots, table_md, goal_tables, output_path, report_type):
    md = f"# Benchmark Report ({report_type})\n\n"
    for plot in plots:
        md += f"![{os.path.basename(plot)}]({os.path.basename(plot)})\n\n"
    if table_md:
        md += "## Summary Table\n\n"
        md += table_md + "\n\n"

    if goal_tables:
        md += "## Results Filtered by Individual Goals\n\n"
        for goal_title, table in goal_tables:
            md += f"### {goal_title}\n\n"
            md += table + "\n\n"

    report_file = os.path.join(output_path, "report.md")
    with open(report_file, "w") as f:
        f.write(md)

def gen_single_prompt_experiment_report(args, df):
    agg_df = aggregate_metrics(df)
    plots = generate_single_prompt_plots(agg_df, args.output_path, args.experiments)
    table_md = generate_table_markdown(agg_df, args.experiments)
    return plots, table_md, None

def calculate_real_time(initial_times, final_times):
    times = sorted(zip(map(float, initial_times), map(float, final_times)))

    if not times:
        return 0.0

    total_time = 0.0
    current_start, current_end = times[0]

    for start, end in times[1:]:
        if start <= current_end: 
            current_end = max(current_end, end)
        else:  
            total_time += current_end - current_start
            current_start, current_end = start, end

    total_time += current_end - current_start
    return total_time

def plot_throughputs(df, args):
    plots = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Dicionários para armazenar dados de todos os experimentos
    all_total_data = {}
    all_prefill_data = {}
    all_decode_data = {}
    
    for exp_idx, experiment in enumerate(args.experiments):
        experiment_df = df[df["experiment_key"] == experiment]
        
        x_total, y_total = [], []
        x_prefill, y_prefill = [], []
        x_decode, y_decode = [], []
        
        for req_per_sec in sorted(experiment_df["req_per_sec"].unique()):
            filtered_df = experiment_df[experiment_df["req_per_sec"] == req_per_sec]
            if filtered_df.empty:
                continue

            total_prompt_tokens = filtered_df["prompt_size"].sum()
            total_decode_tokens = filtered_df["num_decoded_tokens"].sum()
            total_tokens = total_prompt_tokens + total_decode_tokens

            initial_times = filtered_df["initial_time"]
            final_ttft_times = initial_times + filtered_df["ttft"]
            decode_end_times = filtered_df["final_time"]

            total_ttft_time = calculate_real_time(initial_times, final_ttft_times)
            total_decode_time = calculate_real_time(final_ttft_times, decode_end_times)
            total_time = filtered_df["final_time"].max() - filtered_df["initial_time"].min()
            
            throughput_total = total_tokens / total_time if total_time > 0 else 0.0
            throughput_prefill = total_prompt_tokens / total_ttft_time if total_ttft_time > 0 else 0.0
            throughput_decode = total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.0
            
            x_total.append(req_per_sec)
            y_total.append(throughput_total)
            x_prefill.append(req_per_sec)
            y_prefill.append(throughput_prefill)
            x_decode.append(req_per_sec)
            y_decode.append(throughput_decode)
        
        all_total_data[experiment] = (x_total, y_total, exp_idx)
        all_prefill_data[experiment] = (x_prefill, y_prefill, exp_idx)
        all_decode_data[experiment] = (x_decode, y_decode, exp_idx)
    
    # Plot Total Throughput
    plt.figure(figsize=(10, 6))
    for experiment, (x, y, exp_idx) in all_total_data.items():
        if x:
            color = colors[exp_idx % len(colors)]
            plt.plot(x, y, marker='o', color=color, label=experiment)
            
            max_idx = y.index(max(y))
            max_x = x[max_idx]
            max_y = y[max_idx]
            plt.scatter([max_x], [max_y], color=color, s=100, zorder=5, edgecolors='black', linewidths=2)
    
    plt.xlabel("Requests per Second")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Total Throughput vs Requests per Second")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(args.output_path, "total_throughput.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plots.append(output_path)
    
    # Plot Prefill Throughput
    plt.figure(figsize=(10, 6))
    for experiment, (x, y, exp_idx) in all_prefill_data.items():
        if x:
            color = colors[exp_idx % len(colors)]
            plt.plot(x, y, marker='s', color=color, label=experiment)
            
            max_idx = y.index(max(y))
            max_x = x[max_idx]
            max_y = y[max_idx]
            plt.scatter([max_x], [max_y], color=color, s=100, zorder=5, edgecolors='black', linewidths=2)
    
    plt.xlabel("Requests per Second")
    plt.ylabel("Prefill Throughput (tokens/s)")
    plt.title("Prefill Throughput vs Requests per Second")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(args.output_path, "prefill_throughput.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plots.append(output_path)
    
    # Plot Decode Throughput
    plt.figure(figsize=(10, 6))
    for experiment, (x, y, exp_idx) in all_decode_data.items():
        if x:
            color = colors[exp_idx % len(colors)]
            plt.plot(x, y, marker='^', color=color, label=experiment)
            
            max_idx = y.index(max(y))
            max_x = x[max_idx]
            max_y = y[max_idx]
            plt.scatter([max_x], [max_y], color=color, s=100, zorder=5, edgecolors='black', linewidths=2)
    
    plt.xlabel("Requests per Second")
    plt.ylabel("Decode Throughput (tokens/s)")
    plt.title("Decode Throughput vs Requests per Second")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(args.output_path, "decode_throughput.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plots.append(output_path)
    
    return plots

def gen_load_experiment_report(args, df):
    plots = []
    
    # Gerar plots de throughput
    throughput_plots = plot_throughputs(df, args)
    plots.extend(throughput_plots)
    
    # Gerar plots de latência vs req/s
    latency_plots = generate_latency_vs_req_per_sec_plots(df, args.output_path, args.experiments)
    plots.extend(latency_plots)
    
    # Gerar tabela de throughput máximo
    goal_value = args.e2e_goal
    goal_column = "e2e_goal"
    goal_name = "End-to-End Latency"

    filtered_goal_df = filter_by_single_goal(df, goal_column, goal_value)

    table_rows = []

    for exp in args.experiments:
        filtered_exp_df = filtered_goal_df[filtered_goal_df["experiment_key"] == exp]

        if filtered_exp_df.empty:
            table_rows.append({
                "Experiment": exp,
                "Model": "N/A",
                "Max Total Throughput (tokens/s)": 0,
                "Prefill Throughput (tokens/s)": 0,
                "Decode Throughput (tokens/s)": 0,
            })
            continue

        max_total_throughput = 0
        best_prefill_thp = 0
        best_decode_thp = 0
        best_model = None
        decoded_tokens = None
        avg_prompt_size = None
        best_req_per_sec = None

        for req_per_sec in filtered_exp_df["req_per_sec"].unique():
            df_ps = filtered_exp_df[filtered_exp_df["req_per_sec"] == req_per_sec]
            if df_ps.empty:
                continue

            total_prompt_tokens = df_ps["prompt_size"].sum()
            total_decoded_tokens = df_ps["num_decoded_tokens"].sum()
            all_req_df = df[(df["experiment_key"] == exp) & (df["req_per_sec"] == req_per_sec)]

            initial_times = all_req_df["initial_time"]
            final_ttft_times = initial_times + all_req_df["ttft"]
            decode_end_times = all_req_df["final_time"]

            total_ttft_time = calculate_real_time(initial_times, final_ttft_times)
            total_decode_time = calculate_real_time(final_ttft_times, decode_end_times)
            total_time = all_req_df["final_time"].max() - all_req_df["initial_time"].min()
            
            total_throughput = (total_prompt_tokens + total_decoded_tokens) / total_time if total_time > 0 else 0.0
            prefill_throughput = total_prompt_tokens / total_ttft_time if total_ttft_time > 0 else 0.0
            decode_throughput = total_decoded_tokens / total_decode_time if total_decode_time > 0 else 0.0

            if total_throughput > max_total_throughput:
                max_total_throughput = total_throughput
                best_prefill_thp = prefill_throughput
                best_decode_thp = decode_throughput
                best_model = df_ps["model_alias"].iloc[0] if "model_alias" in df_ps.columns else "Unknown"
                decoded_tokens = df_ps["num_decoded_tokens"].mean()
                avg_prompt_size = df_ps["prompt_size"].mean()
                best_req_per_sec = req_per_sec

        table_rows.append({
            "Experiment": exp,
            "Model": best_model if best_model else "Unknown",
            "Max Total Throughput (tokens/s)": round(max_total_throughput, 2),
            "Prefill Throughput (tokens/s)": round(best_prefill_thp, 2),
            "Decode Throughput (tokens/s)": round(best_decode_thp, 2),
            "Avg Prompt size": round(avg_prompt_size, 2),  
            "Avg Decoded Tokens per prompt": round(decoded_tokens, 2) if decoded_tokens else "N/A",
            "Req/s": best_req_per_sec if best_req_per_sec else "N/A",
            "Best batch (tok/s)": round(best_req_per_sec * (avg_prompt_size), 2) if best_req_per_sec else "N/A"
        })

    table_df = pd.DataFrame(table_rows)

    table_md = f"## Max Throughput for {goal_name} < {goal_value}s\n\n"
    table_md += table_df.to_markdown(index=False)
    table_md += "\n\n"

    return plots, table_md, None

def main():
    args = parse_args()
    
    for experiment in args.experiments:
        os.makedirs(os.path.join(args.output_path, experiment), exist_ok=True)

    df = load_data(args.csv_path)

    if args.experiment_is_group:
        df = df[df["experiment_group"].isin(args.experiments)]
    else:
        df = df[df["experiment_key"].isin(args.experiments)]

    if df.empty:
        print("Warning: No data found for the selected experiments/groups.")
        return

    if args.type == "single_prompt":
        print("Generating report for single prompt experiments...")
        plots, table_md, table_md_goals = gen_single_prompt_experiment_report(args, df)
        report_type = "Single Prompt"
    else:
        print("Generating report for load experiments...")
        plots, table_md, table_md_goals = gen_load_experiment_report(args, df)
        report_type = "Load"

    generate_markdown_report(plots, table_md, table_md_goals, args.output_path, report_type)
    print(f"Report generated at {os.path.join(args.output_path, 'report.md')}")


if __name__ == "__main__":
    main()