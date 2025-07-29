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
    
    for (key, prompt_size, model_name), group in df.groupby(["key", "prompt_size", "model_name"]):
        agg = {}
        agg["key"] = key
        agg["prompt_size"] = prompt_size
        agg["model_name"] = model_name
        agg["num_decoded_tokens"] = group["num_decoded_tokens"].mean()
        agg["e2e_latency"] = group["e2e_latency"].mean()
        agg["ttft"] = group["ttft"].mean()
        agg["total_decode_time"] = group["total_decode_time"].mean()
        agg["req_per_sec"] = group["req_per_sec"].mean() if "req_per_sec" in group else np.nan

        tbt_lists = group["tbt_times"].tolist()

        if len(tbt_lists) > 0:
            tbt_array = np.array(tbt_lists)  # shape (n_runs, sequence_len)
            mean_tbt_curve = np.nanmean(tbt_array, axis=0)

            # --> aqui calculamos valores únicos sobre a curva média:
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
            subset = df[df["key"] == exp]
            for model in subset["model_name"].unique():
                model_data = subset[subset["model_name"] == model]
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

def generate_latency_vs_tok_per_sec_plot(df, output_dir, experiments, column, percentil, xlabel, ylabel, title, image_name, y_is_list):
    plt.figure(figsize=(8,5))
    for exp in experiments:
        subset = df[df["key"] == exp]
        if subset.empty:
            raise ValueError(f"No data found for experiment '{exp}'")

        models = subset["model_name"].unique()
        if len(models) != 1:
            raise ValueError(f"Expected exactly one model for experiment {exp}, but found {models}")
        model = models[0]

        x_vals = []
        y_vals = []
        for req_per_sec in sorted(subset["req_per_sec"].unique()):
            group = subset[subset["req_per_sec"] == req_per_sec]
            if group.empty:
                continue

            total_tokens  = group["prompt_size"].sum() + group["num_decoded_tokens"].sum()
            total_time = group["final_time"].max() - group["initial_time"].min()
            throughput = total_tokens / total_time
            if not y_is_list:
                y_value = np.percentile(group[column], percentil)
            else:
                tbt_times = group["tbt_times"].explode()
                y_value = np.percentile(tbt_times, percentil) if not tbt_times.empty else np.nan
            
            x_vals.append(throughput)
            y_vals.append(y_value)

        if len(x_vals) > 0:
            plt.plot(x_vals, y_vals, marker='o', label=f"{exp} - {model}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    filename = os.path.join(output_dir, image_name)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return filename

def generate_table_markdown(df, experiments):
    rows = []
    for exp in experiments:
        subset = df[df["key"] == exp]
        for _, row in subset.iterrows():
            rows.append([
                row["model_name"],
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
        "Decode Throughput (tok/s)", "TBT median (s)", "TBT 99th percentile (s)"    ]
    
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

    def compute_throughput_for_experiment(experiment_df):
        x_total, y_total = [], []
        x_prefill, y_prefill = [], []
        x_decode, y_decode = [], []
        columns = [
                "req_per_sec",
                "total_decoded_tokens",
                "total_prefill_tokens",
                "total_time(max-min)",
                "total_time(real_time)",
                "throughput_total",
                "total_ttft_time",
                "throughput_prefill",
                "total_decode_time",
                "throughput_decode"
        ]
        # header = "  ".join(f"{col:<22}" for col in columns)

        # print(header)

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
            total_time2 = calculate_real_time(initial_times, decode_end_times)
            throughput_total = total_tokens / total_time if total_time > 0 else 0.0
            throughput_prefill = total_prompt_tokens / total_ttft_time if total_ttft_time > 0 else 0.0
            throughput_decode = total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.0
            # print(filtered_df["final_time"].max(), filtered_df["initial_time"].min())
            # print(f"{req_per_sec:<22} {total_decode_tokens:<22} {total_prompt_tokens:<22} "
            #       f"{total_time:<22} {total_time2:<22} {throughput_total:<22} "
            #       f"{total_ttft_time:<22} {throughput_prefill:<22} {total_decode_time:<22} {throughput_decode:<22}")
            x_val = req_per_sec
            x_total.append(x_val)
            y_total.append(throughput_total)
            x_prefill.append(x_val)
            y_prefill.append(throughput_prefill)
            x_decode.append(x_val)
            y_decode.append(throughput_decode)

        return (x_total, y_total), (x_prefill, y_prefill), (x_decode, y_decode)

    def save_plot(x, y, title, ylabel, filename, prompt_size, decode_size, experiment, color=None, marker='o'):
        if not x:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker=marker, color=color, label=experiment)

        # Ajuste automático da posição do texto da anotação
        if y:
            max_idx = y.index(max(y))
            max_x = x[max_idx]
            max_y = y[max_idx]

            plt.scatter([max_x], [max_y], color='red', zorder=5)

            y_offset = 25 if max_y <= max(y) * 0.9 else -40  

            plt.annotate(
                f"Max\n{max_y:.2f} tok/s\n@ {max_x} req/s",
                (max_x, max_y),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
                fontsize=9,
                color='black'
            )

        plt.xlabel("Requests per second")
        plt.ylabel(ylabel)
        plt.title(f"{title} vs requests per sec with {int(prompt_size)}/{decode_size} prompt size/max decode size per request")
        plt.legend()
        output_path = os.path.join(args.output_path, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        plots.append(output_path)

    for experiment in args.experiments:
        experiment_df = df[df["key"] == experiment]

        (x_total, y_total), (x_prefill, y_prefill), (x_decode, y_decode) = compute_throughput_for_experiment(experiment_df)

        prompt_size = experiment_df["prompt_size"].mean()
        decode_size = experiment_df["num_decoded_tokens"].max()

        save_plot(x_total, y_total, "Total Throughput", "Throughput (tokens/s)", f"{experiment}_total_throughput.png", prompt_size, decode_size, experiment)
        save_plot(x_prefill, y_prefill, "Prefill Throughput", "Prefill Throughput (tokens/s)", f"{experiment}_prefill_throughput.png", prompt_size, decode_size, experiment, color="orange", marker="s")
        save_plot(x_decode, y_decode, "Decode Throughput", "Decode Throughput (tokens/s)", f"{experiment}_decode_throughput.png", prompt_size, decode_size, experiment, color="green", marker="^")

    return plots



def gen_load_experiment_report(args, df):
    plots = []
    columns = ["e2e_latency","e2e_latency", "ttft", "ttft", "tbt_times", "tbt_times"]
    percetiles = [50, 99, 50, 99, 50, 99]

    plt.figure(figsize=(8, 5))
    plots = plot_throughputs(df, args)

    titles = [
        "End-to-End Latency Median(s)",
        "End-to-End Latency P99(s)",
        "Time To First Token Median (s)",
        "Time To First Token P99 (s)",
        "Time Between Tokens Median (s)",
        "Time Between Tokens P99 (s)"
    ]
    prompt_size = df["prompt_size"].mean()
    decode_size = df["num_decoded_tokens"].mean()
    titles = [title + f" vs Throughput (tokens/s) with {prompt_size}/{round(decode_size, 2)} prompt size/mean decode size" for title in titles]
    x_label = "Throughput (tokens/s)"

    image_names = [
        "e2e_median_latency_vs_tok_per_sec.png",
        "e2e_latency_vs_tok_per_sec.png",
        "ttft_median_vs_tok_per_sec.png",
        "ttft_p99_vs_tok_per_sec.png",
        "tbt_median_vs_tok_per_sec.png",
        "tbt_p99_vs_tok_per_sec.png"
    ]

    y_is_list_flags = [False, False, False, False, True, True]

    y_labels = [
        "End-to-End Latency Median (s)",
        "End-to-End Latency P99 (s)",
        "Time To First Token Median (s)",
        "Time To First Token P99 (s)",
        "Time Between Tokens Median (s)",
        "Time Between Tokens P99 (s)"
    ]
    for i, (column, percentil, title, image_name, is_list, y_label) in enumerate(zip(
        columns, percetiles, titles, image_names, y_is_list_flags, y_labels
    )):
        plot = generate_latency_vs_tok_per_sec_plot(
            df, args.output_path, args.experiments, column, percentil, x_label, y_label,  title, image_name, is_list
        )
        plots.append(plot)


    goal_value = args.e2e_goal
    goal_column = "e2e_goal"
    goal_name = "End-to-End Latency"

    filtered_goal_df = filter_by_single_goal(df, goal_column, goal_value)

    table_rows = []

    for exp in args.experiments:
        filtered_exp_df = filtered_goal_df[filtered_goal_df["key"] == exp]

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
            all_req_df = df[(df["key"] == exp) & (df["req_per_sec"] == req_per_sec)]

            initial_times = all_req_df["initial_time"]
            final_ttft_times = initial_times + all_req_df["ttft"]
            decode_end_times = all_req_df["final_time"]

            total_ttft_time = calculate_real_time(initial_times, final_ttft_times)
            total_decode_time = calculate_real_time(final_ttft_times, decode_end_times)
            total_time2 = calculate_real_time(initial_times, decode_end_times)
            total_time = all_req_df["final_time"].max() - all_req_df["initial_time"].min()
            assert(total_time2 <= total_time)
            total_throughput = (total_prompt_tokens + total_decoded_tokens) / total_time if total_time > 0 else 0.0
            prefill_throughput = total_prompt_tokens / total_ttft_time if total_ttft_time > 0 else 0.0
            decode_throughput = total_decoded_tokens / total_decode_time if total_decode_time > 0 else 0.0

            if total_throughput > max_total_throughput:
                max_total_throughput = total_throughput
                best_prefill_thp = prefill_throughput
                best_decode_thp = decode_throughput
                best_model = df_ps["model_name"].iloc[0] if "model_name" in df_ps.columns else "Unknown"
                decoded_tokens = df_ps["num_decoded_tokens"].mean()
                avg_prompt_size= df_ps["prompt_size"].mean()
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
            "Best batch (tok/s)": best_req_per_sec*(avg_prompt_size + decoded_tokens) if best_req_per_sec else "N/A"
        })

    # Monta a tabela em Markdown
    table_df = pd.DataFrame(table_rows)

    table_md = f"## Max Throughput for {goal_name} < {goal_value}s\n\n"
    table_md += table_df.to_markdown(index=False)
    table_md += "\n\n"

        # goal_tables.append("",table_md)

    return plots, table_md, None


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    df = load_data(args.csv_path)
    df = df[df["key"].isin(args.experiments)]

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
