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
    parser.add_argument("--goal_ttft", type=float, default=None, help="Goal threshold for TTFT in seconds")
    parser.add_argument("--goal_e2e", type=float, default=None, help="Goal threshold for End-to-End latency in seconds")
    parser.add_argument("--goal_decode", type=float, default=None, help="Goal threshold for decode time in seconds")
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
        for prompt_size in sorted(subset["prompt_size"].unique()):
            group = subset[subset["prompt_size"] == prompt_size]
            if group.empty:
                continue

            total_tokens  = group["prompt_size"].mean()
            # total_time = group["final_time"].max() - group["initial_time"].min()
            # tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
            if not y_is_list:
                y_value = np.percentile(group[column], percentil)
            else:
                tbt_times = group["tbt_times"].explode()
                y_value = np.percentile(tbt_times, percentil) if not tbt_times.empty else np.nan
            
            x_vals.append(total_tokens)
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
    elif goal_name == "goal_e2e":
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


def gen_load_experiment_report(args, df):
    plots = []
    columns = ["e2e_latency", "ttft", "ttft", "tbt_times", "tbt_times"]
    percetiles = [99, 50, 99, 50, 99]

    ## Throughput plot (tudo em um só gráfico)
    plt.figure(figsize=(8, 5))

    for experiment in args.experiments:
        thp_x_values = []
        thp_y_values = []

        filtered_df_exp = df[df["key"] == experiment]

        for prompt_size in filtered_df_exp["prompt_size"].unique():
            filtered_df = filtered_df_exp[filtered_df_exp["prompt_size"] == prompt_size]
            if filtered_df.empty:
                continue
            total_prompts = len(filtered_df)
            
            total_processed_tokens = (
                filtered_df["prompt_size"].sum()
                + filtered_df["num_decoded_tokens"].sum()
            )
            total_time = (
                filtered_df["final_time"].max()
                - filtered_df["initial_time"].min()
            )
            throughput = (
                total_processed_tokens / total_time
                if total_time > 0
                else 0.0
            )
            prompts_per_req = filtered_df.groupby("request_id").count()["e2e_latency"].mean()
            decode_size = filtered_df["num_decoded_tokens"].mean()
            req_per_sec = filtered_df.iloc[0]["req_per_sec"]
            requested_tokens_per_sec = (
            (prompt_size + decode_size)
                * req_per_sec
                * prompts_per_req
            )
            if prompt_size == 1024:
                a = 0
            thp_x_values.append(requested_tokens_per_sec)
            thp_y_values.append(throughput)

        if len(thp_x_values) > 0:
            plt.plot(
                thp_x_values,
                thp_y_values,
                marker='o',
                label=experiment
            )

    plt.xlabel("Requested Tokens per Second (prompt tokens/s)")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Throughput vs Requested Tokens per Second (all experiments)")
    plt.legend()
    image_name = "throughput_vs_requested_tokens_all_experiments.png"
    output_path = os.path.join(args.output_path, image_name)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plots.append(output_path)

    titles = [
        "End-to-End Latency P99(s)",
        "Time To First Token Median (s)",
        "Time To First Token P99 (s)",
        "Time Between Tokens Median (s)",
        "Time Between Tokens P99 (s)"
    ]
    decode_size = df["num_decoded_tokens"].mean()
    titles = [title + f" vs Throughput (prompt tokens/s) with {decode_size} decoded tokens" for title in titles]
    x_label = "Throughput (tokens/s)"
    image_names = [
        "e2e_latency_vs_tok_per_sec.png",
        "ttft_median_vs_tok_per_sec.png",
        "ttft_p99_vs_tok_per_sec.png",
        "tbt_median_vs_tok_per_sec.png",
        "tbt_p99_vs_tok_per_sec.png"
    ]
    y_is_list_flags = [False, False, False, True, True]
    y_labels = [
        "End-to-End Latency (s)",
        "Time To First Token Median (s)",
        "Time To First Token P99 (s)",
        "Time Between Tokens Median (s)",
        "Time Between Tokens P99 (s)"
    ]
    for column, percentil, title, image_name, is_list, y_label in zip(
        columns, percetiles, titles, image_names, y_is_list_flags, y_labels
    ):
        plot = generate_latency_vs_tok_per_sec_plot(
            df, args.output_path, args.experiments, column, percentil, x_label, y_label,  title, image_name, is_list
        )
        plots.append(plot)



    goals = {
        "goal_ttft": args.goal_ttft,
        "goal_e2e": args.goal_e2e,
        "goal_decode": args.goal_decode
    }
    goal_column_to_name = {
        "goal_ttft": "TTFT",
        "goal_e2e": "End-to-End Latency",
        "goal_decode": "Decode Time"
    }
    table_md = ""
    for goal_column, goal_value in goals.items():

        filtered_goal_df = filter_by_single_goal(df, goal_column, goal_value)
        
        table_rows = []

        for exp in args.experiments:
            filtered_exp_df = filtered_goal_df[filtered_goal_df["key"] == exp]

            if filtered_exp_df.empty:
                table_rows.append({
                    "Experiment": exp,
                    "Model": "N/A",
                    "Max Throughput (tokens/s)": 0
                })
                continue

            max_throughput = 0
            best_model = None
            decoded_tokens = None
            for prompt_size in filtered_exp_df["prompt_size"].unique():
                df_ps = filtered_exp_df[filtered_exp_df["prompt_size"] == prompt_size]
                
                if df_ps.empty:
                    continue

                total_prompt_tokens = df_ps["prompt_size"].sum() 
                all_prompt_df = df[df["key"] == exp][df["prompt_size"] == prompt_size]
                exp_time = all_prompt_df["final_time"].max() - all_prompt_df["initial_time"].min()

                
                if exp_time > 0:
                    throughput = total_prompt_tokens / exp_time
                else:
                    throughput = 0

                if throughput > max_throughput:
                    max_throughput = throughput
                    best_model = df_ps["model_name"].iloc[0]
                
                if decoded_tokens is None:
                    decoded_tokens = df_ps["num_decoded_tokens"].mean()

            table_rows.append({
                "Experiment": exp,
                "Model": best_model if best_model else "Unknown",
                f"Max Throughput (prompt tokens/s) with {decoded_tokens} decoded tokens": round(max_throughput, 2)
            })

        table_df = pd.DataFrame(table_rows)
        
        table_md += f"## Max Throughput for {goal_column_to_name[goal_column]} < {goal_value}s\n\n"
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
