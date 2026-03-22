import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tabulate import tabulate

EXPERIMENT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Markdown report from model CSV results.")
    def parse_tuple_list(string):
        # Accept formats like "acc_key:perf_key" or "acc_key,perf_key" or "acc_key perf_key"
        # Return tuple (acc_key, perf_key)
        if ":" in string:
            parts = string.split(":")
        elif "," in string:
            parts = string.split(",")
        else:
            parts = string.split()
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "associated_experiments items must contain two keys separated by ':' or ',' or space, e.g. acc:perf"
            )
        return (parts[0].strip(), parts[1].strip())

    parser.add_argument("--csv_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Path to save results (directory)")
    parser.add_argument("--experiments", nargs="+", required=True, help="List of experiments (key) to compare")
    parser.add_argument("--experiment_is_group", action="store_true")
    parser.add_argument("--type", choices=["single_prompt", "load"], required=True)
    parser.add_argument("--e2e_goal", type=float, default=None, help="Goal threshold for End-to-End latency in seconds")
    parser.add_argument("--acc_csv_path", required=False, help="List of accuracy and performance result files in CSV format")
    parser.add_argument("--associated_experiments", type=parse_tuple_list, nargs='*', required=False, default=None, help="Pairs of keys to consider for Pareto frontier in the format acc_key:perf_key")
    return parser.parse_args()

def get_exp_color_map(experiments):
    return {exp: EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)] for i, exp in enumerate(experiments)}

def load_acc_csv(csv_path):
    df = pd.read_csv(csv_path)
    columns_to_remove = ["dataset","version", "metric", "mode"]
    # remove columns if they exist
    columns_to_remove = [c for c in columns_to_remove if c in df.columns]
    filtred_df = df.drop(columns=columns_to_remove, axis=1)
    return filtred_df


def get_mean_acc_dict_from_df(df):
    columns = df.columns
    ret = {}
    for col in columns:
        try:
            ret[col] = df[col].mean()
        except Exception:
            ret[col] = np.nan
    return ret


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_tbt(val):
        if pd.isna(val):
            return []
        if isinstance(val, str):
            try:
                return [float(x) for x in json.loads(val)]
            except Exception:
                # try simple splitting if not JSON
                try:
                    return [float(x) for x in val.strip().split()]
                except Exception:
                    return []
        # if already list-like
        return val

    # Make sure expected columns exist
    if "tbt_times" in df.columns:
        df["tbt_times"] = df["tbt_times"].apply(parse_tbt)
    else:
        df["tbt_times"] = [[] for _ in range(len(df))]

    # Ensure numeric columns are correct dtype where applicable
    numeric_cols = ["prompt_size", "num_decoded_tokens", "e2e_latency", "ttft", "total_decode_time", "req_per_sec", "initial_time", "final_time"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
        agg["req_per_sec"] = group["req_per_sec"].mean() if "req_per_sec" in group.columns else np.nan

        tbt_lists = group["tbt_times"].tolist()

        if len(tbt_lists) > 0:
            # normalize ragged lists by padding with nan to compute mean per position
            max_len = max((len(x) for x in tbt_lists if isinstance(x, (list, tuple)) ), default=0)
            if max_len > 0:
                padded = []
                for lst in tbt_lists:
                    if isinstance(lst, (list, tuple)):
                        arr = list(lst) + [np.nan] * (max_len - len(lst))
                        padded.append(arr)
                    else:
                        padded.append([np.nan] * max_len)
                tbt_array = np.array(padded, dtype=float)
                mean_tbt_curve = np.nanmean(tbt_array, axis=0)
                median_tbt_value = np.nanmedian(mean_tbt_curve)
                p99_tbt_value = np.nanpercentile(mean_tbt_curve, 99)
                agg["tbt_times"] = mean_tbt_curve.tolist()
                agg["tbt_median"] = float(median_tbt_value)
                agg["tbt_p99"] = float(p99_tbt_value)
            else:
                agg["tbt_times"] = []
                agg["tbt_median"] = np.nan
                agg["tbt_p99"] = np.nan
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
            if e2e_latency_mean and e2e_latency_mean > 0 else 0.0
        )
        
        agg["prefill_throughput"] = (
            prompt_size_mean / ttft_mean
            if ttft_mean and ttft_mean > 0 else 0.0
        )

        agg["decode_throughput"] = (
            decode_size_mean / total_decode_time_mean
            if total_decode_time_mean and total_decode_time_mean > 0 else 0.0
        )

        grouped.append(agg)

    agg_df = pd.DataFrame(grouped)
    return agg_df

def generate_single_prompt_plots(df, output_dir, experiments):
    metrics = [
        ("ttft", "Time To First Token (s)"),
        ("tbt_median", "Median Time-between-tokens Latency (s)"),
        ("tbt_p99", "P99 Time-between-tokens Latency (s)"),
        ("total_throughput", "Throughput (tokens/s)"),
        ("e2e_latency", "End-to-End Latency (s)"),
        ("prefill_throughput", "Prefill Throughput (tokens/s)"),
        ("decode_throughput", "Decode Throughput (tokens/s)")
    ]

    plots = []
    for metric, title in metrics:
        plt.figure(figsize=(12,5))
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
    """
    points: array-like de shape (N,2) com (throughput, accuracy)
    Retorna os índices dos pontos na fronteira de Pareto (maior é melhor).
    """
    if len(points) == 0:
        return []

    points = np.array(points)
    pareto_indices = []

    # Ordena por throughput decrescente
    sorted_indices = np.argsort(-points[:, 0])  # maior throughput primeiro
    max_acc = -np.inf

    for idx in sorted_indices:
        if points[idx, 1] >= max_acc:
            pareto_indices.append(idx)
            max_acc = points[idx, 1]

    # Ordena de volta por throughput crescente para plot
    pareto_indices.sort(key=lambda i: points[i, 0])
    return pareto_indices


def compute_max_throughput_for_experiment(experiment_df):
    max_throughput = 0
    if experiment_df.empty:
        return max_throughput
    for req_per_sec in sorted(experiment_df["req_per_sec"].dropna().unique()):
        filtered_df = experiment_df[experiment_df["req_per_sec"] == req_per_sec]
        if filtered_df.empty:
            continue

        total_prompt_tokens = filtered_df["prompt_size"].sum()
        total_decode_tokens = filtered_df["num_decoded_tokens"].sum()
        total_tokens = total_prompt_tokens + total_decode_tokens

        total_time = filtered_df["final_time"].max() - filtered_df["initial_time"].min()
        throughput_total = total_tokens / total_time if total_time and total_time > 0 else 0.0
        max_throughput = max(max_throughput, throughput_total)

    return max_throughput
def get_pareto_plot(associated_experiments, mean_acc, perf_df, output_path, experiments, filename="pareto_front.png"):
    exp_to_color = get_exp_color_map(experiments)

    # Mapear experimentos de acurácia para experimentos de performance
    acc_exp_key_to_perf_exp_key = {acc: perf for acc, perf in associated_experiments}
    all_points = []

    for acc_key, perf_key in acc_exp_key_to_perf_exp_key.items():
        df_exp = perf_df[perf_df["experiment_key"] == perf_key]
        if df_exp.empty:
            continue
        max_throughput = compute_max_throughput_for_experiment(df_exp)
        acc = mean_acc.get(acc_key, np.nan)
        all_points.append((max_throughput, acc, perf_key))

    if not all_points:
        print("⚠️ No points found for Pareto frontier.")
        return None

    points_array = np.array([(p[0], p[1]) for p in all_points])

    # Função de Pareto: pontos não dominados
    pareto_indices = compute_pareto_frontier(points_array)
    pareto_points = [all_points[i] for i in pareto_indices]

    plt.figure(figsize=(12,5))
    unique_keys = list({p[2] for p in all_points})

    # Plotar todos os pontos
    for key in unique_keys:
        xs = [p[0] for p in all_points if p[2] == key]
        ys = [p[1] for p in all_points if p[2] == key]
        color = exp_to_color.get(key, '#333333')
        plt.scatter(xs, ys, color=color, s=100, alpha=0.7, edgecolors='black', label=key, zorder=2)

    # Linha preta conectando pontos da fronteira
    if pareto_points:
        pareto_throughputs = [p[0] for p in pareto_points]
        pareto_accuracies = [p[1] for p in pareto_points]
        plt.plot(pareto_throughputs, pareto_accuracies, color="black", linestyle="-", linewidth=2.0, marker=None, zorder=3)

    plt.xlabel("Throughput (Tok/s)", fontsize=12)
    plt.ylabel("Mean Accuracy", fontsize=12)
    plt.title("Pareto Frontier: Mean Accuracy vs Throughput (Tok/s)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Modelos", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    return output_file




def generate_latency_vs_req_per_sec_plots(df, output_dir, experiments):
    plots = []
    exp_to_color = get_exp_color_map(experiments)
    
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
    y_labels = titles
    image_names = [
        "e2e_median_latency_vs_req_per_sec.png",
        "e2e_p99_latency_vs_req_per_sec.png",
        "ttft_median_vs_req_per_sec.png",
        "ttft_p99_vs_req_per_sec.png",
        "tbt_median_vs_req_per_sec.png",
        "tbt_p99_vs_req_per_sec.png"
    ]
    y_is_list_flags = [False, False, False, False, True, True]

    for column, percentile, title, y_label, image_name, is_list in zip(
        columns, percentiles, titles, y_labels, image_names, y_is_list_flags
    ):
        plt.figure(figsize=(12,5))
        for exp in experiments:
            subset = df[df["experiment_key"] == exp]
            if subset.empty:
                continue
            
            models = subset["model_alias"].unique()
            model = models[0] if len(models) > 0 else "unknown"
            
            x_vals, y_vals = [], []
            for req_per_sec in sorted(subset["req_per_sec"].dropna().unique()):
                group = subset[subset["req_per_sec"] == req_per_sec]
                if group.empty:
                    continue

                if not is_list:
                    y_value = np.percentile(group[column].dropna(), percentile) if not group[column].dropna().empty else np.nan
                else:
                    tbt_times = group["tbt_times"].explode().dropna().astype(float) if "tbt_times" in group.columns else pd.Series(dtype=float)
                    y_value = np.percentile(tbt_times, percentile) if not tbt_times.empty else np.nan
                
                x_vals.append(req_per_sec)
                y_vals.append(y_value)
            
            if x_vals:
                color = exp_to_color[exp]
                plt.plot(x_vals, y_vals, marker='o', linestyle='-', color=color, label=f"{exp} - {model}")
        
        plt.xlabel("Load (req/s)")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
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
        "E2E (s)", "Throughput (tok/s)", "TTFT (s)", 
        "Prefill Throughput (tok/s)", "Decode Time (s)", 
        "Decode Throughput (tok/s)", "TBT median (s)", "TBT 99th percentile (s)"
    ]
    
    rows_sorted = sorted(rows, key=lambda x: (x[0], x[1]))
    
    if not rows_sorted:
        return tabulate([], headers, tablefmt="github")

    # Destaque: maior throughput e maior acurácia média (se disponível)
    max_throughput = max(r[4] for r in rows_sorted)
    max_acc = max(r[9] for r in rows_sorted if not np.isnan(r[9]))

    for r in rows_sorted:
        if r[4] == max_throughput:
            r[0] = f"**{r[0]}** (↑Throughput)"
        if r[9] == max_acc:
            r[0] = f"**{r[0]}** (↑Accuracy)"
    
    md_table = tabulate(rows_sorted, headers, tablefmt="github")
    return md_table


def filter_by_single_goal(df, goal_name, threshold):
    if threshold is None:
        return df.copy()
    if goal_name == "goal_ttft":
        filtered_df = df[df["ttft"] < threshold]
    elif goal_name == "e2e_goal":
        filtered_df = df[df["e2e_latency"] < threshold]
    elif goal_name == "goal_decode":
        filtered_df = df[df["total_decode_time"] < threshold]
    else:
        return df.copy()
    return filtered_df

def generate_markdown_report(plots, table_md, goal_tables, output_path, report_type):
    md = f"# Benchmark Report ({report_type})\n\n"
    for plot in plots:
        if plot:
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
    # initial_times and final_times can be pandas Series or lists
    try:
        zipped = sorted(zip(map(float, initial_times), map(float, final_times)))
    except Exception:
        zipped = []
    if not zipped:
        return 0.0

    total_time = 0.0
    current_start, current_end = zipped[0]

    for start, end in zipped[1:]:
        if start <= current_end: 
            current_end = max(current_end, end)
        else:  
            total_time += current_end - current_start
            current_start, current_end = start, end

    total_time += current_end - current_start
    return total_time
def plot_throughputs(df, args):
    plots = []
    exp_to_color = get_exp_color_map(args.experiments)
    
    # Preparar dados
    all_data = {"Total": {}, "Prefill": {}, "Decode": {}}
    for exp in args.experiments:
        experiment_df = df[df["experiment_key"] == exp]
        x_total, y_total, x_prefill, y_prefill, x_decode, y_decode = [], [], [], [], [], []
        for req_per_sec in sorted(experiment_df["req_per_sec"].dropna().unique()):
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
            
            y_total.append(total_tokens / total_time if total_time > 0 else 0.0)
            y_prefill.append(total_prompt_tokens / total_ttft_time if total_ttft_time > 0 else 0.0)
            y_decode.append(total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.0)
            x_total.append(req_per_sec)
            x_prefill.append(req_per_sec)
            x_decode.append(req_per_sec)
        
        all_data["Total"][exp] = (x_total, y_total)
        all_data["Prefill"][exp] = (x_prefill, y_prefill)
        all_data["Decode"][exp] = (x_decode, y_decode)
    
    # Função genérica de plot
    def plot_metric(metric_name, data_dict):
        plt.figure(figsize=(12,5))
        for exp, (x, y) in data_dict.items():
            if x:
                color = exp_to_color[exp]
                plt.plot(x, y, marker='o', linestyle='-', color=color, label=exp)
        plt.xlabel("Load (req/s)")
        plt.ylabel(f"{metric_name} (tokens/s)")
        plt.title(f"{metric_name} vs Load (req/s)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        filename = os.path.join(args.output_path, f"{metric_name.lower().replace(' ','_')}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename

    plots.append(plot_metric("Total Throughput", all_data["Total"]))
    plots.append(plot_metric("Prefill Throughput", all_data["Prefill"]))
    plots.append(plot_metric("Decode Throughput", all_data["Decode"]))
    
    return plots



def gen_load_experiment_report(args, df):
    plots = []

    # --- Pareto plot ---
    if args.associated_experiments and args.acc_csv_path:
        mean_acc = get_mean_acc_dict_from_df(load_acc_csv(args.acc_csv_path))
        pareto_plot = get_pareto_plot(
            args.associated_experiments, 
            mean_acc, 
            df, 
            args.output_path,
            experiments=args.experiments
        )
        if pareto_plot:
            plots.append(pareto_plot)

    # --- Plots de throughput e latência ---
    plots.extend(plot_throughputs(df, args))
    plots.extend(generate_latency_vs_req_per_sec_plots(df, args.output_path, args.experiments))

    # --- Função auxiliar: gera tabela filtrando por req_per_sec ---
    def build_table(df_input):
        table_rows = []

        for exp in args.experiments:
            exp_df = df_input[df_input["experiment_key"] == exp]
            if exp_df.empty:
                table_rows.append({
                    "Experiment": exp,
                    "Model": "N/A",
                    "Max Throughput (tokens/s)": 0,
                    "Prefill Throughput (tokens/s)": 0,
                    "Decode Throughput (tokens/s)": 0,
                    "Avg Prompt size": 0,
                    "Avg Decoded Tokens per prompt": 0,
                    "Req/s": "N/A",
                    "Best batch (tok/s)": "N/A"
                })
                continue

            max_total_throughput = 0
            best_prefill_thp = 0
            best_decode_thp = 0
            best_model = None
            avg_prompt_size = None
            decoded_tokens = None
            best_req_per_sec = None

            for req_per_sec in sorted(exp_df["req_per_sec"].dropna().unique()):
                df_ps = exp_df[exp_df["req_per_sec"] == req_per_sec]
                if df_ps.empty:
                    continue

                total_prompt_tokens = df_ps["prompt_size"].sum()
                total_decoded_tokens = df_ps["num_decoded_tokens"].sum()

                initial_times = df_ps["initial_time"]
                final_ttft_times = initial_times + df_ps["ttft"]
                decode_end_times = df_ps["final_time"]

                total_ttft_time = calculate_real_time(initial_times, final_ttft_times)
                total_decode_time = calculate_real_time(final_ttft_times, decode_end_times)
                total_time = df_ps["final_time"].max() - df_ps["initial_time"].min()
                
                total_throughput = (total_prompt_tokens + total_decoded_tokens) / total_time if total_time > 0 else 0.0
                prefill_throughput = total_prompt_tokens / total_ttft_time if total_ttft_time > 0 else 0.0
                decode_throughput = total_decoded_tokens / total_decode_time if total_decode_time > 0 else 0.0

                if total_throughput > max_total_throughput:
                    max_total_throughput = total_throughput
                    best_prefill_thp = prefill_throughput
                    best_decode_thp = decode_throughput
                    best_model = df_ps["model_alias"].iloc[0] if "model_alias" in df_ps.columns else "Unknown"
                    avg_prompt_size = df_ps["prompt_size"].mean()
                    decoded_tokens = df_ps["num_decoded_tokens"].mean()
                    best_req_per_sec = req_per_sec

            table_rows.append({
                "Experiment": exp,
                "Model": best_model if best_model else "Unknown",
                "Max Throughput (tokens/s)": round(max_total_throughput, 2),
                "Prefill Throughput (tokens/s)": round(best_prefill_thp, 2),
                "Decode Throughput (tokens/s)": round(best_decode_thp, 2),
                "Avg Prompt size": round(avg_prompt_size, 2) if avg_prompt_size else "N/A",
                "Avg Decoded Tokens per prompt": round(decoded_tokens, 2) if decoded_tokens else "N/A",
                "Req/s": best_req_per_sec if best_req_per_sec else "N/A",
                "Best batch (tok/s)": round(best_req_per_sec * avg_prompt_size, 2) if best_req_per_sec and avg_prompt_size else "N/A"
            })

        headers = list(table_rows[0].keys())
        rows_list = [[r[h] for h in headers] for r in table_rows]
        return tabulate(rows_list, headers, tablefmt="github")

    # --- 1. Tabela completa ---
    table_md_full = build_table(df)

    # --- 2. Tabela filtrada por e2e_goal ---
    filtered_df = filter_by_single_goal(df, "e2e_goal", args.e2e_goal)
    table_md_filtered = build_table(filtered_df)

    goal_title = f"E2E < {args.e2e_goal}s" if args.e2e_goal else "No filter"
    goal_tables = [(goal_title, table_md_filtered)]

    return plots, table_md_full, goal_tables

def main():
    args = parse_args()
    
    for experiment in args.experiments:
        os.makedirs(os.path.join(args.output_path, experiment), exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    df = load_data(args.csv_path)

    if args.experiment_is_group:
        if "experiment_group" not in df.columns:
            print("Warning: 'experiment_group' column not found in CSV; proceeding without grouping filter.")
        else:
            df = df[df["experiment_group"].isin(args.experiments)]
            args.experiments = list(df["experiment_key"].unique())
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
