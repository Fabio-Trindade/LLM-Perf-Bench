from src.data_structures.csv_data_format import CSVDataFormat
from src.data_structures.device.accelerator_data import AcceleratorData
from src.data_structures.data_format import DataFormat
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import numpy as np

class SinglePromptReporter():
    @staticmethod
    def report_single_prompt_perf_md(lines: list, config, data_list: list[DataFormat], prompt_sizes: list[int], model_: str):
        output_dir = config.path_to_save_results + config.experiment_key
        filename = "report.md"
        os.makedirs(output_dir, exist_ok=True)
        markdown_path = Path(output_dir) / filename
        image_dir = Path(output_dir) / "images"
        image_dir.mkdir(exist_ok=True)

        def save_violin_plot_grouped(values_dict, title, image_name):
            all_data = []
            all_labels = []
            for label, values in values_dict.items():
                flat = [v for sub in values for v in (sub if isinstance(sub, list) else [sub])]
                all_data.extend(flat)
                all_labels.extend([label] * len(flat))
            plt.figure(figsize=(8, 5))
            sns.violinplot(x=all_labels, y=all_data)
            plt.title(title)
            plt.xlabel("Prompt Size")
            plt.ylabel("TBT")
            image_path = image_dir / image_name
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            return image_path

        def save_line_plot(x, y_dict, title, image_name, xlabel="Prompt Size", ylabel="Value"):
            plt.figure(figsize=(6, 4))
            for label, y in y_dict.items():
                plt.plot(x, y, marker='o', label=label)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            image_path = image_dir / image_name
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            return image_path

        lines.append(f"# 📊 Performance Report - Single Prompt Inference\n")
        lines.append(f"## Configuration\n")
        lines.append(f"**Model**: `{model_}`\n")
        lines.append(f"**Backend**: `{config.requester}`\n")

        tbt_distributions = {}
        summary_metrics = { "THP": [], "TTFT": [], "E2ET": [], "DT": [] }
        key_map =  { "THP": "Throughput", 
                    "TTFT": "Time To First Token", 
                    "E2ET":"End To End Time", 
                    "DT": "Decode Time"}
        tbt_stats = { "median": [], "p99": [] }

        # Armazenar métricas por entrada
        throughput_rows = []
        detailed_rows = []

        for size, df in zip(prompt_sizes, data_list):
            # TBTs
            tbt_values = df.prompt_distributions.TBT
            flat = [v for sub in tbt_values for v in (sub if isinstance(sub, list) else [sub])]
            tbt_distributions.setdefault(size, []).extend(flat)
            tbt_stats["median"].append(np.median(flat))
            tbt_stats["p99"].append(np.percentile(flat, 99))

            # Métricas resumidas
            for key in summary_metrics:
                if  key == "TTFT":
                    summary_metrics[key].append(df.total_prefill_time)
                elif key == "DT":
                    summary_metrics[key].append(df.total_decode_time)
                elif key == "E2ET":
                    summary_metrics[key].append(df.total_time)
                elif key == "THP" :
                    summary_metrics[key].append(df.total_throughput)

            # Tabelas (sem média)
            throughput_rows.append([
                size,
                df.total_decode_tokens,
                df.total_time,
                df.total_throughput
            ])
            detailed_rows.append([
                size,
                df.total_prefill_time,
                df.total_prefill_throughput,
                df.total_decode_tokens,
                df.total_decode_time,
                df.total_decode_throughput
            ])

        # Tabela de throughput
        lines.append("## 🚀 Throughput and Time by Prompt Size\n")
        lines.append("| Prompt Size | Output Tokens | Total Time (s) | Throughput (tok/s) |")
        lines.append("|-------------|----------------|----------------|---------------------|")
        for row in throughput_rows:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]:.2f} | {row[3]:.2f} |")

        # Tabela detalhada de prefill/decode
        lines.append("\n## 🔍 Prefill & Decode Details by Prompt Size\n")
        lines.append("| Prompt Size | Prefill Time (s) | Prefill Throughput (tok/s) | Output Tokens | Decode Time (s) | Decode Throughput (tok/s) |")
        lines.append("|-------------|------------------|-----------------------------|----------------|------------------|-----------------------------|")
        for row in detailed_rows:
            lines.append(f"| {row[0]} | {row[1]:.2f} | {row[2]:.2f} | {row[3]} | {row[4]:.2f} | {row[5]:.2f} |")

        # Gráficos de métricas
        lines.append("\n## 📉 Summary Metrics by Prompt Size\n")
        for metric, values in summary_metrics.items():
            metric = key_map[metric]
            img_path = save_line_plot(prompt_sizes, {metric: values}, f"{metric} vs Prompt Size", f"line_{metric}.png")
            lines.append(f"### {metric}\n")
            lines.append(f"![{metric} line plot]({img_path.relative_to(markdown_path.parent)})\n")

        # Estatísticas TBT
        lines.append("## 📏 Time Between Tokens Statistics by Prompt Size\n")
        img_path = save_line_plot(
            prompt_sizes,
            {
                "Median": tbt_stats["median"],
                "P99": tbt_stats["p99"]
            },
            "Time Between Tokens Statistics",
            "line_TBT_stats.png",
            ylabel="TBT"
        )
        lines.append(f"![TBT Stats line plot]({img_path.relative_to(markdown_path.parent)})\n")

        # Violin Plot
        lines.append("## 🎻 Time Between Tokens Distributions by Prompt Size\n")
        img_path = save_violin_plot_grouped(tbt_distributions, "TBT Distributions", "grouped_violin_TBT.png")
        lines.append(f"![Grouped TBT Violin Plot]({img_path.relative_to(markdown_path.parent)})\n")
        
       
    def save_line_plot(
        x: list,
        series: dict[str, list[float]],
        title: str,
        filename: str,
        xlabel: str = "Time (s)",
        ylabel: str = "Value",
        output_dir: Path = Path("plots")
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        plt.figure(figsize=(10, 6))

        for label, y_values in series.items():
            if len(y_values) != len(x):
                print(f"Skipping '{label}' due to length mismatch (x: {len(x)}, y: {len(y_values)})")
                continue
            plt.plot(x, y_values, label=label)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        return filepath

    @staticmethod
    def report_host_usage_md(lines: list[str], all_host_data: list, prompt_sizes: list[int]):
        lines.append("## 🖥️ Host System Information and Usage\n")

        if all_host_data:
            host = all_host_data[0]
            lines.append(f"- CPU Model: `{host.cpu_name}`")
            lines.append(f"- Total CPUs: `{host.total_cpus}`")
            lines.append(f"- RAM Name: `{host.ram_mem_name}`")
            lines.append(f"- RAM Capacity (GB): `{host.ram_capacity:.2f}`\n")

        # Gera curvas separadas por prompt size
        cpu_curves = {
            f"Prompt Size {size}": data.cpu_consumption_over_time
            for size, data in zip(prompt_sizes, all_host_data)
        }
        ram_curves = {
            f"Prompt Size {size}": data.ram_consumption_over_time
            for size, data in zip(prompt_sizes, all_host_data)
        }

        # Plota CPU
        img_cpu = SinglePromptReporter.save_line_plot(
            x=list(range(len(next(iter(cpu_curves.values()))))),  # tempo (amostras)
            series=cpu_curves,
            title="CPU Usage Over Time by Prompt Size",
            filename="host_cpu_dist.png",
            ylabel="CPU Usage (%)"
        )
        lines.append(f"![CPU Usage Distribution]({img_cpu.relative_to(Path('.'))})\n")

        # Plota RAM
        img_ram = SinglePromptReporter.save_line_plot(
            x=list(range(len(next(iter(ram_curves.values()))))),
            series=ram_curves,
            title="RAM Usage Over Time by Prompt Size",
            filename="host_ram_dist.png",
            ylabel="RAM Usage (%)"
        )
        lines.append(f"![RAM Usage Distribution]({img_ram.relative_to(Path('.'))})\n")


    @staticmethod
    def report_accelerator_usage_md(lines: list[str], all_accelerator_data: list[AcceleratorData], prompt_sizes: list[int]):
        lines.append("## ⚡ Accelerator Information and Usage\n")

        if all_accelerator_data:
            accel = all_accelerator_data[0]
            lines.append(f"- Accelerator Name: `{accel.name}`")
            lines.append(f"- Number of Devices: `{accel.num_devices}`")
            lines.append(f"- Number of Nodes: `{accel.num_nodes}`")
            lines.append(f"- Accelerators per Node: `{accel.accelerators_per_node}`")
            lines.append(f"- VRAM per Device (MB): {', '.join(f'{v/(1024**2):.2f}' for v in accel.vram_per_device)}")
            lines.append(f"- VRAM per Node (MB): {', '.join(f'{v/(1024**2):.2f}' for v in accel.vram_per_node)}\n")

        # --- Power Distribution per GPU ---
        watt_series_by_device = {}
        energy_summary_by_prompt = {}

        for i, accel_data in enumerate(all_accelerator_data):
            prompt_label = f"Prompt {prompt_sizes[i]}"
            energy_summary_by_prompt[prompt_label] = {}

            for device_id, watt_list in accel_data.watt_per_device_distribution.items():
                label = f"{prompt_label} - GPU {device_id}"
                watt_series_by_device[label] = watt_list

                # Compute total energy: sum of watts over time (assuming 1s interval) → Wh
                total_energy_wh = sum(watt_list) / 3600  # W·s → Wh
                energy_summary_by_prompt[prompt_label][device_id] = total_energy_wh

        # Gráfico consumo de energia (W)
        if watt_series_by_device:
            length = min(len(v) for v in watt_series_by_device.values())
            img_watt = SinglePromptReporter.save_line_plot(
                x=list(range(length)),
                series={k: v[:length] for k, v in watt_series_by_device.items()},
                title="Power Usage Over Time by Device and Prompt",
                filename="accel_power_dist.png",
                ylabel="Watts (W)"
            )
            lines.append(f"![Power Usage]({img_watt.relative_to(Path('.'))})\n")

        # Relatório de energia total
        lines.append("### 🔋 Total Energy Consumed per Device (Wh)\n")
        for prompt, device_energy in energy_summary_by_prompt.items():
            for device_id, energy_wh in device_energy.items():
                lines.append(f"- {prompt} - GPU {device_id}: `{energy_wh:.2f} Wh`")

        # --- VRAM Distribution per GPU ---
        vram_series_by_device = {}
        for i, accel_data in enumerate(all_accelerator_data):
            for device_id, vram_list in accel_data.vram_per_device_distribution.items():
                label = f"Prompt {prompt_sizes[i]} - GPU {device_id}"
                vram_series_by_device[label] = [v / (1024 ** 2) for v in vram_list]  # bytes → MB

        # Gráfico uso de VRAM
        if vram_series_by_device:
            length = min(len(v) for v in vram_series_by_device.values())
            img_vram = SinglePromptReporter.save_line_plot(
                x=list(range(length)),
                series={k: v[:length] for k, v in vram_series_by_device.items()},
                title="VRAM Usage Over Time by Device and Prompt",
                filename="accel_vram_dist.png",
                ylabel="VRAM (MB)"
            )
            lines.append(f"![VRAM Usage]({img_vram.relative_to(Path('.'))})\n")
