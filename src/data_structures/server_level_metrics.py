from src.buffers.performance.performance_metrics_buffer import PerformanceMetricsBuffer


class ServerLevelMetrics():
    def __init__(self, e2e_time, total_processed_tokens):
        self.e2e_time = e2e_time
        self.e2e_throughput = total_processed_tokens/e2e_time
    