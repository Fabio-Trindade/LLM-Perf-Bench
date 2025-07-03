from types import SimpleNamespace
from src.data_structures.data_format import DataFormat
from src.data_structures.prompt_performance_metrics import PPM
from src.data_structures.perf_results import PerfResults
from src.utils.util_list import flatten_list
class StatsGenerator():
    
    @staticmethod
    def gen_stats_for_single_prompt_from_csv(csv_pathname):
        raise ValueError("Should be implemented")
    
    @staticmethod
    #TODO: can be a general function for metric processing => change the name later
    def gen_single_prompt_data_from_metrics(perf_results: PerfResults) -> dict:
        tbt_metrics = perf_results.get_metrics(PPM.TBT)
        e2et_metrics = perf_results.get_metrics(PPM.E2ET)
        ttft_metrics = perf_results.get_metrics(PPM.TTFT)
        plen_metrics = list(flatten_list(perf_results.get_metrics(PPM.PLEN)))
        dlen_metrics = list(flatten_list(perf_results.get_metrics(PPM.DLEN)))

        prompt_keys = ["TBT", "E2ET", "TTFT", "MAX-TBT" ]
        all_metrics = [tbt_metrics, e2et_metrics,ttft_metrics]
        
        prompt_metrics  = [list(flatten_list(l)) for l in all_metrics]
        prompt_metrics.append([max(prompt_tbt) for req_tbt in tbt_metrics for prompt_tbt in req_tbt])
        build_simpleNS = lambda x,y : SimpleNamespace(**{name: metrics_list for name,metrics_list in zip(x,y)})
        
        req_keys = ["MAX-TBT", "MAX-E2ET", "MAX-TTFT", "MAX-THP"]
        req_metrics = [
            [max(flatten_list(req_m)) for req_m in metric]
            for metric in all_metrics
        ]

        total_decode_time = sum(prompt_metrics[0])
        total_prefill_time = sum(prompt_metrics[2])
        total_prefill_tokens = sum(plen_metrics)
        total_decode_tokens = sum(dlen_metrics)

        data = DataFormat(
                        prompt_distributions = build_simpleNS(prompt_keys,prompt_metrics),
                        request_distributions = build_simpleNS(req_keys, req_metrics) ,
                        total_prefill_tokens = total_prefill_tokens, 
                        total_decode_tokens= total_decode_tokens,
                        total_prefill_time = total_prefill_time,
                        total_decode_time = total_decode_time
                                       )              
        return data