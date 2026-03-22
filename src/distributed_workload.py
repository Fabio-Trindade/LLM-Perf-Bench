from copy import deepcopy
import os
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from typing import List, Tuple
from src.load_results import LoadResults
from src.data_structures.perf_results import PerfResults
from src.components.requesters.requester_interface import RequesterI
from src.workload_launcher import WorkloadLauncher
import time
import traceback

class DistributedWorkload:
    def __init__(self, config, requester: RequesterI, prompts: list):
        self.config = config
        self.requester = requester
        self.prompts = prompts

    @staticmethod
    async def run_single_async_workload(config, requester, prompts):
        launcher = WorkloadLauncher(config, requester, prompts)
        start = time.time()
        # print(f"[Async Workload] Started at {start:.2f}s")
        if hasattr(requester, "__aenter__"):
            async with requester:
                perf = await launcher.async_run()
        else:
            perf = await launcher.async_run()
        end = time.time()
        # print(f"[Async Workload] Finished at {end:.2f}s (Duration: {end-start:.2f}s)")
        host_data = launcher.get_host_data()
        acc_data = launcher.get_accelerator_data()
        return perf, host_data, acc_data

    @staticmethod
    def _thread_target_async(config, requester_cls, prompts):
        thread_name = multiprocessing.current_process().name
        start = time.time()
        # print(f"[Thread {thread_name}] Starting thread at {start:.2f}s")
        
        async def runner():
            requester = requester_cls(config)
            return await DistributedWorkload.run_single_async_workload(config, requester, prompts)
        
        result = asyncio.run(runner())
        end = time.time()
        # print(f"[Thread {thread_name}] Finished thread at {end:.2f}s (Duration: {end-start:.2f}s)")
        return result

    @staticmethod
    def _set_cpu_affinity(cpu_index: int):
        try:
            os.sched_setaffinity(0, {cpu_index})
        except Exception:
            pass

    @staticmethod
    def _process_target(proc_idx, threads_in_proc, configs_per_thread, requester_cls, prompts, return_dict, start_event):
        process_name = multiprocessing.current_process().name
        # print(f"[Process {proc_idx} - {process_name}] Ready, waiting to start")
        
        start_event.wait()
        start_time = time.time()
        # print(f"[Process {proc_idx} - {process_name}] Starting workload at {start_time:.2f}s")

        try:
            DistributedWorkload._set_cpu_affinity(proc_idx % multiprocessing.cpu_count())

            results = []
            futures = []
            with ThreadPoolExecutor(max_workers=threads_in_proc) as executor:
                for i in range(threads_in_proc):
                    futures.append(
                        executor.submit(
                            DistributedWorkload._thread_target_async,
                            configs_per_thread[i],
                            requester_cls,
                            prompts
                        )
                    )

                done, not_done = wait(futures, return_when=FIRST_EXCEPTION)

                for f in done:
                    exc = f.exception()
                    if exc:
                        for nf in not_done:
                            nf.cancel()
                        raise exc

                for f in done:
                    results.append(f.result())

            perf_agg = None
            host_all, acc_all = [], []
            for perf, host, acc in results:
                if perf_agg is None:
                    perf_agg = perf
                else:
                    try:
                        perf_agg.requests_metrics += perf.requests_metrics
                    except Exception:
                        pass
                host_all.extend(host if isinstance(host, list) else [host])
                acc_all.extend(acc if isinstance(acc, list) else [acc])

            return_dict[proc_idx] = (perf_agg, host_all, acc_all)

        except Exception as e:
             return_dict[proc_idx] = traceback.format_exc()
        finally:
            end_time = time.time()
            # print(f"[Process {proc_idx} - {process_name}] Finished at {end_time:.2f}s (Duration: {end_time-start_time:.2f}s)")

    @staticmethod
    def run_multiprocess_workload(config, requester_cls, prompts) -> Tuple[PerfResults, List, List]:
        total_requesters = config.concurrent_requesters
        cpu_count = multiprocessing.cpu_count()
        if total_requesters <= 0:
            raise ValueError("config.concurrent_requesters must be > 0")

        num_procs = min(cpu_count, total_requesters)
        base_threads = total_requesters // num_procs
        remainder = total_requesters % num_procs
        threads_per_proc = [base_threads + (1 if i < remainder else 0) for i in range(num_procs)]

        configs_all = []
        for proc_idx, threads_in_proc in enumerate(threads_per_proc):
            configs_per_thread = []
            for i in range(threads_in_proc):
                temp_config = deepcopy(config)
                setattr(temp_config, "seed", config.seed + config.concurrent_requesters * proc_idx + i)
                configs_per_thread.append(temp_config)
            configs_all.append(configs_per_thread)

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []
        start_event = multiprocessing.Event()

        for idx, threads_in_proc in enumerate(threads_per_proc):
            p = multiprocessing.Process(
                target=DistributedWorkload._process_target,
                args=(idx, threads_in_proc, configs_all[idx], requester_cls, prompts, return_dict, start_event)
            )
            processes.append(p)

        for p in processes:
            p.start()

        start_event.set()

        for p in processes:
            p.join()

        if not return_dict:
            raise RuntimeError("No results returned by child processes")

        for idx, res in return_dict.items():
            if isinstance(res, Exception):
                raise RuntimeError(f"[Process {idx}] failed: {res}")

        perf_agg = None
        host_all, acc_all = [], []
        for res in return_dict.values():
            perf, host, acc = res
            if perf_agg is None:
                perf_agg = perf
            else:
                try:
                    perf_agg.requests_metrics += perf.requests_metrics
                except Exception:
                    pass
            host_all.extend(host if isinstance(host, list) else [host])
            acc_all.extend(acc if isinstance(acc, list) else [acc])

        if perf_agg is None:
            raise RuntimeError("No performance results returned")

        return perf_agg, host_all, acc_all

    @staticmethod
    def run_param_loop(config, requester: RequesterI, prompts, loop_values=[None], var_to_set=None) -> LoadResults:
        results = LoadResults()
        for value in loop_values:
            if var_to_set is not None:
                setattr(config, var_to_set, value)
            perf, hosts, accs = DistributedWorkload.run_multiprocess_workload(config, type(requester), prompts)
            results.add_data(perf, hosts, accs)
        return results
