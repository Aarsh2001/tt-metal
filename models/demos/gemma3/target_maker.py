import pandas as pd
from tracy.process_model_log import get_latest_ops_log_filename

from models.demos.llama3_70b_galaxy.tests.test_prefill_device_perf import (
    average_per_instance_dict,
    build_duration_dict,
    build_duration_per_instance_dict,
    max_per_instance_dict,
    merge_device_rows,
    min_per_instance_dict,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf


def compare_with_target(kernel_duration_per_instance_averaged_dict, perf_targets, profiler):
    benchmark_data = BenchmarkData()
    passing = True
    for op_index, op_code_with_id in enumerate(kernel_duration_per_instance_averaged_dict.keys()):
        if op_code_with_id in perf_targets:
            op_name = perf_targets[op_code_with_id]["op_name"]

            avg_kernel_duration = kernel_duration_per_instance_averaged_dict[op_code_with_id]
            min_kernel_duration = kernel_duration_per_instance_averaged_dict[op_code_with_id]
            max_kernel_duration = kernel_duration_per_instance_max_dict_trace[op_code_with_id]

            # average
            benchmark_data.add_measurement(profiler, 0, step_name, op_name + "-model-kernel-avg", avg_kernel_duration)
            benchmark_data.add_measurement(
                profiler, 0, step_name, op_name + "-model-op_to_op-avg", avg_dispatch_duration
            )

            # min
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-kernel-min",
                min_kernel_duration,
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-op_to_op-min",
                dispatch_duration_per_instance_min_dict[op_code_with_id],
            )

            # max
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-kernel-max",
                max_kernel_duration,
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-op_to_op-max",
                dispatch_duration_per_instance_max_dict[op_code_with_id],
            )

            # Verify kernel duration is within tolerance
            upper_limit = (
                perf_targets[op_code_with_id]["kernel_duration"]
                + perf_targets[op_code_with_id]["kernel_duration_relative_margin"]
                * perf_targets[op_code_with_id]["kernel_duration"]
            )
            lower_limit = (
                perf_targets[op_code_with_id]["kernel_duration"]
                - perf_targets[op_code_with_id]["kernel_duration_relative_margin"]
                * perf_targets[op_code_with_id]["kernel_duration"]
            )

            if avg_kernel_duration > upper_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns is larger than target "
                    f"({perf_targets[op_code_with_id]['kernel_duration']}) ns, difference: "
                    f"{abs(avg_kernel_duration - upper_limit)} ns, margin: "
                    f"{perf_targets[op_code_with_id]['kernel_duration_relative_margin']}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id]['kernel_duration'] - avg_kernel_duration) / perf_targets[op_code_with_id]['kernel_duration']}"
                )
            elif avg_kernel_duration < lower_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns is smaller than target "
                    f"({perf_targets[op_code_with_id]['kernel_duration']}) ns, difference: "
                    f"{abs(lower_limit - avg_kernel_duration)} ns, margin: "
                    f"{perf_targets[op_code_with_id]['kernel_duration_relative_margin']}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id]['kernel_duration'] - avg_kernel_duration) / perf_targets[op_code_with_id]['kernel_duration']}"
                )
        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in perf_targets")

    assert passing, "One or more ops did not meet performance targets. Check logs for details."


def test():
    profiler = BenchmarkProfiler()
    batch_size = 1
    subdir = f"ttnn_gemma_cross_attention_perf"
    num_iterations = 1
    command = f"pytest models/demos/gemma3/tests/test_vision_cross_attention_transformer.py::test_gemma_vision"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    profiler.start("run")
    profiler.start("PROFILLING OP TO OP")

    all_results_average = []
    all_results_max = []
    all_results_min = []

    # Run tracy 3 times and make target out of it
    for i in range(3):
        post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=False)
        profiler.end("PROFILLING OP TO OP")
        profiler.end("run")

        filename = get_latest_ops_log_filename(subdir)
        df = pd.read_csv(filename)
        df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
        df = merge_device_rows(df)

        ops_raw_dict = df[["OP CODE", "DEVICE KERNEL DURATION [ns]"]].to_dict(orient="records")
        kernel_duration_dict = build_duration_dict(ops_raw_dict, "DEVICE KERNEL DURATION [ns]")
        kernel_duration_per_instance_dict = build_duration_per_instance_dict(kernel_duration_dict, 1)
        # Min over all iterations of each op instance
        kernel_duration_per_instance_min_dict = min_per_instance_dict(kernel_duration_per_instance_dict)
        # Max over all iterations of each op instance
        kernel_duration_per_instance_max_dict = max_per_instance_dict(kernel_duration_per_instance_dict)
        # Average over all iterations of each op instance (in this specific case it is the same)
        kernel_duration_per_instance_averaged_dict = average_per_instance_dict(kernel_duration_per_instance_dict)

        all_results_average.append(kernel_duration_per_instance_averaged_dict)
        all_results_max.append(kernel_duration_per_instance_max_dict)
        all_results_min.append(kernel_duration_per_instance_min_dict)

    logger.info(f"Generated target kernel durations: {all_results_average}")

    # expected_perf_cols = {}
    # with open(f"models/demos/gemma3/tests/perf_targets/targets_test_perf_vision_cross_attention_ops.json", "r") as f:
    #     expected_perf_cols = json.load(f)
    # compare_with_target(kernel_duration_per_instance_averaged_dict, expected_perf_cols, profiler)
