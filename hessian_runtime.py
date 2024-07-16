import time

import torch
import numpy as np

from model_compression_toolkit.core.common.hessian import TraceHessianRequest, HessianMode, HessianInfoGranularity
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.runner import core_runner

NUM_SAMPLES = 16
SINGLE = False  # single sample at each iteration (vs. Batch)
ALL_NODES = True  # single node per iteration (vs. all nodes)


def hessian_runtime(model, representative_data_gen, core_config, target_platform_capabilities,
                    target_resource_utilization, wandb_log):
    fw_info = DEFAULT_PYTORCH_INFO

    ptq_graph, _, hessian_service = core_runner(in_model=model,
                                                representative_data_gen=representative_data_gen,
                                                core_config=core_config,
                                                fw_info=fw_info,
                                                fw_impl=PytorchImplementation(),
                                                tpc=target_platform_capabilities,
                                                target_resource_utilization=target_resource_utilization)

    # Fetch hessian approximations for each target node
    approximations = {}

    compare_points = [n for n in ptq_graph.get_topo_sorted_nodes() if n.type == torch.nn.Conv2d]

    print(f"Starting to compute hessians for nodes with {NUM_SAMPLES} samples")
    start_time = time.time()
    if ALL_NODES:
        trace_hessian_request = TraceHessianRequest(
            mode=HessianMode.ACTIVATION,
            granularity=HessianInfoGranularity.PER_TENSOR,
            target_nodes=compare_points
        )
        node_approximations = hessian_service.fetch_hessian(
            trace_hessian_request=trace_hessian_request,
            required_size=NUM_SAMPLES,
            batch_size=NUM_SAMPLES,
        )
        for i, target_node in enumerate(compare_points):
            approximations[target_node] = node_approximations[i]
    else:
        for target_node in compare_points:
            trace_hessian_request = TraceHessianRequest(
                mode=HessianMode.ACTIVATION,
                granularity=HessianInfoGranularity.PER_TENSOR,
                target_nodes=target_node
            )
            node_approximations = hessian_service.fetch_hessian(
                trace_hessian_request=trace_hessian_request,
                required_size=NUM_SAMPLES
            )
            approximations[target_node] = node_approximations

    # Process the fetched hessian approximations to gather them per images
    if SINGLE:
        trace_hessian_approx_by_image = []
        for image_idx in range(NUM_SAMPLES):
            approx_by_interest_point = []
            for target_node in compare_points:
                trace_approx = approximations[target_node][image_idx]
                approx_by_interest_point.append(trace_approx[0])

            trace_hessian_approx_by_image.append(approx_by_interest_point)
    else:
        approx_by_interest_point = []
        for target_node in compare_points:
            trace_approx = approximations[target_node]
            # mean over images
            # TODO: need trace_approx to be a tensor not a list with tensors even for single image and singe node
            approx_by_interest_point.append(torch.mean(trace_approx).item())

    f = time.time() - start_time
    print("Total Hessian runtime: {:.2f} seconds ({:.2f} minutes)".format(f, f / 60))

    if SINGLE:
        res = np.mean(trace_hessian_approx_by_image, axis=0)
    else:
        res = approx_by_interest_point

    wandb_log(num_hessian_samples=NUM_SAMPLES, num_nodes=len(compare_points), total_runtime=f, hessian_res=res)

    exit()
