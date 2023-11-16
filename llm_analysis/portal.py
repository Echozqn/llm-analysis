
import argparse
import json
import os.path

from llm_analysis.constant import *
# python portal.py --model_name open_llama_3b_v2 --gpu_name a100-pcie-40gb --tp_size 1 --pp_size 3 --sp_size 1 --dp_size 1 --gradient_accumulation_steps 4 -b 16 --seq_len 1400  --total_num_tokens 1280 --activation_recomputation 2 --flops_efficiency 0.43 --hbm_memory_efficiency 0.55 --mlp_recompute_gelu True  --output_dir ./data
# step1 启动llm-analysis
def read_json(filename):
    with open(filename, "r") as file:
        gpu_efficiencies = json.load(file)
    return gpu_efficiencies
def get_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add optional arguments with their default values
    parser.add_argument('--model_name', type=str, default="facebook_opt-1.3b",
                        help='model name to query the pre-defined model_configs dict, or model config json file path')
    parser.add_argument('--gpu_name', type=str, default="a100-sxm-40gb",
                        help='gpu name to query the pre-defined gpu_configs dict')
    parser.add_argument('--dtype_name', type=str, default="w16a16e16",
                        help='data type name to pre-defined dtype_configs dict')
    parser.add_argument('--log_level', type=str, default="INFO",
                        help='logging level')
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help='batch size per GPU (micro batch size)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='gradient accumulation steps')
    parser.add_argument('--global_batch_size', type=int, default=None,
                        help='global batch size')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='sequence length')
    parser.add_argument('--total_num_tokens', type=int, default=None,
                        help='total number of tokens used for training')
    parser.add_argument('--activation_recomputation', type=int, default=0,
                        help='activation recomputation strategy')
    parser.add_argument('--ds_zero', type=int, default=0,
                        help='DeepSpeed ZeRO stage to use')
    parser.add_argument('--dp_size', type=int, default=None,
                        help='data parallelism size')
    parser.add_argument('--tp_size', type=int, default=1,
                        help='tensor parallelism size')
    parser.add_argument('--pp_size', type=int, default=1,
                        help='pipeline parallelism size')
    parser.add_argument('--sp_size', type=int, default=None,
                        help='sequence parallelism size, defaults to tp_size')
    parser.add_argument('--ep_size', type=int, default=1,
                        help='expert parallelism size')
    parser.add_argument('--total_num_gpus', type=int, default=None,
                        help='total number of GPUs used for training')
    parser.add_argument('--layernorm_dtype_bytes', type=int, default=BYTES_FP32,
                        help='bytes in the data type for the layernorm activations')
    parser.add_argument('--master_weights_dtype_bytes', type=int, default=BYTES_FP32,
                        help='bytes in the data type for the optimizer master weights')
    parser.add_argument('--other_op_bytes', type=int, default=None,
                        help='bytes in the optimizer state')
    parser.add_argument('--flash_attn', action='store_true', default=True,
                        help='whether to use Flash Attention')
    parser.add_argument('--softmax_dropout', action='store_true', default=False,
                        help='whether to apply dropout after softmax')
    parser.add_argument('--mlp_activation_quant_bits', type=int, default=None,
                        help='bits to quantize MLP activations')
    parser.add_argument('--mlp_1linear_quant_bits', type=int, default=None,
                        help='bits to quantize the input activations of the first linear layer')
    parser.add_argument('--mlp_gelu_input_quant_bits', type=int, default=None,
                        help='bits to quantize the GELU input activations')
    parser.add_argument('--mlp_2linear_quant_bits', type=int, default=None,
                        help='bits to quantize the input activations of the second linear layer')
    parser.add_argument('--mlp_recompute_gelu', type=str, choices=["True", "False"], default="False",
                        help='whether to recompute the GELU activation in the MLP backward pass')
    parser.add_argument('--mlp_gated_linear_units', action='store_true', default=False,
                        help='whether to use gated linear units in the MLP')
    parser.add_argument('--achieved_tflops', type=float, default=None,
                        help='achieved TFLOPS per GPU')
    parser.add_argument('--flops_efficiency', type=float, default=None,
                        help='flops efficiency, ranging from 0 to 1')
    parser.add_argument('--hbm_memory_efficiency', type=float, default=HBM_MEMORY_EFFICIENCY,
                        help='GPU HBM memory efficiency, ranging from 0 to 1')
    parser.add_argument('--intra_node_memory_efficiency', type=float, default=INTRA_NODE_MEMORY_EFFICIENCY,
                        help='intra-node memory efficiency, ranging from 0 to 1')
    parser.add_argument('--inter_node_memory_efficiency', type=float, default=INTER_NODE_MEMORY_EFFICIENCY,
                        help='inter-node memory efficiency, ranging from 0 to 1')
    parser.add_argument('--num_gpus_per_node', type=int, default=NUM_GPUS_PER_NODE,
                        help='number of GPUs per node')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='directory path for writing the return summary dict')
    parser.add_argument('--output_detail_file_suffix', type=str, default="-test",
                        help='output_detail_file_suffix')

    # Parse the arguments
    args = parser.parse_args()
    # Read the JSON file
    gpu_efficiencies = read_json("gpu_efficiency.json")
    # Retrieve the efficiency details for the specified GPU type
    print(gpu_efficiencies,args.gpu_name)
    gpu_info = gpu_efficiencies.get(args.gpu_name)
    if gpu_info:
        args.flops_efficiency = gpu_info['flops_efficiency']
        args.hbm_memory_efficiency = gpu_info['hbm_memory_efficiency']
        print(f"FLOPS Efficiency: {args.flops_efficiency}")
        print(f"HBM Memory Efficiency: {args.hbm_memory_efficiency}")
    else:
        print(f"No data available for GPU type: {args.gpu_type}")

    def get_configs_desc(args) -> str:
        return (
            f"{args.model_name}-{args.gpu_name}-{args.dtype_name}"
            f"-tp{args.tp_size}-pp{args.pp_size}-dp{args.dp_size}"
            f"-sp{args.sp_size}-fe{round(args.flops_efficiency, 2)}"
            f"-ep{args.ep_size}-hbme{round(args.hbm_memory_efficiency, 2)}"
        )

    command_line = (
        f"python -m analysis train "
        f"--model_name {args.model_name} "
        f"--gpu_name {args.gpu_name} "
        f"--tp_size {args.tp_size} "
        f"--pp_size {args.pp_size} "
        f"--sp_size {args.sp_size} "
        f"--dp_size {args.dp_size} "
        f"--gradient_accumulation_steps {args.gradient_accumulation_steps} "
        f"-b {args.batch_size} "
        f"--seq_len {args.seq_len} "
        f"--total_num_tokens {args.total_num_tokens} "
        f"--activation_recomputation {int(args.activation_recomputation)} "
        f"--flops_efficiency {args.flops_efficiency} "
        f"--hbm_memory_efficiency {args.hbm_memory_efficiency} "
        f"--mlp_recompute_gelu {args.mlp_recompute_gelu} "
        f"--output_dir {args.output_dir} "
        f"--output_detail_file_suffix {args.output_detail_file_suffix} "
    )
    print(command_line)
    os.system(command_line)
    file_name = get_configs_desc(args) + args.output_detail_file_suffix + "-summary.json"
    # 从file_name 读取llm-analysis分析出来的东西，在放入pipeline_model 预测时间
    print(file_name)
    detail_info = read_json(os.path.join("./data",file_name))
    print(detail_info)





get_args()
"""
python portal.py --model_name open_llama_3b_v2 --gpu_name a100-pcie-40gb --tp_size 1 --pp_size 3 --sp_size 1 --dp_size 1 --gradient_accumulation_steps 4 -b 16 --seq_len 1400  --total_num_tokens 1280 --activation_recomputation 2 --flops_efficiency 0.43 --hbm_memory_efficiency 0.55 --mlp_recompute_gelu True --output_dir ./data --output_detail_file_suffix test 

127.0.0.1 www.parallels.cn
127.0.0.1 www.parallels.com
127.0.0.1 www.parallels.de
127.0.0.1 www.parallels.es
127.0.0.1 www.parallels.fr
127.0.0.1 www.parallels.nl
127.0.0.1 www.parallels.pt
127.0.0.1 www.parallels.ru
127.0.0.1 www.parallelskorea.com
127.0.0.1 reportus.parallels.com
127.0.0.1 parallels.cn
127.0.0.1 parallels.com
127.0.0.1 parallels.de
127.0.0.1 parallels.es
127.0.0.1 parallels.fr
127.0.0.1 parallels.nl
127.0.0.1 parallels.pt
127.0.0.1 parallels.ru
127.0.0.1 parallelskorea.com
127.0.0.1 pax-manager.myparallels.com
127.0.0.1 myparallels.com
127.0.0.1 my.parallels.com



"""


