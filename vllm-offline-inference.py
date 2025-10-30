import sys
import json
import os
import importlib.util
from vllm import LLM, SamplingParams
from datetime import datetime
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

def load_config() -> Dict[str, Any]:
    """从环境变量加载配置。"""
    config = {
        # 模型参数
        'model_path': os.environ.get('VLLM_MODEL_PATH', 'meta-llama/Meta-Llama-3-8B-Instruct'),
        'tensor_parallel_size': int(os.environ.get('VLLM_TENSOR_PARALLEL_SIZE', 1)),
        'pipeline_parallel_size': int(os.environ.get('VLLM_PIPELINE_PARALLEL_SIZE', 1)),
        'gpu_memory_utilization': float(os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', 0.9)),
        'dtype': os.environ.get('VLLM_DTYPE', 'auto'),
        'max_model_len': int(os.environ.get('VLLM_MAX_MODEL_LEN', 4096)),
        'seed': int(os.environ.get('VLLM_SEED', 42)),
        'enable_prefix_caching': os.environ.get('VLLM_ENABLE_PREFIX_CACHING', 'false').lower() == 'true',
        'block_size': int(os.environ.get('VLLM_BLOCK_SIZE', 16)),
        'max_num_batched_tokens': int(os.environ.get('VLLM_MAX_NUM_BATCHED_TOKENS', 8192)),
        'trust_remote_code': os.environ.get('VLLM_TRUST_REMOTE_CODE', 'false').lower() == 'true',
        
        # 推理参数
        'temperature': float(os.environ.get('VLLM_TEMPERATURE', 0.7)),
        'top_p': float(os.environ.get('VLLM_TOP_P', 0.95)),
        'top_k': int(os.environ.get('VLLM_TOP_K', -1)),
        'max_tokens': int(os.environ.get('VLLM_MAX_TOKENS', 512)),
        'repetition_penalty': float(os.environ.get('VLLM_REPETITION_PENALTY', 1.0)),
        
        # 输入/输出路径
        'input_file': os.environ.get('INPUT_FILE', '/input/requests.jsonl'),
        'output_file': os.environ.get('OUTPUT_FILE', '/output/results.jsonl'),
        
        # 批处理大小
        'batch_size': int(os.environ.get('VLLM_BATCH_SIZE', 32)),
        
        # Guided decoding 文件路径
        'guided_decoding_file': os.environ.get('VLLM_GUIDED_DECODING_FILE', '/input/guided_decoding.py'),
        
        # 是否使用 chat template
        'use_chat_template': os.environ.get('VLLM_USE_CHAT_TEMPLATE', 'true').lower() == 'true',
    }
    return config

def convert_pydantic_to_schema(pydantic_model):
    """将 Pydantic Model 转换为 JSON Schema。只支持 Pydantic v2。"""
    if hasattr(pydantic_model, 'model_json_schema'):
        return pydantic_model.model_json_schema()
    else:
        raise ValueError("仅支持 Pydantic v2 的 model_json_schema 方法。确保 Pydantic v2 已安装并正确定义模型。")

def load_guided_decoding(file_path: str) -> Optional[Dict[str, Any]]:
    """
    加载 guided_decoding 参数。
    
    支持三种定义方式：
    
    1. Pydantic Model (推荐):
       from pydantic import BaseModel
       class MyModel(BaseModel):
           name: str
           age: int
       guided_decoding_params = MyModel
    
    2. JSON Schema:
       guided_decoding_params = {
           "type": "object",
           "properties": {
               "name": {"type": "string"},
               "age": {"type": "integer"}
           }
       }
    
    3. Choice (分类):
       guided_decoding_params = ['positive', 'negative', 'neutral']
    """
    logger = logging.getLogger(__name__)
    if not file_path or not os.path.exists(file_path):
        logger.info(f"ℹ 未找到 guided_decoding 文件: {file_path}")
        return None
    
    abs_path = os.path.abspath(file_path)
    if not abs_path.startswith('/input/'):
        logger.warning(f"⚠ guided_decoding 文件路径 {abs_path} 不以 /input/ 开头，出于安全考虑，已跳过加载。")
        return None
    
    try:
        # 如果是 JSON 文件，优先安全加载作为 dict
        if file_path.lower().endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            if isinstance(params, dict):
                logger.info("✓ 从 JSON 文件加载 guided_decoding_params 作为 JSON Schema")
                return {'json': params}
            elif isinstance(params, list):
                logger.info(f"✓ 从 JSON 文件加载 guided_decoding_params 作为 Choice: {len(params)} 个选项")
                return {'choice': params}
            else:
                logger.warning(f"⚠ JSON 文件内容不支持的类型: {type(params)}")
                return None
        
        # 否则，加载 Python 文件（有风险）
        logger.warning("⚠ 加载外部Python代码可能有安全风险，确保文件来源可信。")
        spec = importlib.util.spec_from_file_location("guided_decoding_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'guided_decoding_params'):
            logger.warning(f"⚠ {file_path} 中未找到 'guided_decoding_params'")
            return None
        
        params = module.guided_decoding_params
        
        # 尝试 Pydantic v2 支持
        try:
            from pydantic import BaseModel
            if isinstance(params, type) and issubclass(params, BaseModel):
                logger.info("✓ 检测到 Pydantic Model，转换为 JSON Schema (v2)")
                schema = convert_pydantic_to_schema(params)
                return {'json': schema}
        except ImportError:
            logger.warning("⚠ Pydantic 未安装，跳过 Pydantic Model 支持。只支持 JSON Schema 或 Choice。")
        except ValueError as ve:
            logger.warning(f"⚠ Pydantic Model 转换失败: {ve}")
        
        # 情况 2: 列表 (Choice)
        if isinstance(params, list):
            logger.info(f"✓ 检测到 Choice 模式: {len(params)} 个选项")
            return {'choice': params}
        
        # 情况 3: 字典 (JSON Schema)
        if isinstance(params, dict):
            logger.info("✓ 检测到 JSON Schema")
            return {'json': params}
        
        logger.warning(f"⚠ 不支持的 guided_decoding_params 类型: {type(params)}")
        return None
        
    except Exception as e:
        logger.exception(f"⚠ 加载 guided_decoding 失败: {e}")
        return None

def load_requests(input_file: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载请求。"""
    logger = logging.getLogger(__name__)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    requests = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if 'messages' not in req:
                    logger.warning(f"⚠ 第 {line_num} 行缺少 'messages' 字段，已跳过")
                    continue
                if 'custom_id' not in req:
                    req['custom_id'] = f'request-{line_num}'
                requests.append(req)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠ 第 {line_num} 行 JSON 格式错误: {e}")
    
    if not requests:
        raise ValueError("输入文件中没有有效请求")
    
    logger.info(f"✓ 加载了 {len(requests)} 个请求\n")
    return requests

def initialize_llm(config: Dict[str, Any]) -> LLM:
    """初始化 vLLM 模型。"""
    logger = logging.getLogger(__name__)
    logger.info(f"{'='*60}")
    logger.info(f"初始化模型: {config['model_path']}")
    logger.info(f"Tensor Parallel: {config['tensor_parallel_size']}, "
          f"GPU Memory: {config['gpu_memory_utilization']}")
    logger.info(f"{'='*60}\n")
    
    llm = LLM(
        model=config['model_path'],
        tensor_parallel_size=config['tensor_parallel_size'],
        pipeline_parallel_size=config['pipeline_parallel_size'],
        gpu_memory_utilization=config['gpu_memory_utilization'],
        dtype=config['dtype'],
        max_model_len=config['max_model_len'],
        seed=config['seed'],
        enable_prefix_caching=config['enable_prefix_caching'],
        block_size=config['block_size'],
        max_num_batched_tokens=config['max_num_batched_tokens'],
        trust_remote_code=config['trust_remote_code']
    )
    logger.info("✓ 模型初始化完成\n")
    return llm

def build_prompt(messages: List[Dict[str, str]], llm: LLM, use_chat_template: bool) -> str:
    """构建提示词。"""
    logger = logging.getLogger(__name__)
    if use_chat_template:
        try:
            tokenizer = llm.get_tokenizer()
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"⚠ apply_chat_template 失败，回退到简单拼接: {e}")
    # 简单拼接
    return ''.join(f"{msg['role']}: {msg['content']}\n" for msg in messages)

def create_sampling_params(req: Dict[str, Any], default_params: Dict[str, Any],
                          guided_decoding: Optional[Dict[str, Any]]) -> SamplingParams:
    """创建采样参数。"""
    params = {
        'temperature': req.get('temperature', default_params['temperature']),
        'top_p': req.get('top_p', default_params['top_p']),
        'top_k': req.get('top_k', default_params['top_k']),
        'max_tokens': req.get('max_tokens', default_params['max_tokens']),
        'repetition_penalty': req.get('repetition_penalty', default_params['repetition_penalty']),
    }
    
    # 添加 guided decoding
    if guided_decoding:
        params.update(guided_decoding)
    
    # 可选参数
    for key in ['stop', 'stop_token_ids']:
        if key in req:
            params[key] = req[key]
    
    return SamplingParams(**params)

def process_batch(llm: LLM, batch: List[Dict[str, Any]], default_params: Dict[str, Any],
                 guided_decoding: Optional[Dict[str, Any]], model_path: str,
                 use_chat_template: bool) -> List[Dict[str, Any]]:
    """批量处理请求。"""
    logger = logging.getLogger(__name__)
    prompts = []
    sampling_params_list = []
    
    # 准备数据
    for req in batch:
        try:
            prompt = build_prompt(req['messages'], llm, use_chat_template)
            sampling_params = create_sampling_params(req, default_params, guided_decoding)
            prompts.append(prompt)
            sampling_params_list.append(sampling_params)
        except Exception as e:
            logger.warning(f"⚠ 请求 {req['custom_id']} 准备失败: {e}")
            prompts.append("")
            sampling_params_list.append(SamplingParams(max_tokens=1))
    
    # 批量推理
    try:
        outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)
    except Exception as e:
        logger.error(f"✗ 批量推理失败: {e}")
        outputs = [None] * len(batch)
    
    # 构建结果
    results = []
    for req, output in zip(batch, outputs):
        custom_id = req['custom_id']
        result = {
            'id': f'vllm-{custom_id}',
            'custom_id': custom_id,
            'response': {},
            'error': None
        }
        
        if output is None:
            result['error'] = {'message': '推理失败', 'type': 'InferenceError'}
        else:
            try:
                if not output.outputs:
                    raise ValueError("vLLM 输出为空（可能由于生成参数或错误）")
                result['response'] = {
                    'id': f'cmpl-{custom_id}',
                    'object': 'chat.completion',
                    'created': int(datetime.now().timestamp()),
                    'model': model_path,
                    'choices': [{
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': output.outputs[0].text
                        },
                        'finish_reason': output.outputs[0].finish_reason
                    }],
                    'usage': {
                        'prompt_tokens': len(output.prompt_token_ids),
                        'completion_tokens': len(output.outputs[0].token_ids),
                        'total_tokens': len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
                    }
                }
            except Exception as e:
                result['error'] = {'message': str(e), 'type': type(e).__name__}
        
        results.append(result)
    
    return results

def write_results(output_file: str, results: List[Dict[str, Any]]):
    """写入结果。"""
    logger = logging.getLogger(__name__)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    success = sum(1 for r in results if r['error'] is None)
    failed = len(results) - success
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ 推理完成")
    logger.info(f"  成功: {success} / {len(results)}")
    if failed > 0:
        logger.info(f"  失败: {failed}")
    logger.info(f"  输出: {output_file}")
    logger.info(f"{'='*60}\n")

def main():
    """主函数。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/output/inference.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        config = load_config()
        guided_decoding = load_guided_decoding(config['guided_decoding_file'])
        requests = load_requests(config['input_file'])
        llm = initialize_llm(config)
        
        default_params = {
            'temperature': config['temperature'],
            'top_p': config['top_p'],
            'top_k': config['top_k'],
            'max_tokens': config['max_tokens'],
            'repetition_penalty': config['repetition_penalty'],
        }
        
        # 批量处理
        results = []
        total_batches = math.ceil(len(requests) / config['batch_size'])
        
        logger.info(f"开始推理 (批大小: {config['batch_size']})...\n")
        
        for i in range(0, len(requests), config['batch_size']):
            batch = requests[i:i + config['batch_size']]
            batch_num = i // config['batch_size'] + 1
            
            logger.info(f"[{batch_num}/{total_batches}] 处理 {len(batch)} 个请求...")
            
            batch_results = process_batch(
                llm, batch, default_params, guided_decoding,
                config['model_path'], config['use_chat_template']
            )
            results.extend(batch_results)
            logger.info("✓")
        
        write_results(config['output_file'], results)
        
    except Exception as e:
        logger.exception(f"\n✗ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# ============================================================================
# guided_decoding.py 定义示例
# ============================================================================

"""
方式 1: Pydantic Model (推荐，类型安全)
----------------------------------------
from pydantic import BaseModel, Field
from typing import List, Literal

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
    keywords: List[str]

guided_decoding_params = SentimentResult


方式 2: JSON Schema (灵活，精确控制)
----------------------------------------
guided_decoding_params = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["sentiment", "confidence", "keywords"]
}


方式 3: Choice (简单分类)
----------------------------------------
guided_decoding_params = ['positive', 'negative', 'neutral']


注意：
- Pydantic 方式最推荐，支持 v2
- JSON Schema 方式最灵活
- Choice 方式最简单，适合分类任务
- 只需要定义 guided_decoding_params 变量
- 现在支持 .json 文件作为输入（更安全）
"""
