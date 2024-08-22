try:
    from .test_llm import skip_single_gpu
    from .test_llm_models import (baichuan2_7b_model_path, falcon_model_path,
                                  gemma_2b_model_path, gptj_model_path,
                                  llm_test_harness, qwen2_model_path,
                                  sampling_params)
except ImportError:
    from test_llm import skip_single_gpu
    from test_llm_models import (baichuan2_7b_model_path, falcon_model_path,
                                 gemma_2b_model_path, gptj_model_path,
                                 llm_test_harness, qwen2_model_path,
                                 sampling_params)


@skip_single_gpu
def test_llm_gptj_tp2():
    llm_test_harness(gptj_model_path,
                     prompts=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)


@skip_single_gpu
def test_llm_falcon_tp2():
    llm_test_harness(falcon_model_path,
                     prompts=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)


@skip_single_gpu
def test_llm_baichuan2_7b_tp2():
    llm_test_harness(baichuan2_7b_model_path,
                     prompts=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)


@skip_single_gpu
def test_llm_gemma_2b_tp2():
    llm_test_harness(gemma_2b_model_path,
                     prompts=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)


@skip_single_gpu
def test_llm_qwen2_tp2():
    llm_test_harness(qwen2_model_path,
                     prompts=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)
