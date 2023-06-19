from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import multi_gpu_test, single_gpu_test 
from .test_multihead import multihead_test_save
from .test_chg import change_test_save
from .test_chg_mtl import change_test_mtl_save
from .test_chg_mtl_prob import change_test_prob_save
from .test_chg_mtl_index import change_test_mtl_index_save
from .test_reg import change_test_reg_save
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_segmentor)
from .test_chg_sr import change_sr_save
__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', 'single_gpu_test',
    'multihead_test_save', 'change_test_save',
    'show_result_pyplot','change_test_mtl_save'
]
