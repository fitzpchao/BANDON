_base_ = [
    '../../configs/bandon/traindata/BANDON_c512_mtl.py',
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_40k_changeformer.py',
    '../../configs/bandon/models/change_former.py'
]

