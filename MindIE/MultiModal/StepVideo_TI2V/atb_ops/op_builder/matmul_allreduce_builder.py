import os
from .atb_builder import AtbOpBuilder


class MatmulAllreduceOpBuilder(AtbOpBuilder):
    OP_NAME = "matmul_allreduce"

    def __init__(self):
        super(MatmulAllreduceOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        atb_path = os.path.join(os.path.dirname(__file__), "../ops/csrc/atb")
        return [os.path.join(atb_path, 'matmul_allreduce.cpp'),
                os.path.join(atb_path, 'utils/atb_adapter.cpp')]