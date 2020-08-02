import pickle

import numpy as np
import pymia.data.assembler as asmbl
import pymia.data.indexexpression as expr


def on_sample_ensure_index_expression_validity(params: dict):
    """Ensures the validity of index expressions and the data for array slicing.

    This callback can be used in case :py:class:`PatchWiseIndexing` is used with argument `ignore_incomplete=True`.
    Note that currently only the upper boundaries are checked as it is implemented in the :py:class:`PatchWiseIndexing`.
    """
    key = '__prediction'
    data = params[key]
    idx = params['batch_idx']
    batch = params['batch']
    predictions = params['predictions']

    subject_index = batch['subject_index'][idx]

    index_expr = batch['index_expr'][idx]
    if isinstance(index_expr, bytes):
        index_expr = pickle.loads(index_expr)

    valid_index_expr = []
    is_valid = True
    for idx, slicer in enumerate(index_expr.expression):
        if type(slicer) == slice:
            if slicer.stop > predictions[subject_index][key].shape[idx]:
                valid_stop = predictions[subject_index][key].shape[idx]
                is_valid = False
            else:
                valid_stop = slicer.stop
            valid_index_expr.append([slicer.start, valid_stop])
        else:
            break

    if is_valid:
        return data, index_expr
    else:
        valid_index_expr = expr.IndexExpression(valid_index_expr)
        valid_data = data[0:valid_index_expr.expression[1].stop - valid_index_expr.expression[1].start,
                     0:valid_index_expr.expression[2].stop - valid_index_expr.expression[2].start,
                     :]
        return valid_data, valid_index_expr


def init_subject_assembler():
    return asmbl.SubjectAssembler(zero_fn=lambda shape, id_, batch, idx: np.zeros(shape, np.float32),
                                  on_sample_fn=on_sample_ensure_index_expression_validity)
