import numpy as np

from .mask import tensor_to_numpy, normalize_mask


def predict_with_reference(model, target_image):

    model.set_target(target_image)

    pred = model.segment()

    pred = tensor_to_numpy(pred)

    pred = normalize_mask(pred)

    return pred


def generate_class_mask(model, references, target_image):

    if len(references) == 0:
        return None

    for ref_img, ref_mask in references:
        model.set_reference(str(ref_img), str(ref_mask))


    pred = predict_with_reference(model, target_image)

    final_mask = (pred).astype(np.uint8)

    return final_mask
