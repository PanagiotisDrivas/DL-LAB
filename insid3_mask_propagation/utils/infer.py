import numpy as np

from .mask import tensor_to_numpy, normalize_mask


def predict_with_reference(model, ref_img, ref_mask, target_image):

    model.set_reference(str(ref_img), str(ref_mask))

    model.set_target(target_image)

    pred = model.segment()

    pred = tensor_to_numpy(pred)

    pred = normalize_mask(pred)

    return pred


def generate_class_mask(model, references, target_image):

    if len(references) == 0:
        return None

    votes = None

    for ref_img, ref_mask in references:

        pred = predict_with_reference(model, ref_img, ref_mask, target_image)

        if votes is None:

            votes = np.zeros_like(pred, dtype=np.uint16)

        votes += pred

    threshold = len(references) // 2 + 1

    final_mask = (votes >= threshold).astype(np.uint8)

    return final_mask
