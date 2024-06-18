def compute_padding_1d(in_: int, *, out_=None, min_div=1):
    """Returns tuples for padding and unpadding.

    Args:
        in_: Input length.
        out_: Output length.
        min_div: Length that output dimensions should be divisible by.
    """
    if out_ is None:
        out_ = (in_ + min_div - 1) // min_div * min_div

    if out_ % min_div != 0:
        raise ValueError(f"Padded output length is not divisible by min_div={min_div}.")

    left = (out_ - in_) // 2
    right = out_ - in_ - left

    pad = (left, right)
    unpad = (-left, -right)

    return pad, unpad
