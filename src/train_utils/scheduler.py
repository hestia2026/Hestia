import math

def get_custom_lr_lambda(
    total,
    stage_boundary_ratio=0.5,
    first_stage_scale=1,
    second_stage_scale=0.666,
    warmup=500,
):
    def lr_lambda(current_step: int):
        progress = current_step / total
        if progress < stage_boundary_ratio:
            if current_step < warmup:
                return first_stage_scale * (current_step / warmup)
            else:
                return first_stage_scale * (1 - (current_step - warmup) / total)
        else:
            return second_stage_scale * (1 - progress)

    return lr_lambda


def get_two_stage_cosine_lr_lambda(
    total,
    stage_boundary_ratio=0.5,
    first_stage_scale=1.0,
    second_stage_scale=0.667,
    min_lr_ratio = 0.006667,
    warmup=500,
):
    stage_boundary = int(total * stage_boundary_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup:
            return (current_step / warmup) * first_stage_scale

        progress = (current_step - warmup) / (total - warmup)
        if current_step < stage_boundary:
            cosine_decay = (1- min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)) + min_lr_ratio
            return first_stage_scale * cosine_decay
        else:
            cosine_decay = (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)) + min_lr_ratio
            return second_stage_scale * cosine_decay

    return lr_lambda


def get_wsd_lr_lambda(
    total: int,
    wsd_ratio: float = 0.1,
    min_lr_ratio: float = 0.05,
    warmup: int = 375,
):
    """
    Warm-Start Decay: keep lr at 1.0 after warmup, cosine decay only in last wsd_ratio of steps.
    """
    wsd_steps = max(1, int(total * wsd_ratio))
    decay_start = max(warmup, total - wsd_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup:
            return current_step / max(1, warmup)
        if current_step < decay_start:
            return 1.0
        progress = (current_step - decay_start) / max(1, total - decay_start)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return lr_lambda