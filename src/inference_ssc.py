import torch
from PIL import Image


def get_low_res_image(image, down_scale_factor=8):
    """Construct a degraded view by downsampling and upsampling the image."""
    original_size = image.size
    small_size = (
        original_size[0] // down_scale_factor,
        original_size[1] // down_scale_factor,
    )
    image_small = image.resize(small_size, Image.Resampling.BILINEAR)
    return image_small.resize(original_size, Image.Resampling.BILINEAR)


def safe_get_last_logit(output):
    logits = output.logits
    if logits.dim() == 3:
        return logits[:, -1, :]
    if logits.dim() == 2:
        return logits
    return logits.view(-1, logits.size(-1))[-1:].contiguous()


@torch.inference_mode()
def generate_ssc_logic(
    model,
    tokenizer,
    input_ids,
    image_tensor,
    image_tensor_low,
    alpha=1.0,
    K=50,
    epsilon=1e-12,
    max_new_tokens=512,
):
    device = input_ids.device
    current_ids = input_ids.clone()
    output_ids = []

    image_orig = image_tensor.unsqueeze(0).to(device, dtype=torch.float16)
    image_low = image_tensor_low.unsqueeze(0).to(device, dtype=torch.float16)

    past_orig, past_low = None, None

    for step in range(max_new_tokens):
        # Use the same text prefix for both views and maintain separate KV caches.
        if step == 0:
            out_orig = model(current_ids, images=image_orig, use_cache=True)
            out_low = model(current_ids, images=image_low, use_cache=True)
        else:
            out_orig = model(
                current_ids, use_cache=True, past_key_values=past_orig
            )
            out_low = model(
                current_ids, use_cache=True, past_key_values=past_low
            )

        past_orig = out_orig.past_key_values
        past_low = out_low.past_key_values

        P_orig = torch.softmax(safe_get_last_logit(out_orig), dim=-1)
        P_low = torch.softmax(safe_get_last_logit(out_low), dim=-1)

        # Select the Top-K candidates from the original-view distribution.
        topk_p_orig, topk_indices = torch.topk(P_orig, K, dim=-1)
        topk_p_low = torch.gather(P_low, dim=-1, index=topk_indices)
        M_K = topk_p_orig.sum(dim=-1, keepdim=True)

        # Compute the log-probability difference and saturated gain.
        Delta = (
            torch.log(torch.clamp(topk_p_orig, min=epsilon))
            - torch.log(torch.clamp(topk_p_low, min=epsilon))
        )
        Phi = torch.sign(Delta) * (1.0 - torch.exp(-torch.abs(Delta)))
        weights = 1.0 + alpha * Phi

        # Calibrate within Top-K while preserving its original probability mass.
        weighted_probs = topk_p_orig * weights
        Z = weighted_probs.sum(dim=-1, keepdim=True)
        calibrated_probs = M_K * weighted_probs / Z

        # Greedily select the highest-scoring calibrated Top-K candidate.
        best_position = torch.argmax(
            calibrated_probs, dim=-1, keepdim=True
        )
        next_token = torch.gather(
            topk_indices, dim=-1, index=best_position
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

        output_ids.append(next_token.item())
        current_ids = next_token

    return tokenizer.decode(
        output_ids, skip_special_tokens=True
    ).strip()
