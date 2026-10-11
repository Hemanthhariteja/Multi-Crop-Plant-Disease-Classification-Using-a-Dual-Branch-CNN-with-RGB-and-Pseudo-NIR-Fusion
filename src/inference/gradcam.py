import torch


def gradcam_simple(model, x, target_class=None):
    model.eval()
    features = []
    gradients = []

    def forward_hook(_, __, output):
        features.append(output)

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM")

    handle_f = last_conv.register_forward_hook(forward_hook)
    handle_b = last_conv.register_full_backward_hook(backward_hook)

    logits = model(x)
    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()

    loss = logits[0, target_class]
    model.zero_grad()
    loss.backward()

    fmap = features[0]
    grads = gradients[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-6)

    handle_f.remove()
    handle_b.remove()
    return cam
