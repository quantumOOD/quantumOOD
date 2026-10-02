"""FGSM, PGD, SPSA, and Salt-and-Pepper adversarial attack implementations."""
import torch
import torch.nn.functional as F

from adversarial_classifier import preprocess_for_classifier


def _predict(model, dataset, images):
    with torch.no_grad():
        return model(preprocess_for_classifier(images, dataset)).argmax(dim=1)


def fgsm_attack(model, dataset, images, labels, epsilon=0.3):
    """Single-step white-box attack: x_adv = x + epsilon * sign(grad_x CE)."""
    images = images.clone().detach().requires_grad_(True)
    logits = model(preprocess_for_classifier(images, dataset))
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, images)[0]
    adv = torch.clamp(images.detach() + epsilon * grad.sign(), 0.0, 1.0)
    success = _predict(model, dataset, adv) != labels
    return adv.detach(), success


def pgd_attack(model, dataset, images, labels, epsilon=0.3, alpha=0.075, num_steps=10, random_start=True):
    """Iterative white-box attack: repeated FGSM-style steps, each
    projected back into the L-inf epsilon-ball around the original image
    and clipped to [0,1]."""
    orig = images.clone().detach()
    if random_start:
        adv = torch.clamp(orig + torch.empty_like(orig).uniform_(-epsilon, epsilon), 0.0, 1.0).detach()
    else:
        adv = orig.clone().detach()

    for _ in range(num_steps):
        adv.requires_grad_(True)
        logits = model(preprocess_for_classifier(adv, dataset))
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - orig, -epsilon, epsilon)
        adv = torch.clamp(orig + delta, 0.0, 1.0).detach()

    success = _predict(model, dataset, adv) != labels
    return adv, success


def spsa_attack(model, dataset, images, labels, epsilon=0.3, alpha=0.075, num_steps=10, num_samples=32, delta=0.01):
    """Black-box iterative attack: at each step, estimates the loss
    gradient via Simultaneous Perturbation Stochastic Approximation
    (average of (loss(x+delta*v) - loss(x-delta*v)) / (2*delta) * v over
    random Rademacher directions v), never backpropagating through the
    classifier -- only its scalar loss is queried. Otherwise the same
    sign-gradient-step-then-project structure as PGD."""
    orig = images.clone().detach()
    adv = orig.clone().detach()

    for _ in range(num_steps):
        grad_est = torch.zeros_like(adv)
        for _ in range(num_samples):
            v = (torch.randint(0, 2, adv.shape, device=adv.device, dtype=adv.dtype) * 2 - 1)  # Rademacher +-1
            with torch.no_grad():
                x_plus = torch.clamp(adv + delta * v, 0.0, 1.0)
                x_minus = torch.clamp(adv - delta * v, 0.0, 1.0)
                loss_plus = F.cross_entropy(model(preprocess_for_classifier(x_plus, dataset)), labels, reduction="none")
                loss_minus = F.cross_entropy(model(preprocess_for_classifier(x_minus, dataset)), labels, reduction="none")
            coef = (loss_plus - loss_minus).view(-1, *([1] * (adv.dim() - 1))) / (2 * delta)
            grad_est += coef * v
        grad_est /= num_samples

        adv = adv.detach() + alpha * grad_est.sign()
        delta_clip = torch.clamp(adv - orig, -epsilon, epsilon)
        adv = torch.clamp(orig + delta_clip, 0.0, 1.0).detach()

    success = _predict(model, dataset, adv) != labels
    return adv, success


def salt_pepper_attack(model, dataset, images, labels, max_amount=0.5, step=0.05):
    """Black-box, gradient-free: randomly sets a growing fraction of pixels
    to 0 (pepper) or 1 (salt), increasing the corruption level per sample
    only until that sample's classifier prediction flips (or max_amount is
    reached) -- gives each image the smallest successful corruption found,
    rather than one fixed noise level applied uniformly regardless of how
    much a given image actually needed."""
    adv = images.clone()
    success = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)

    amount = 0.0
    while amount < max_amount and not success.all():
        amount += step
        salt_mask = torch.rand_like(images) < (amount / 2)
        pepper_mask = (torch.rand_like(images) < (amount / 2)) & (~salt_mask)
        candidate = images.clone()
        candidate[salt_mask] = 1.0
        candidate[pepper_mask] = 0.0

        preds = _predict(model, dataset, candidate)
        newly_success = (preds != labels) & (~success)
        adv[newly_success] = candidate[newly_success]
        success = success | newly_success

    return adv, success


ATTACKS = {"fgsm": fgsm_attack, "pgd": pgd_attack, "spsa": spsa_attack, "salt_pepper": salt_pepper_attack}
