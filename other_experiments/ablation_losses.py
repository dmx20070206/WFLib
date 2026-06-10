import torch
import torch.nn.functional as F


def gaussian_kernel(source, target):
    sample_count = source.size(0) + target.size(0)
    combined = torch.cat([source, target], dim=0)
    l2_distance = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(dim=2)
    bandwidth = l2_distance.sum() / max(sample_count**2 - sample_count, 1)
    bandwidth = torch.clamp(bandwidth, min=1e-6)
    return torch.exp(-l2_distance / bandwidth)


def mmd_loss(source_features, target_features):
    batch_size = min(source_features.size(0), target_features.size(0))
    if batch_size == 0:
        device = source_features.device if source_features.numel() > 0 else target_features.device
        return torch.tensor(0.0, device=device)

    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]
    kernels = gaussian_kernel(source_features, target_features)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return (xx + yy - xy - yx).mean()


def softmax_entropy(logits):
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def energy_weights(energies, tau_center, gamma):
    return torch.sigmoid(-(energies - tau_center) / gamma)


def source_classification_loss(logits, labels, unknown_label, criterion):
    known_mask = labels != unknown_label
    if known_mask.any():
        return criterion(logits[known_mask], labels[known_mask])
    return torch.tensor(0.0, device=logits.device)


def target_entropy_loss(logits):
    probs = F.softmax(logits, dim=-1)
    marginal = probs.mean(dim=0)
    return softmax_entropy(logits).mean() + (marginal * torch.log(marginal + 1e-5)).sum()


def pseudo_label_loss(model, inputs, labels, criterion):
    logits = model(inputs)[0]
    return criterion(logits, labels)


def source_target_energy_loss(
    source_logits,
    source_labels,
    target_logits,
    unknown_label,
    tau_center,
    temperature,
    energy_m_in,
    energy_m_out,
):
    known_mask = source_labels != unknown_label
    unknown_mask = ~known_mask

    energy_known = torch.tensor(0.0, device=source_logits.device)
    energy_unknown = torch.tensor(0.0, device=source_logits.device)
    if known_mask.any():
        energy_known = torch.relu(energy_score(source_logits[known_mask], temperature) - energy_m_in).mean()
    if unknown_mask.any():
        energy_unknown = torch.relu(energy_m_out - energy_score(source_logits[unknown_mask], temperature)).mean()

    target_energy = energy_score(target_logits, temperature)
    target_weights = energy_weights(target_energy.detach(), tau_center, gamma=3.0)
    target_term = (
        target_weights * torch.relu(target_energy - energy_m_in)
        + (1.0 - target_weights) * torch.relu(energy_m_out - target_energy)
    ).mean()
    return energy_known + energy_unknown + 0.2 * target_term