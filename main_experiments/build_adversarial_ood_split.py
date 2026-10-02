"""Assembles adversarial OOD test sets from the generated adversarial image pools."""
import os

import numpy as np
import torch

ATTACK_ORDER = ["fgsm", "pgd", "spsa", "salt_pepper"]


def _take_with_padding(pool, want, seed, pool_label):
    """First `want` images if the pool is big enough; otherwise the whole
    pool plus a seeded with-replacement sample of the pool itself to make
    up the shortfall (duplicates, not synthesized images)."""
    if len(pool) >= want:
        return pool[:want]
    shortfall = want - len(pool)
    print(f"WARNING: '{pool_label}' pool only has {len(pool)}/{want} successful images -- "
          f"padding the missing {shortfall} by duplicating existing ones from this same pool.")
    rng = np.random.default_rng(seed)
    pad_idx = rng.integers(0, len(pool), size=shortfall)
    return torch.cat([pool, pool[pad_idx]], dim=0)


def load_adversarial_test_set(dataset, norm_cls, adversarial_dir, attack, seed=0):
    """Returns (test_imgs, test_labels, composition) --
      - test_imgs: (N,1,H,W) float32 [0,1]
      - test_labels: (N,) int (0=ID/clean, 1=OOD/adversarial), 50/50 split
      - composition: dict {"n_id": ..., "fgsm": ..., "pgd": ..., "spsa": ...,
        "salt_pepper": ...} -- exact image count contributed by clean ID and
        by `attack` (0 for the other three), INCLUDES any padding/duplicate
        images, since those are still real images actually scored, just
        repeated.

    OOD is drawn entirely from `attack`'s pool. `seed` controls the
    (reproducible) duplication used to pad the pool if it falls short of
    what's needed.
    """
    assert attack in ATTACK_ORDER, f"attack must be one of {ATTACK_ORDER}, got {attack}"

    clean_path = os.path.join(adversarial_dir, f"clean_id_{dataset}_normcls{norm_cls}.pt")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"No clean-ID pool found at '{clean_path}' -- run generate_adversarial_images.py for "
            f"--dataset {dataset} --norm_cls {norm_cls} first."
        )
    clean_imgs = torch.load(clean_path, map_location="cpu", weights_only=False)["images"]
    n_id = len(clean_imgs)

    pool_path = os.path.join(adversarial_dir, f"adv_{dataset}_normcls{norm_cls}_{attack}.pt")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(
            f"No adversarial pool found at '{pool_path}' -- run generate_adversarial_images.py "
            f"--attacks {attack} for --dataset {dataset} --norm_cls {norm_cls} first."
        )
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)["images"]

    composition = {"n_id": n_id, **{a: 0 for a in ATTACK_ORDER}}
    adv_imgs = _take_with_padding(pool, n_id, seed, f"{dataset}_normcls{norm_cls}_{attack}")
    composition[attack] = len(adv_imgs)

    test_imgs = torch.cat([clean_imgs, adv_imgs], dim=0).numpy()
    test_labels = torch.cat([torch.zeros(n_id, dtype=torch.long), torch.ones(len(adv_imgs), dtype=torch.long)]).numpy()
    return test_imgs, test_labels, composition


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick standalone check: assembles and reports the shape/"
                                                  "balance of one adversarial-OOD test set.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--norm_cls", type=int, default=0)
    parser.add_argument("--adversarial_dir", type=str, default="outputs/adversarial_images")
    parser.add_argument("--attack", type=str, required=True, choices=ATTACK_ORDER)
    parser.add_argument("--seed", type=int, default=0, help="controls reproducible padding of any short pool")
    args = parser.parse_args()

    test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                       args.attack, args.seed)
    n_id = int((test_labels == 0).sum())
    n_ood = int((test_labels == 1).sum())
    print(f"{args.dataset} class {args.norm_cls} ({args.attack}): "
          f"test_imgs shape={test_imgs.shape}, ID={n_id}, OOD={n_ood}")
    print(f"composition: {composition}")
