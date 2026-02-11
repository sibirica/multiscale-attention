import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import RecombineDecoder, lift  # noqa: E402


def _set_identity_mlp(decoder: RecombineDecoder) -> None:
    in_dim = decoder.fc1.in_features
    hidden_dim = decoder.fc1.out_features
    decoder.fc1.weight.data.zero_()
    decoder.fc1.bias.data.zero_()
    decoder.fc2.weight.data.zero_()
    decoder.fc2.bias.data.zero_()
    for i in range(min(in_dim, hidden_dim, decoder.fc2.out_features)):
        decoder.fc1.weight.data[i, i] = 1.0
        decoder.fc2.weight.data[i, i] = 1.0


def main() -> None:
    rate = 4
    spatial_tokens = 9
    fast_dim = 8
    slow_dim = 8
    t_fast = 5
    t_slow = (t_fast + rate - 1) // rate

    z_fast = torch.zeros(1, t_fast * spatial_tokens, fast_dim)
    z_slow = torch.zeros(1, t_slow * spatial_tokens, slow_dim)
    z_slow[:, :, 0] = torch.arange(t_slow).repeat_interleave(spatial_tokens)
    z_slow[:, :, 1] = torch.arange(spatial_tokens).repeat(t_slow)

    decoder = RecombineDecoder(
        fast_embed_dim=fast_dim,
        slow_embed_dim=slow_dim,
        rate=rate,
        hidden_dim=fast_dim + slow_dim,
        act="relu",
        dropout=0.0,
        time_dim=1,
        spatial_tokens=spatial_tokens,
    )
    _set_identity_mlp(decoder)

    out = decoder(z_fast, z_slow)
    z_slow_lift = lift(z_slow, rate=rate, time_dim=1, spatial_tokens=spatial_tokens)
    if z_slow_lift.size(1) > z_fast.size(1):
        z_slow_lift = z_slow_lift[:, : z_fast.size(1)]

    # Identity MLP should keep the first fast_dim channels (from z_fast) unchanged.
    if not torch.equal(out[:, :, : fast_dim], z_fast):
        raise AssertionError("Recombine identity MLP did not preserve fast channels.")

    print("Recombine identity MLP check passed.")


if __name__ == "__main__":
    main()
