def compute_block_lengths(input_len: int, output_len: int, rate: int) -> list[int]:
    remaining = output_len
    cur_len = input_len
    step_len = (-cur_len) % rate
    if step_len == 0:
        step_len = rate
    blocks = []
    while remaining > 0:
        block_len = min(step_len, remaining)
        blocks.append(block_len)
        cur_len += block_len
        remaining -= block_len
        step_len = rate
    return blocks


def main() -> None:
    # Input ends at t=6 -> input_len=7, output_len=11 (t=7..17) with rate=4
    blocks = compute_block_lengths(input_len=7, output_len=11, rate=4)
    expected = [1, 4, 4, 2]
    cur_t = 6
    for block_len in blocks:
        start = cur_t + 1
        end = cur_t + block_len
        print(f"generate t={start}..{end}")
        cur_t = end
    assert blocks == expected, f"Unexpected block lengths: {blocks}"
    print("Generate step alignment check passed.")


if __name__ == "__main__":
    main()
