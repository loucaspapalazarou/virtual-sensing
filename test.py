import torch


def training_step(batch):
    batch_size, seq_len, input_size = batch.size()
    total_loss = 0.0
    # Extract sub-sequences
    window_size = 2
    prediction_length = 2
    stride = 1
    # Calculate source length based on window size and prediction length
    for i in range(0, seq_len - (window_size + prediction_length + 1), stride):
        # Extract source and target sequences
        start_idx = i
        src = batch[:, start_idx : start_idx + window_size, :]
        tgt = batch[
            :,
            start_idx + prediction_length : start_idx + window_size + prediction_length,
            :,
        ]

        # print(i, src, tgt)
        print(src.shape, tgt.shape)
        # print()

    # Normalize the total loss by the number of sub-sequences
    num_subsequences = (seq_len - window_size) // stride + 1
    total_loss /= num_subsequences


batch = torch.rand(1, 12, 1)
print(batch)
training_step(batch)
