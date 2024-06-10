import torch


class TransformerMasking:
    @staticmethod
    def _generate_square_subsequent_mask(size: int) -> torch.Tensor:
        """用于decoder的mask, only Look-Ahead Mask
        shape: (size, size).
        """
        # Generate an upper-triangular matrix of 'inf' values, except for the diagonal (by setting diagonal=1).
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        # Use 'float('-inf')' to represent the masked positions.
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def _generate_padding_mask(x: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """Padding Mask, 用于encoder和decoder
        Shape: x: (batch_size, seq_len)
        return (batch_size, seq_len)
        """
        # Create a mask for positions that are equal to the pad_idx.
        # The mask will have 'True' where the 'pad_idx' is present.
        mask = (x == pad_idx)
        return mask


if __name__ == "__main__":
    # Example usage
    size = 4  # example sequence length for the Look-Ahead Mask

    batch_size = 2  # example batch size
    seq_len = 10  # example sequence length
    pad_idx = 0  # example padding index

    # Example tensor for generating padding mask
    x_example = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                              [4, 5, 6, 7, 8, 9, 10, 0, 0, 0]])
    print(x_example.shape)
    look_ahead_mask = TransformerMasking._generate_square_subsequent_mask(size)
    padding_mask = TransformerMasking._generate_padding_mask(
        x_example, pad_idx)

    print("Look-Ahead Mask:", look_ahead_mask)
    print("Padding Mask :", padding_mask.shape)
    print(padding_mask)
