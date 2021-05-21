import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from vocabulary import Vocabulary


class Seq2SeqModel(nn.Module):
    """
    A Pytorch module for a Seq2Seq model based on LSTMs

    Arguments:
        embedding_dim: the size of the vocabulary embeddings
        hidden_dim: the size of the hidden states for both LSTMs
        num_layers: the number of LSTM layers to use in both encoder and decoder
        vocab_size: the size of the vocabulary
        padding_index: the index of the padding token
        dropout: the dropout rate to use after embedding and before output to logits. Default: `0.0`
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        vocab_size: int,
        padding_index: int,
        dropout: float = 0.0
    ):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(2 * hidden_dim, vocab_size)
        self.padding_index = float(padding_index)
        self.dropout_prob = dropout
        if self.dropout_prob:
            self.dropout = nn.Dropout(dropout)

    def generate(
        self,
        source: Tensor,
        source_lengths: list[int],
        bos_index: int,
        eos_index: int,
        vocab: Vocabulary,
        temperature: float,
        max_length: int = 512,
    ) -> list[str]:
        """
        Generate text output from Seq2Seq input

        Arguments:
            source: the source sequence. `(1, seq_length)`
            source_lengths: the list of lengths of the source sequences
            bos_index: the index of the BOS token
            eos_index: the index of the EOS token
            vocab: the Vocabulary of the model
            max_length: when to stop generating if EOS is never reached

        Returns:
            string output generation
        """
        # Freebie: don't alter this method
        source_emb = self.embedding(source)
        bos = torch.tensor([bos_index], dtype=source.dtype)
        next_char = bos.unsqueeze(0)
        encodings, (hidden, cell) = self.encode(source_emb, source_lengths)
        text = [next_char.item()]
        eos_generated = False
        length = 0
        with torch.no_grad():
            while length < max_length:
                previous_emb = self.embedding(next_char)
                logits, (hidden, cell) = self.decode(
                    (hidden, cell),
                    previous_emb,
                    [1],
                    encoder_states=encodings
                )
                logits = logits[:, -1, :].unsqueeze(dim=1) * temperature
                probs = torch.distributions.categorical.Categorical(logits=logits)
                next_char = probs.sample()
                # next_char = torch.argmax(logits[0,-1]).unsqueeze(0).unsqueeze(1)
                text.append(next_char.item())
                if next_char.item() == eos_index:
                    eos_generated = True
                    break
                length += 1
        # Trim the BOS token, and if EOS was generated, trim that too
        text = text[1:]
        if eos_generated:
            text = text[:-1]
        text = "".join(vocab.indices_to_tokens(text))
        return text

    def forward(
        self, source: Tensor, target: Tensor, lengths: tuple[list[int], list[int]]
    ) -> Tensor:
        """
        Perform a forward pass of the seq2seq model. Assumes both source and target sequences are
        available

        Arguments:
            source: a padded batch of source sequences. `(batch_size, source_sequence_length)`
            target: a padded batch of the corresponding target sequences.
                `(batch_size, target_sequence_length)`
            lengths: a pair of lists of sequence lengths. The first corresponds to the source
                lengths, and the second to the target lengths

        Returns:
            the decoded (and unpacked) output of the decoder.
                `(batch_size, target_sequence_length, vocab_size)`
        """
        # DONE: Implement forward() using the signature, docstring, and comments as guide

        # Embed the source and target sequences
        # (batch_size, sequence_length, embedding_dim)
        source_embeddings = self.embedding(source)
        target_embeddings = self.embedding(target)

        # Apply dropout to the source and target embeddings if using non-zero dropout
        if self.dropout_prob:
            source_embeddings = self.dropout(source_embeddings)
            target_embeddings = self.dropout(target_embeddings)

        # Get the encodings and final hidden representations from the encoder
        source_lengths, target_lengths = lengths
        encoder_output, (h, c) = self.encode(source_embeddings, source_lengths)

        # Freebie: you'll need these lines to get the attention padding mask
        source_padding_mask = self.get_padding_mask(
            batch_size=source.size(0),
            max_key_len=source.size(1),
            key_lengths=source_lengths
        )
        source_padding_mask = source_padding_mask.type(self.embedding.weight.dtype)

        # Get the decoder output and return (can discard the hidden and cell state)
        # (batch_size, target_sequence_length, vocab_size)
        decoder_output, _ = self.decode((h, c),
                                        target_embeddings,
                                        target_lengths,
                                        encoder_output,
                                        source_padding_mask)

        return decoder_output

    def encode(
        self, source_embeddings: Tensor, source_lengths: list[int]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Encode a batch of source sequences. Assumes sequences have already been embedded

        Arguments:
            source_embeddings: an embedded batch of source sequences.
                `(batch_size, source_sequence_length, embedding_dim)`
            source_lengths: the list of lengths of the source sequences

        Returns:
            encoder_output: the output of the encoder (unpacked).
                `(batch_size, source_sequence_length, hidden_dim)`
            (h, c): the tuple of the final hidden state and cell state from the encoder (per
                sequence in the batch). `((batch_size, hidden_dim), (batch_size, hidden_dim))`
        """
        # DONE: Implement encode() using the signature, docstring, and comments as guide
        # Hint: familiarize yourself with nn.utils.rnn.pack_padded_sequence and
        # nn.utils.rnn.pad_packed_sequence

        # Pack the source sequences
        source_packed = nn.utils.rnn.pack_padded_sequence(source_embeddings,
                                                          source_lengths,
                                                          batch_first=True)

        # Get packed output and (h,c) from encoder
        # Hint: be sure of the return signature for torch.nn.LSTM
        encoder_output, (h, c) = self.encoder(source_packed)

        # Unpack the encoder output
        # (batch_size, source_sequence_length, hidden_dim)
        encoder_output = nn.utils.rnn.pad_packed_sequence(encoder_output, batch_first=True)

        # Return the encoder output and tuple of hidden and cell states
        return encoder_output, (h, c)

    def decode(
        self,
        inits: tuple[Tensor, Tensor],
        target_embeddings: Tensor,
        target_lengths: list[int],
        encoder_states: Tensor,
        padding_mask: Tensor = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Decode a batch of target sequences to logits using the final hidden states of the encoder,
        as well as attention over its encodings. Assumes sequences have already been embedded

        Arguments:
            inits: the batch of final hidden and cell states from the encoder.
                `(batch_size, hidden_dim)` each
            target_embeddings: an embedded batch of target sequences.
                `(batch_size, target_sequence_length, embedding_dim)`
            target_lengths: the list of lengths of the target sequences
            encoder_states: the encodings of the source sequence.
                `(batch_size, source_sequence_length, hidden_dim)`
            padding_mask: an additive attention padding mask blocking queries from attending to key
                positions corresponding to pad tokens in the source.
                `(batch_size, 1, source_sequence_length)`. Default: `None`

        Returns:
            output: the batch of decoded output logits.
                `(batch_size, target_sequence_length, vocab_size)`
            (h, c): the tuple of the final hidden state and cell state from the decoder (per
                sequence in the batch). `((batch_size, hidden_dim), (batch_size, hidden_dim))`
        """
        # DONE: Implement decode() using the signature, docstring, and comments as guide

        # Pack the target sequences
        target_packed = nn.utils.rnn.pack_padded_sequence(target_embeddings,
                                                          target_lengths,
                                                          batch_first=True)

        # Get packed output and (h,c) from the decoder
        decoder_output, (h, c) = self.decoder(target_packed, inits)

        # Unpack the decoder output
        # (batch_size, target_sequence_length, hidden_dim)
        decoder_output = nn.utils.rnn.pad_packed_sequence(decoder_output, batch_first=True)

        # Get the attention values for each decoder position
        # (batch_size, target_sequence_length, hidden_dim)
        # Then concatenate these values with the output from the decoder
        # (batch_size, target_sequence_length, 2 * hidden_dim)
        attention = self.attention(decoder_output, encoder_states, padding_mask=padding_mask)
        output_with_attention = torch.cat([decoder_output, attention], dim=-1)

        # Apply dropout to the concatenated vectors if using non-zero dropout
        if self.dropout_prob:
            output_with_attention = self.dropout(output_with_attention)

        # Apply the output layer to get logits
        # (batch_size, target_sequence_length, vocab_size)
        logits = self.output(output_with_attention)

        return logits, (h, c)

    def attention(
        self, decoder_states: Tensor, encoder_states: Tensor, padding_mask: Tensor = None
    ) -> Tensor:
        """
        Compute attention outputs over decoder states (Queries) and encoder states (Keys, Values).
        Apply a source padding mask if one is input

        Arguments:
            decoder_states: a matrix of decoder hidden states to act as Queries for attention.
                `(batch_size, target_sequence_length, hidden_dim)`
            encoder_states: a matrix of encoder hidden states to act as Keys and Values for
                attention. `(batch_size, source_sequence_length, hidden_dim)`
            padding_mask: an additive attention padding mask blocking queries from attending to key
                positions corresponding to pad tokens in the source.
                `(batch_size, 1, source_sequence_length)`. Default: `None`

        Returns:
            a linear combination of Values, weighted by attention score.
                `(batch_size, target_sequence_length, hidden_dim)`
        """
        # DONE: Implement attention() using the signature, docstring, and comments as guide
        # Return a linear combination of Values per Query, as weighted by attention with the Keys
        # Hint: this can all be accomplished using transpose, matmul, addition, and softmax
        # Hint: add the padding mask before applying softmax
        # Hint: use torch.nn.functional.softmax and pay attention (no pun intended) to the dimension
        # over which it is applied

        q, k, v = decoder_states, encoder_states, encoder_states

        # (batch_size, target_sequence_length, source_sequence_length)
        weights = torch.bmm(q, torch.transpose(k, 1, 2))
        if padding_mask:
            weights = weights + padding_mask
        weights = F.softmax(weights, dim=-1)

        # (batch_size, target_sequence_length, source_sequence_length)
        attention = torch.bmm(weights, v)
        return attention

    @staticmethod
    def get_padding_mask(
        batch_size: int,
        max_key_len: int,
        key_lengths: list[int]
    ) -> Tensor:
        """
        Return an additive padding mask that blocks Queries from attending to Keys that are padded
        out in the source. Unblocked positions have value 0, blocked positions have value -inf

        Arguments:
            batch_size: the size of the source/target batch
            max_key_len: the maximum length of the source batch
            key_lengths: the length for each source sequence

        Returns:
            An additive mask for attention over the source sequences.
                `(batch_size, 1, source_sequence_length)`
        """
        # Freebie: don't alter this method
        padding_mask = np.zeros((batch_size, 1, max_key_len))
        for seq in range(batch_size):
            if key_lengths[seq] < max_key_len:
                for pad in range(key_lengths[seq], max_key_len):
                    padding_mask[seq, :, pad] = np.NINF
        return torch.tensor(padding_mask)
