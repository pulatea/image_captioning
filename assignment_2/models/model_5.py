import torch.nn
import torch.nn.functional as F
import math

from einops import rearrange

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    def __init__(self, n_layers, d_model, num_heads, fc_dim, dropout, vocabulary):
        super().__init__(vocabulary=vocabulary)

        self.image_encoder = ImageEncoder()
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  n_layers=n_layers,
                                                  d_model=d_model,
                                                  num_heads=num_heads,
                                                  fc_dim=fc_dim,
                                                  dropout=dropout,
                                                  image_embed_dim=self.image_encoder.dino.embed_dim)


class ImageEncoder(BaseImageEncoder):
    def __init__(self):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    def freeze(self):
        for parameter in self.dino.parameters():
            parameter.requires_grad = False

    def forward(self, image):
        resized_image = F.interpolate(image, size=[224, 224], mode="bilinear", align_corners=False)
        dino_sp, dino_cls = self.dino.get_intermediate_layers(resized_image,
                                                              n=1,
                                                              reshape=True,
                                                              return_class_token=True,
                                                              norm=True)[-1]

        return dino_cls


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, n_layers, d_model, num_heads, fc_dim, dropout, image_embed_dim):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary_size,
                                            embedding_dim=d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.transformer_blocks = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.transformer_blocks.append(torch.nn.TransformerEncoderLayer(d_model=d_model,
                                                                            nhead=num_heads,
                                                                            dim_feedforward=fc_dim,
                                                                            dropout=dropout,
                                                                            norm_first=True,
                                                                            batch_first=True))

        self.to_d_model = torch.nn.Linear(image_embed_dim, d_model)
        self.to_logits = torch.nn.Linear(d_model, vocabulary_size)

    def freeze(self):
        pass

    def forward(self, encoded_image, caption_indices, *args):
        caption_embeddings = self.embedding(caption_indices)
        image_embeddings = self.to_d_model(encoded_image)

        input_embeddings = torch.cat([image_embeddings.unsqueeze(1), caption_embeddings], dim=1)
        output = self.pos_enc(input_embeddings.permute(1, 0, 2)).permute(1, 0, 2)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(input_embeddings.size(1)).to(encoded_image.device)

        for block in self.transformer_blocks:
            output = block(output, src_mask=mask)

        output = output[:, 1:]
        logits = self.to_logits(output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2)}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = [sos_token_index]

        for _ in range(max_length):
            output = self.forward(encoded_image=encoded_image,
                                  caption_indices=torch.LongTensor(caption_indices).to(encoded_image.device).unsqueeze(
                                      dim=0))

            predicted_index = output['indices'][:, -1]

            caption_indices.append(predicted_index.item())
            if caption_indices[-1] == eos_token_index:
                break

        return caption_indices[1:]


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
