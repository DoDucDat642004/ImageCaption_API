from libary_local import *
# Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_token_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.scale = math.sqrt(embedding_dim)

    def forward(self, tokens):
        return self.embedding(tokens) * self.scale

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, maxlen=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        den = torch.exp(-torch.arange(0, embedding_dim, 2) * math.log(10000.0) / embedding_dim)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros(maxlen, embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.register_buffer('pos_embedding', pos_embedding.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pos_embedding[:, :x.size(1), :])

class DAEProjection(nn.Module):
    def __init__(self, input_dim=1536, latent_dim=1024):
        super().__init__()

        # Feature Weighting để giữ thông tin gốc
        self.feature_weight = nn.Parameter(torch.ones(1, input_dim))

        # LayerNorm giúp giữ thông tin ổn định
        self.norm = nn.LayerNorm(input_dim)

        # Encoder: Thêm Dense Bottleneck để giảm chiều mượt hơn
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1408),  
            nn.GELU(),
            nn.Linear(1408, 1280),  
            nn.GELU(),
            nn.Linear(1280, latent_dim)
        )

        # Self-Attention giữ thông tin quan trọng nhất
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=16, batch_first=True)

        # Residual Connection với trọng số học được
        self.residual_weight = nn.Parameter(torch.tensor(0.1))  

        # Decoder khôi phục với Dense Bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1280),
            nn.GELU(),
            nn.Linear(1280, 1408),
            nn.GELU(),
            nn.Linear(1408, input_dim)
        )

    def forward(self, x):
        # Chuẩn hóa đầu vào để giữ ổn định
        x = self.norm(x) * self.feature_weight

        # Encoder: Giảm chiều từ từ qua Dense Bottleneck
        z = self.encoder(x)
        z, _ = self.attn(z, z, z)  # Self-Attention giữ thông tin quan trọng

        # Decoder: Khôi phục lại với Residual giữ thông tin gốc
        reconstructed = self.decoder(z) + x * self.residual_weight  # Giữ lại thông tin gốc với trọng số học được

        # MSE Loss để đo độ mất thông tin
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')

        return z, reconstructed, recon_loss

# Image Caption Model 
class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_sizes, pad_token_ids, embedding_dim=1024, num_heads=16, num_decoder_layers=8, ffn_dim=4096, dropout=0.1):
        super().__init__()
        self.pad_token_ids = pad_token_ids

        # Encoder : Swinv2
        self.swin_encoder = create_model("swinv2_large_window12_192", pretrained=True, num_classes=0)

        # Self-Attention
        self.mhsa = nn.MultiheadAttention(embed_dim=1536, num_heads=num_heads, dropout=0, batch_first=True)

        # DAE Projection
        self.projection = DAEProjection(1536, embedding_dim)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=0)

        # Token Embeddings
        self.token_embeddings = nn.ModuleList([
            TokenEmbedding(vocab_sizes[lang], embedding_dim, pad_token_ids[lang]) for lang in vocab_sizes
        ])

        # List Decoders
        self.decoders = nn.ModuleList([
            nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout, norm_first=True, batch_first=True),
                num_layers=num_decoder_layers
            ) for _ in vocab_sizes
        ])

        # Generator List
        self.generators = nn.ModuleList([
            nn.Linear(embedding_dim, vocab_sizes[lang]) for lang in vocab_sizes
        ])

    def encode(self, img):
        with torch.no_grad():
            memory = self.swin_encoder.forward_features(img)

        b, h, w, c = memory.shape
        memory = memory.view(b, h * w, c)

        # Self-Attention
        memory, _ = self.mhsa(memory, memory, memory)

        # Giảm chiều bằng Deterministic AE
        reduced_memory, reconstructed, dae_loss = self.projection(memory)

        return reduced_memory, dae_loss

    def forward(self, img, tokens, lang_id):
        device = tokens.device
        lang_idx = lang_id[0].item()

        memory, dae_loss = self.encode(img)

        tgt_emb = self.token_embeddings[lang_idx](tokens)
        tgt_emb = self.positional_encoding(tgt_emb)

        seq_len = tokens.shape[1]
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device) == 1).transpose(0, 1)
        tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        padding_mask = (tokens == self.pad_token_ids[lang_idx]).float().to(device)

        decoded = self.decoders[lang_idx](tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)

        return self.generators[lang_idx](decoded), dae_loss.unsqueeze(0)

    def predict_caption_greedy(self, image, tokenizer, lang_id, max_length=30):
        """Dự đoán caption từ ảnh"""
        self.eval()
        image = image.unsqueeze(0).to(device)
        
        # Bắt đầu với token BOS
        generated_tokens = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)

        for _ in range(max_length):
            output, _ = self.forward(image, generated_tokens, torch.tensor([lang_id], dtype=torch.long, device=device))
            logits = output[:, -1, :]
            next_token_id = logits.argmax(dim=-1).item()

            if next_token_id == tokenizer.eos_token_id:
                break

            generated_tokens = torch.cat([generated_tokens, torch.tensor([[next_token_id]], device=device)], dim=-1)

        return tokenizer.decode(generated_tokens.squeeze().tolist(), skip_special_tokens=True)

    def predict_caption_beam(self, image, tokenizer, lang_id, max_length=30, beam_size=3):
        self.eval()

        # Xử lý ảnh thành tensor [1, C, H, W]
        image = image.unsqueeze(0).to(device)

        # Chuẩn bị decoder
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        # Khởi tạo beam với (chuỗi token, xác suất log)
        beams = [(torch.tensor([[bos_token_id]], dtype=torch.long, device=device), 0)]

        for _ in range(max_length):
            new_beams = []

            for seq, score in beams:
                # Dự đoán token tiếp theo
                with torch.no_grad():
                    output, _ = self.forward(image, seq, torch.tensor([lang_id], dtype=torch.long, device=device))
                
                logits = output[:, -1, :]  # Lấy xác suất token cuối
                probs = F.log_softmax(logits, dim=-1)  # Log xác suất để tránh số nhỏ
                top_k = torch.topk(probs, beam_size, dim=-1)  # Lấy top-k từ beam search

                # Mở rộng beam
                for i in range(beam_size):
                    next_token_id = top_k.indices[0, i].item()  # Lấy token từ top-k
                    next_score = score + top_k.values[0, i].item()  # Cộng log xác suất

                    # Nếu gặp <EOS>, dừng sớm
                    if next_token_id == eos_token_id:
                        return tokenizer.decode(seq.squeeze().tolist(), skip_special_tokens=True)

                    # Thêm vào beam mới
                    new_seq = torch.cat([seq, torch.tensor([[next_token_id]], device=device)], dim=-1)
                    new_beams.append((new_seq, next_score))

            # Chọn beam tốt nhất (theo xác suất cao nhất)
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        # Trả về chuỗi có xác suất cao nhất
        return tokenizer.decode(beams[0][0].squeeze().tolist(), skip_special_tokens=True)

    def predict_caption(self, image, tokenizer, lang_id, mode='beam'):
        if (mode == "greedy"):
            return self.predict_caption_greedy(image, tokenizer, lang_id)
        else :
            return self.predict_caption_beam(image, tokenizer, lang_id)
