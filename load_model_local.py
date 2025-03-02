from libary_local import *
from model_local import *

# Load model và tokenizer
MODEL_PATH = "/Users/dodat/Downloads/multilingual_model.pth"  # Đường dẫn thư mục chứa model

def init_weights(module):    
    if isinstance(module, nn.Linear):  # Linear layers (DAEProjection, Generator)
        if isinstance(module, nn.GELU):
            init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming He cho GELU
        else:
            init.xavier_uniform_(module.weight)  # Xavier Uniform giúp ổn định gradient
        if module.bias is not None:
            init.zeros_(module.bias)  # Bias = 0 giúp tránh offset không cần thiết
    
    elif isinstance(module, nn.Embedding):  # Token Embedding
        init.trunc_normal_(module.weight, mean=0, std=0.02)  # Truncated Normal tránh trọng số quá lớn/nhỏ
    
    elif isinstance(module, nn.LayerNorm):  # LayerNorm trong DAEProjection
        module.bias.data.zero_()
        module.weight.data.fill_(1.0) 

    elif isinstance(module, nn.MultiheadAttention):  # Multihead Attention (DAEProjection, MHSA)
        # Kiểm tra trước khi khởi tạo để tránh lỗi NoneType
        if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
            init.xavier_uniform_(module.in_proj_weight)
        if hasattr(module, "q_proj_weight") and module.q_proj_weight is not None:
            init.xavier_uniform_(module.q_proj_weight)
        if hasattr(module, "k_proj_weight") and module.k_proj_weight is not None:
            init.xavier_uniform_(module.k_proj_weight)
        if hasattr(module, "v_proj_weight") and module.v_proj_weight is not None:
            init.xavier_uniform_(module.v_proj_weight)
        if hasattr(module, "out_proj") and module.out_proj.weight is not None:
            init.xavier_uniform_(module.out_proj.weight)

    elif isinstance(module, nn.Parameter):  # Các tham số học được (self.feature_weight, residual_weight)
        init.uniform_(module, a=0.0, b=1.0)  # Khởi tạo trong khoảng [0, 1] giúp mô hình học nhanh hơn


def apply_init_weights(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.mhsa.apply(init_weights)
    model.token_embeddings.apply(init_weights)
    model.positional_encoding.apply(init_weights)
    model.projection.apply(init_weights)
    model.generators.apply(init_weights)

# Load model và checkpoint
def load_checkpoint(device=device):
    if os.path.exists(MODEL_PATH):
        print("Loading checkpoint...")

        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model = ImageCaptionModel(
            vocab_sizes=checkpoint["vocab_sizes"],
            pad_token_ids=checkpoint["pad_token_ids"]
        )

        apply_init_weights(model)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model

# Load model và checkpoint
model = load_checkpoint()