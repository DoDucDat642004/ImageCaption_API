from libary_local import *

# Định nghĩa mã ngôn ngữ
lang_id = {
    "vi": 0, 
    "en": 1
}

# Load Tokenizers
tokenizers = {
    lang_id["vi"]: AutoTokenizer.from_pretrained("vinai/phobert-base"),
    lang_id["en"]: AutoTokenizer.from_pretrained("gpt2")
}

# Cập nhật special tokens cho GPT-2
special_tokens = {
    "bos_token": "<s>", 
    "eos_token": "</s>", 
    "pad_token": "<pad>"
}
tokenizers[1].add_special_tokens(special_tokens)

# Tạo dictionary lưu thông tin vocab & token đặc biệt
vocab_sizes =   { lid: len(tokenizer) for lid, tokenizer in tokenizers.items() }
pad_token_ids = { lid: tokenizer.pad_token_id for lid, tokenizer in tokenizers.items() }
