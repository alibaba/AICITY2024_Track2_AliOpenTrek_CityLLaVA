from llava.train.train_block import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

# from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()

# from llava.train.train_block import train

# if __name__ == "__main__":
#     train()

