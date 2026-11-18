from tokenizers import Tokenizer


print("====================================")
print("       ZERO TOKENIZER TEST")
print("====================================")

tokenizer = Tokenizer.from_file(
    "tokenizer/zero_tokenizer.json"
)


text = "Once upon a time, there was a little girl."


print("\nOriginal text:")
print(text)


encoded = tokenizer.encode(text)


print("\nTokens:")
print(encoded.tokens)


print("\nToken IDs:")
print(encoded.ids)


print("\nNumber of tokens:")
print(len(encoded.ids))


decoded = tokenizer.decode(encoded.ids)


print("\nDecoded text:")
print(decoded)
