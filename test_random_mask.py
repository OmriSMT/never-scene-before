from mask_strategies import RandomMaskStrategy

strategy = RandomMaskStrategy()

questions = [
    "What state is American Idol contestant Chris Daughtry from?",
    "What are highly resistant dormant structures of certain gram-positive bacteria called?",
    "When did Barack Obama become president?",
]

for q in questions:
    words = q.split("?")[0].split(" ")
    strategy.sample_mask_proportion()
    mask = strategy(words)

    masked_words = [
        "<mask>" if mask[0, i].item() else word
        for i, word in enumerate(words)
    ]

    print("Original:", q)
    print("Words:", words)
    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Masked:", " ".join(masked_words) + "?")
    print("-" * 80)