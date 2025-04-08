DATASET = 'math'
EMB_size = 1024

user_dict = {
    'ask_ubuntu': 3494,
    'math' : 11762,
    # 'math' : 12546,
    'android' : 385,
    'biology' : 339
}

tag_dict = {
    'ask_ubuntu': 58212,
    'math' : 191503,
    # 'math' : 24824,
    'android' : 7710,
    'biology' : 6386
}

TAG_NUMBER = tag_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)

print("\n==========================================================================")
print('Dataset:', DATASET, '#Users:', USER_NUMBER, '#TAG COMB', TAG_NUMBER)
print("==========================================================================\n")


PAD = 0

