DATASET = 'ask_ubuntu'

user_dict = {
    'ask_ubuntu': 5062,
    'math' : 11762,
    # 'math' : 12546,
    'android' : 385,
    'biology' : 339
}

tag_dict = {
    'ask_ubuntu': 61136,
    'math' : 191503,
    # 'math' : 24824,
    'android' : 7710,
    'biology' : 6386
}


emb_dict = {
    'ask_ubuntu' :1024,
    'android' :64
}


TAG_NUMBER = tag_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)
EMB_size = emb_dict.get(DATASET)


print("\n==========================================================================")
print('Dataset:', DATASET, '#Users:', USER_NUMBER, '#TAG COMB', TAG_NUMBER)
print("==========================================================================\n")


PAD = 0

