DATASET = 'android'

user_dict = {
    'ask_ubuntu': 35038,
    'math' : 20458,
    'android' : 5343,
    'biology' : 1765
}

tag_dict = {
    'ask_ubuntu': 84293,
    'math' : 196036,
    'android' : 12357, #16191 ques 
    'biology' : 7608 #11750 ques
}


emb_dict = {
    'ask_ubuntu' :64,
    'android' :16,
    'biology' :16,
    'math':64
}

kernel_dict = {
    'ask_ubuntu' :33,
    'android' :9,
    'biology' :9,
    'math':33
}


TAG_NUMBER = tag_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)
EMB_size = emb_dict.get(DATASET)
KNL_size = kernel_dict.get(DATASET)


print("\n==========================================================================")
print('Dataset:', DATASET, '#Users:', USER_NUMBER, '#TAG COMB', TAG_NUMBER)
print("==========================================================================\n")


PAD = 0

