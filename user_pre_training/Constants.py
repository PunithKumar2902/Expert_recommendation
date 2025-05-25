DATASET = 'ask_ubuntu'

user_dict = {
    'android':3144,
    'ask_ubuntu': 21075,
    'biology' : 1126,
    'codereview':4742,
    'electronics':5744,
    'english':5276,
    'es':7630,
    'gis':6189
}

tag_dict = {
    'android':7544, #10455 ques
    'ask_ubuntu': 51397, #88994 ques
    'biology' : 4824, #8110 ques
    'codereview':15122, #31424 ques
    'electronics':36617, #63369 ques
    'english':12939, #45145 ques
    'es' :17320, #58568 ques
    'gis':26403, #44583
}


emb_dict = {
    'android' :16,
    'ask_ubuntu' :64,
    'biology' :16,
    'codereview':16,
    'electronics':16,
    'english':16,
    'es' :16,
    'gis' :16
}


TAG_NUMBER = tag_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)
EMB_size = emb_dict.get(DATASET)

print("\n==========================================================================")
print('Dataset:', DATASET, '#Users:', USER_NUMBER, '#TAG COMB', TAG_NUMBER)
print("==========================================================================\n")


PAD = 0

