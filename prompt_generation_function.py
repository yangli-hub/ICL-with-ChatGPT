import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import logging
import os
import pandas as pd
import ast
from torch.nn.functional import cosine_similarity
import json


def readfile(raw_file, blip_file, caption_file):
    with open(blip_file, 'r', encoding='UTF-8') as fp:
        df_img = pd.DataFrame()
        img_id_lst = []
        img_entity_lst = []
        img_sentiment_lst = []
        for line in fp:
            line = line.strip()
            if line != '':
                words, img_caption, img_entity, img_sentiment, img_id = line.split('####')  # yl add
                img_id_lst.append(img_id)
                img_entity_lst.append(img_entity)
                img_sentiment_lst.append(img_sentiment)
        df_img['img_id'] = img_id_lst
        df_img['img_entity'] = img_entity_lst
        df_img['img_sentiment'] = img_sentiment_lst

    with open(caption_file, 'r', encoding='UTF-8') as fp:
        df_caption = pd.DataFrame()
        img_id_lst = []
        img_caption_lst = []
        for line in fp:
            line = line.strip()
            if line != '':
                words, img_caption, img_id = line.split('####')  # yl add
                img_id_lst.append(img_id)
                img_caption_lst.append(img_caption)
        df_caption['img_id'] = img_id_lst
        df_caption['img_caption'] = img_caption_lst

    with open(raw_file, 'r', encoding='UTF-8') as fp:
        img_id_lst = []
        words_lst = []
        tuple_lst = []
        df_label = pd.DataFrame()
        for line in fp:
            line = line.strip()
            if line != '':
                words, img_words, tuples, img_id, img_tag = line.split('####')  # yl add
                img_id_lst.append(img_id)
                words_lst.append(words)
                tuple_lst.append(tuples)
        df_label['img_id'] = img_id_lst
        df_label['words'] = words_lst
        df_label['tuples'] = tuple_lst

    #df = pd.merge(df_label, df_img, how='left', on='img_id')
    #df = pd.merge(df, df_caption, how='left', on='img_id')
    return df_label


def combine_simiarity_matrix(main_contribute,aux_contribute,similarity_matrix_main,similarity_matrix_aux,df_test,df_train):
    similarity_matrix_combine = [[0 for j in range(df_train.shape[0])] for i in
                         range(df_test.shape[0])]
    for i in range(len(similarity_matrix_main)):
        for j in range(len(similarity_matrix_main[i])):
            #final_score_lst = []
            main_score=similarity_matrix_main[i][j][0]
            aux_score= similarity_matrix_aux[i][j][0]
            final_score = main_score*main_contribute+aux_score*aux_contribute
            #final_score_lst.append(final_score)
            similarity_matrix_combine[i][j] = final_score
    return similarity_matrix_combine


def process_rawfile_new(df):
    entity_final_lst, category_final_lst, sentiment_final_lst, golden_final_lst = [], [], [], []
    for index, row in df.iterrows():
        tuples = row['tuples']
        for element in tuples:
            temp_tuples = tuples.split(']')
            golden_lst = []
            entity_lst = []
            category_lst = []
            sentiment_lst = []
            for l in temp_tuples:
                temp_l = l.split(',')
                temp_l_lst = []
                for k in temp_l:
                    if k not in ['', '.']:
                        k = k.replace("[", "")
                        k = k.replace("'", "")
                        k = k.rstrip()
                        k = k.lstrip()
                        if k != 'NULL':
                            temp_l_lst.append(k)
                            # print('temp_l_lst', temp_l_lst)
                if len(temp_l_lst) > 1:
                    golden_new = []
                    # get the list of entity, category, sentiment for each sample
                    entity = temp_l_lst[0]
                    category = temp_l_lst[1]
                    sentiment = temp_l_lst[2]

                    # golden
                    golden_new.append(entity)
                    golden_new.append(category)
                    golden_new.append(sentiment)

                    entity_lst.append(entity)
                    category_lst.append(category)
                    sentiment_lst.append(sentiment)
                    golden_lst.append(golden_new)
                else:
                    continue
        entity_final_lst.append(entity_lst)
        category_final_lst.append(category_lst)
        sentiment_final_lst.append(sentiment_lst)
        golden_final_lst.append(golden_lst)
        logging.info('finish embedding for sentence' + str(index))
    df['entity_lst'] = entity_final_lst
    df['category_lst'] = category_final_lst
    df['sentiment_lst'] = sentiment_final_lst
    df['golden_lst'] = golden_final_lst

    logging.info('finish process for whole df')
    return df


def create_data_dic(df,tokenizer, model, device):
    entity_final_lst = []
    category_final_lst = []
    for index, row in df.iterrows():
        # not read csv
        entity_lst = row['entity_lst']
        category_lst = row['category_lst']
        for element in range(len(entity_lst)):
            entity_final_lst.append(entity_lst[element])
            category_final_lst.append(category_lst[element])
    final_lst = set(entity_final_lst+category_final_lst+['positive','neutral','negative'])

    # tokenize
    word_embedding_dict = {}
    for i, word in enumerate(final_lst):
        word_token = torch.tensor(tokenizer.encode(word, add_special_tokens=True), device=device).unsqueeze(0)
        # Pass the tokens through the BERT model to get the word embeddings:
        with torch.no_grad():
            word_embedding = model(word_token)[0][:, 0, :]
        word_embedding_dict[word] = word_embedding
    return word_embedding_dict


def get_sentence_embedding(sentence,tokenizer,model,device):
    # Tokenize the sentence and add special tokens [CLS] and [SEP]
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
    inputs.to(device)
    # Get the BERT output for the encoded sentence
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the last hidden state from the BERT output
    last_hidden_state = outputs.last_hidden_state
    # Get the embedding for the whole sentence by taking the sum of the last hidden state over all tokens
    sentence_embedding = torch.sum(last_hidden_state, dim=1)
    return sentence_embedding


def calculate_sentence_similarity_matrix(dataset1_embedding_lst,dataset2_embedding_lst):
    similarity_matrix = [[0 for j in range(len(dataset2_embedding_lst))] for i in range(len(dataset1_embedding_lst))]
    for i in range(len(dataset1_embedding_lst)):
        for j in range(len(dataset2_embedding_lst)):
            temp_comparison_lst = []
            for sentence1 in dataset1_embedding_lst[i]:
                for sentence2 in dataset2_embedding_lst[j]:
                    similar_score = cosine_similarity(sentence1.unsqueeze(0), sentence2.unsqueeze(0)).item()
                    temp_comparison_lst.append(similar_score)
            similarity_matrix[i][j] = temp_comparison_lst
    return similarity_matrix


def selected_top_samples (select_sample_cnt,similarity_matrix_combine,df_train,df_test):
    select_sample_index_list = []
    selected_samples_final_lst, selected_tuples_final_lst, selected_entity_final_lst,selected_sentiment_final_lst,\
    selected_caption_final_lst,selected_imgid_final_lst = [],[],[],[],[],[]
    for i in range(len(similarity_matrix_combine)):
        score_lst = similarity_matrix_combine[i]
        sorted_list = sorted(score_lst, reverse=True)
        selected_indices = [score_lst.index(val) for val in sorted_list[:select_sample_cnt]]
        select_sample_index_list.append(selected_indices)

        # query for testing dataset
        selected_samples = df_train.loc[selected_indices, 'words'].tolist()
        selected_tuples = df_train.loc[selected_indices, 'golden_lst'].tolist()
        selected_imgid = df_train.loc[selected_indices, 'img_id'].tolist()

        # combine to train
        selected_samples_final_lst.append(selected_samples)
        selected_tuples_final_lst.append(selected_tuples)
        selected_imgid_final_lst.append(selected_imgid)

    df_test['selected_samples'] = selected_samples_final_lst
    df_test['selected_tuples'] = selected_tuples_final_lst
    df_test['selected_imgid'] = selected_imgid_final_lst
    return df_test


def selected_top_samples_train_dev(select_sample_cnt, similarity_matrix_main_train, df_train_dev_comb, df_train):
    select_sample_index_list = []
    selected_samples_final_lst, selected_tuples_final_lst, selected_imgid_final_lst = [],[],[]
    for i in range(len(similarity_matrix_main_train)):
        score_lst = similarity_matrix_main_train[i]
        sorted_list = sorted(score_lst, reverse=True)
        selected_indices = [score_lst.index(val) for val in sorted_list[1:(select_sample_cnt+1)]]
        select_sample_index_list.append(selected_indices)

        # query for testing dataset
        selected_samples = df_train_dev_comb.loc[selected_indices, 'words'].tolist()
        selected_tuples = df_train_dev_comb.loc[selected_indices, 'golden_lst'].tolist()
        selected_imgid = df_train_dev_comb.loc[selected_indices, 'img_id'].tolist()

        # combine to train
        selected_samples_final_lst.append(selected_samples)
        selected_tuples_final_lst.append(selected_tuples)
        selected_imgid_final_lst.append(selected_imgid)

    df_train['selected_samples'] = selected_samples_final_lst
    df_train['selected_tuples'] = selected_tuples_final_lst
    df_train['selected_imgid'] = selected_imgid_final_lst
    return df_train


def generate_prompt(df_test, select_sample_cnt,element_type):
    tuple_final_lst = []
    string_final_lst = []
    for index, row in df_test.iterrows():
        words = row['selected_samples']
        tuples = row['selected_tuples']
        string_lst = []
        for i in range(select_sample_cnt):
            tuple_lst = []
            temp = tuples[i]
            entity_lst = []
            for j in temp:
                if element_type == 'entity_cat_sent_fine':
                    new_j = j
                elif element_type == 'entity_cat':  # ,'entity','sentiment'
                    new_j = []
                    new_j.append(j[0])
                    new_j.append(j[1])
                elif element_type == 'entity_sent':  # ,'entity','sentiment'
                    new_j = []
                    new_j.append(j[0])
                    new_j.append(j[2])
                elif element_type == 'entity':
                    new_j = j[0]
                elif element_type == 'sentiment':
                    new_j = []
                    new_j.append(j[0])
                    new_j.append(j[2])
                    entity_j = j[0]
                    entity_lst.append(entity_j)
                tuple_lst.append(new_j)

            temp_word = words[i]
            if element_type == 'sentiment':
                string = 'Example' + str(i) + ': ' + "The main sentences are {" + temp_word + ".} " \
                            " The entity list of the sentence is " + str(entity_lst)+"; " \
                            "The sentiment towards the entity is " + str(tuple_lst)
            else:
                string = 'Example'+ str(i)+': '+ "The input sentence is {"+temp_word +".} " \
                    "Label list is "+str(tuple_lst)
            string_lst.append(string)
            print('element_type', element_type)
            print(tuple_lst)
        tuple_final_lst.append(tuple_lst)
        string_final_lst.append(string_lst)
    df_test['Prompt'] = string_final_lst
    return df_test


def generate_sentence_embedding(tokenizer, model, device, df):
    main_sentence_embedding_lst,words_lst = [],[]
    for index, row in df.iterrows():
        sentence = row['words']
        main_sentence_embedding = get_sentence_embedding(sentence,tokenizer,model,device)
        main_sentence_embedding_lst.append(main_sentence_embedding)
        words_lst.append(index)
    return main_sentence_embedding_lst,words_lst

