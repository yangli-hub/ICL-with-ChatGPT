import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
import logging
import os
import ast
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel


def readfile(raw_file, blip_file, caption_file, output):
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

    df = pd.merge(df_label, df_img, how='left', on='img_id')
    df = pd.merge(df, df_caption, how='left', on='img_id')
    return df


def process_rawfile_new(df, category_dic):
    entity_final_lst, category_final_lst, sentiment_final_lst, golden_final_lst = [], [], [], []
    for index, row in df.iterrows():
        tuples = row['tuples']
        img_entity = row['img_entity']
        img_sentiment = row['img_sentiment']
        img_caption = row['img_caption']
        auxilary_sentence = "The image is about " + img_caption + ", it implies the sentiment of " + img_sentiment + ", " \
                                                                                                                     "and the entity of " + img_entity + " is present in the image."
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
                    org_category = temp_l_lst[1]
                    try:
                        category = category_dic[org_category]
                    except:
                        category = org_category
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

    #df.to_csv(output)
    logging.info('finish process for whole df')
    return df

def get_word_embedding(tokenizer, model, device, word):
    # Tokenize the input words using the tokenizer and convert them to torch tensors:
    word_tokens = torch.tensor(tokenizer.encode(word, add_special_tokens=True),device=device).unsqueeze(0)
    # Pass the tokens through the BERT model to get the word embeddings:
    with torch.no_grad():
        word_embedding = model(word_tokens)[0][:, 0, :]
    return word_embedding


def calculate_embedding_similarity(word1_embedding, word2_embedding):
    similarity = cosine_similarity(word1_embedding, word2_embedding)
    return similarity


def calculate_word_similarity(tokenizer,model,device,word1, word2):
    #Tokenize the input words using the tokenizer and convert them to torch tensors:
    word1_tokens = torch.tensor(tokenizer.encode(word1, add_special_tokens=True),device=device).unsqueeze(0)
    word2_tokens = torch.tensor(tokenizer.encode(word2, add_special_tokens=True),device=device).unsqueeze(0)

    # Pass the tokens through the BERT model to get the word embeddings:
    with torch.no_grad():
        word1_embedding = model(word1_tokens)[0][:, 0, :]
        word2_embedding = model(word2_tokens)[0][:, 0, :]

    similarity = cosine_similarity(word1_embedding, word2_embedding)
    return similarity


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


def calculate_word_similarity_update(word_embedding_dict,word1, word2):
    word1_embedding = word_embedding_dict.get(word1, 0)
    word2_embedding = word_embedding_dict.get(word2, 0)
    similarity = cosine_similarity(word1_embedding, word2_embedding)
    return similarity


def calculate_similarity_matrix_update(golden_entity_lst,golden_category_lst,golden_sentiment_lst,word_embedding_dict,element_type):
    similarity_matrix = [[0 for j in range(len(golden_entity_lst))] for i in range(len(golden_entity_lst))]
    entity_matrix = [[0 for j in range(len(golden_entity_lst))] for i in range(len(golden_entity_lst))]
    similarity_matrix_mean = [[0 for j in range(len(golden_entity_lst))] for i in range(len(golden_entity_lst))]

    for i in range(len(golden_entity_lst)):
        logging.info('start the sample' + str(i))
        for j in range(len(golden_entity_lst)):
            temp_temp_comparison_lst = []
            temp_temp_entity_lst = []
            temp_temp_comparison_mean_lst = []
            for index, entity1 in enumerate(golden_entity_lst[i]):
                category1 = str(golden_category_lst[index][0])
                sentiment1 = str(golden_sentiment_lst[index][0])
                temp_comparison_lst = []
                temp_entity_lst = []
                for index, entity2 in enumerate(golden_entity_lst[j]):
                    category2 = str(golden_category_lst[index][0])
                    sentiment2 = str(golden_sentiment_lst[index][0])

                    entity_similar_score = calculate_word_similarity_update(word_embedding_dict,entity1, entity2)
                    category_similar_score = calculate_word_similarity_update(word_embedding_dict,category1, category2)
                    sentiment_similar_score = calculate_word_similarity_update(word_embedding_dict,sentiment1, sentiment2)

                    #
                    if element_type in ['entity_cat_sent']:
                        similar_score = entity_similar_score * 0.333 + category_similar_score * 0.333 + sentiment_similar_score * 0.333
                    elif element_type == 'entity_sent':  # ,'entity','sentiment'
                        similar_score = entity_similar_score*0.5+sentiment_similar_score*0.5
                    elif element_type == 'entity':
                        similar_score = entity_similar_score
                    else:
                        similar_score = sentiment_similar_score

                    similar_score = float(similar_score)
                    temp_comparison_lst.append(similar_score)

                    temp_entity_lst.append(entity1+'_VS_'+entity2)
                # mean
                if len(temp_comparison_lst) !=0:
                    max_score = max(temp_comparison_lst)
                temp_temp_comparison_mean_lst.append(max_score)

                temp_temp_comparison_lst.append(temp_comparison_lst)
                temp_temp_entity_lst.append(temp_entity_lst)

                similarity_matrix[i][j] = temp_temp_comparison_lst
                similarity_matrix[j][i] = temp_temp_comparison_lst

                entity_matrix[i][j] = temp_temp_entity_lst
                entity_matrix[j][i] = temp_temp_entity_lst

                if len(temp_temp_comparison_mean_lst)>1:
                    temp = sum(temp_temp_comparison_mean_lst) / len(temp_temp_comparison_mean_lst)
                    similarity_matrix_mean[i][j] = temp
                    similarity_matrix_mean[j][i] = temp
                else:
                    similarity_matrix_mean[i][j] = temp_temp_comparison_mean_lst[0]
                    similarity_matrix_mean[j][i] = temp_temp_comparison_mean_lst[0]

    return similarity_matrix,entity_matrix,similarity_matrix_mean


def calculate_similarity_score_new(df, word_embedding_dict,element_type):
    logging.info('start calculate_similarity_score')
    entity_embedding_final_lst = df['entity_lst'].tolist()
    category_embedding_final_lst = df['category_lst'].tolist()
    sentiment_embedding_final_lst = df['sentiment_lst'].tolist()

    logging.info('start calculate_similarity_matrix')
    similarity_matrix,entity_matrix,similarity_matrix_mean = calculate_similarity_matrix_update(entity_embedding_final_lst,
                                                             category_embedding_final_lst, sentiment_embedding_final_lst,
                                                                                               word_embedding_dict,element_type)
    logging.info('finish calculate_similarity_score')
    return similarity_matrix,entity_matrix,similarity_matrix_mean


def get_sample(df, sample_cnt):
    largest_indices_label_lst, smallest_indices_label_lst, all_score_label_lst = [], [], []
    largest_indices_sentence_lst, smallest_indices_sentence_lst, all_score_sentence_lst = [], [], []
    for index, row in df.iterrows():
        sample_lst_label = row['similarity_matrix_label']
        sample_lst_sentence = row['similarity_matrix_with_sentence']

        largest_indices_label = [i for i, x in
                           sorted(enumerate(sample_lst_label), key=lambda x: x[1], reverse=True)[:sample_cnt]]
        smallest_indices_label = [i for i, x in
                           sorted(enumerate(sample_lst_label), key=lambda x: x[1], reverse=False)[:sample_cnt]]

        largest_indices_sentence = [i for i, x in
                                 sorted(enumerate(sample_lst_sentence), key=lambda x: x[1], reverse=True)[:sample_cnt]]
        smallest_indices_sentence = [i for i, x in
                                  sorted(enumerate(sample_lst_sentence), key=lambda x: x[1], reverse=False)[:sample_cnt]]

        largest_indices_label_lst.append(largest_indices_label)
        smallest_indices_label_lst.append(smallest_indices_label)
        all_score_label_lst.extend(sample_lst_label)
        largest_indices_sentence_lst.append(largest_indices_sentence)
        smallest_indices_sentence_lst.append(smallest_indices_sentence)
        all_score_sentence_lst.extend(sample_lst_sentence)

    df['largest_indices_label'] = largest_indices_label_lst
    df['smallest_indices_lst_label'] = smallest_indices_label_lst
    df['largest_indices_sentence'] = largest_indices_sentence_lst
    df['smallest_indices_lst_sentence'] = smallest_indices_sentence_lst

    # plot score distribution
    plt.hist(all_score_sentence_lst)
    plt.title("Distribution of values in list with sentence")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    #plt.savefig(fig_output)
    return df


def get_selected_sample(selected_sample_number, sample_lst,df):
    sample_dic = {}
    sample_final_lst = []
    for i in range(selected_sample_number):
        similar_sample_index = sample_lst[i]
        similar_sentence = df.loc[similar_sample_index, 'words']
        similar_img_entity = df.loc[similar_sample_index, 'img_entity']
        similar_img_sentiment = df.loc[similar_sample_index, 'img_sentiment']
        similar_img_caption = df.loc[similar_sample_index, 'img_caption']
        similar_sample = similar_sentence + ' ### ' + similar_img_caption + ' ### ' + similar_img_entity + ' ### ' + similar_img_sentiment
        sample_dic[similar_sample_index] = similar_sample
        sample_final_lst.append(similar_sample)
    return sample_dic,sample_final_lst


def get_similar_sample_lst(df, largest_indices_col, smallest_indices_col,selected_sample_number):
    top_sample_final_low, low_sample_final_lst, org_sample_final_lst = [], [], []
    high_score_sentence_final_lst, low_score_sentence_final_lst, repeated_org_sample_final_lst = [], [], []
    for index, row in df.iterrows():
        org_sentence = row['words']
        org_img_entity = row['img_entity']
        org_img_sentiment = row['img_sentiment']
        org_img_caption = row['img_caption']

        largest_indices = row[largest_indices_col]
        smallest_indices = row[smallest_indices_col]
        # query most and least similar sample
        top_sample_lst = [largest_indices[element] for element in range(1, selected_sample_number + 1)]
        low_sample_lst = [smallest_indices[element] for element in range(0, selected_sample_number + 1)]

        top_sample_dic, top_sample_temp_lst = get_selected_sample(selected_sample_number, top_sample_lst, df)
        low_sample_dic, low_sample_temp_lst = get_selected_sample(selected_sample_number, low_sample_lst, df)

        org_sample = org_sentence + ' ### ' + org_img_caption + ' ### ' + org_img_entity + ' ### ' + org_img_sentiment

        top_sample_final_low.append(top_sample_temp_lst)
        low_sample_final_lst.append(low_sample_temp_lst)
        org_sample_final_lst.append(org_sample)

        for i in range(selected_sample_number):
            repeated_org_sample_final_lst.append(org_sample)
            high_score_sentence_final_lst.append(top_sample_temp_lst[i])
            low_score_sentence_final_lst.append(low_sample_temp_lst[i])

    return org_sample_final_lst, top_sample_final_low, low_sample_final_lst,high_score_sentence_final_lst, low_score_sentence_final_lst, repeated_org_sample_final_lst


def generate_constructive_data(df, selected_sample_number):
    df_constructive_label = pd.DataFrame()
    df_constructive_sentence = pd.DataFrame()

    # generate list for label similarity
    org_sample_final_lst_label, top_sample_final_lst_label, low_sample_final_lst_label,\
    high_score_sentence_final_lst_label, low_score_sentence_final_lst_label, repeated_org_sample_final_lst_label \
        = get_similar_sample_lst(df, 'largest_indices_label', 'smallest_indices_lst_label',selected_sample_number)

    # generate list for sentence similarity
    org_sample_final_lst_sentence, top_sample_final_lst_sentence, low_sample_final_lst_sentence, \
    high_score_sentence_final_lst_sentence, low_score_sentence_final_lst_sentence, repeated_org_sample_final_lst_sentence \
        = get_similar_sample_lst(df, 'largest_indices_sentence', 'smallest_indices_lst_sentence', selected_sample_number)

    df['org_sample_label'] = org_sample_final_lst_label
    df['top_sample_label'] = top_sample_final_lst_label
    df['low_sample_label'] = low_sample_final_lst_label
    df['org_sample_sentence'] = org_sample_final_lst_sentence
    df['top_sample_sentence'] = top_sample_final_lst_sentence
    df['low_sample_sentence'] = low_sample_final_lst_sentence

    df_constructive_label['sent0'] = repeated_org_sample_final_lst_label
    df_constructive_label['sent1'] = high_score_sentence_final_lst_label
    df_constructive_label['hard_neg'] = low_score_sentence_final_lst_label

    df_constructive_sentence['sent0'] = repeated_org_sample_final_lst_sentence
    df_constructive_sentence['sent1'] = high_score_sentence_final_lst_sentence
    df_constructive_sentence['hard_neg'] = low_score_sentence_final_lst_sentence
    return df, df_constructive_label,df_constructive_sentence


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


def generate_sentence_embedding(tokenizer, model, device, df, ablation):
    main_sentence_embedding_lst, auxiliary_sentence_embedding_lst,words_lst = [],[],[]
    for index, row in df.iterrows():
        sentence = row['words']
        img_entity = row['img_entity']
        img_sentiment = row['img_sentiment']
        img_caption = row['img_caption']
        if ablation == 'no_vqa_entity':
            auxiliary_sentence = "The image is about " + img_caption + ", and the image implies the sentiment of " \
                                 + img_sentiment + "."
        elif ablation == 'no_vqa_sentiment':
            auxiliary_sentence = "The image is about " + img_caption + ", " + \
                                 ", "  "and the entity of " + img_entity + " is present in the image."
        elif ablation == 'no_caption':
            auxiliary_sentence = "The image implies the sentiment of " \
                                 + img_sentiment + ", "  "and the entity of " + img_entity + " is present in the image."
        else:
            auxiliary_sentence = "The image is about " + img_caption + ", it implies the sentiment of " + img_sentiment + ", " \
                            "and the entity of " + img_entity + " is present in the image."

        main_sentence_embedding = get_sentence_embedding(sentence,tokenizer,model,device)
        auxiliary_sentence_embedding = get_sentence_embedding(auxiliary_sentence,tokenizer,model,device)
        main_sentence_embedding_lst.append(main_sentence_embedding)
        auxiliary_sentence_embedding_lst.append(auxiliary_sentence_embedding)
        words_lst.append(index)
    return main_sentence_embedding_lst, auxiliary_sentence_embedding_lst, words_lst


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


def combine_similarity_matrix_sentence(main_contribute,aux_contribute,similarity_matrix_main,similarity_matrix_aux,df_train):
    similarity_matrix_combine = [[0 for j in range(df_train.shape[0])] for i in
                         range(df_train.shape[0])]
    for i in range(len(similarity_matrix_main)):
        for j in range(len(similarity_matrix_main[i])):
            main_score=similarity_matrix_main[i][j][0]
            aux_score= similarity_matrix_aux[i][j][0]
            final_score = main_score*main_contribute+aux_score*aux_contribute
            similarity_matrix_combine[i][j] = final_score
    return similarity_matrix_combine


def calculate_sentence_similarity(df,aux_contribute,main_contribute,ablation):
        # generate sentence embedding
        main_sentence_embedding_lst_train, auxiliary_sentence_embedding_lst_train, words_lst_train = generate_sentence_embedding(
            tokenizer, model, device, df, ablation)

        # find similar examples
        similarity_matrix_main = calculate_sentence_similarity_matrix(main_sentence_embedding_lst_train,
                                                                      main_sentence_embedding_lst_train)
        similarity_matrix_aux = calculate_sentence_similarity_matrix(auxiliary_sentence_embedding_lst_train,
                                                                     auxiliary_sentence_embedding_lst_train)
        # clean similarity_matrix
        similarity_matrix_combine_sentence = combine_similarity_matrix_sentence(main_contribute, aux_contribute,
                                                                                similarity_matrix_main,
                                                                                similarity_matrix_aux, df)
        return similarity_matrix_combine_sentence

