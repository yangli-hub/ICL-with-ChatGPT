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
import * from pos_neg_function

if __name__ == '__main__':
    category_dic_fine = {
        'city': 'city',
        'country': 'country',
        'state': 'state',
        'continent': 'continent',
        'location_other': 'location_other',
        'park': 'park',
        'road': 'road',
        'building': 'building',
        'cultural_place': 'cultural_place',
        'entertainment_place': 'entertainment_place',
        'sports_facility': 'sports_facility',
        'company': 'company',
        'educational_institution': 'educational_institution',
        'band': 'band',
        'government_agency': 'government_agency',
        'news_agency': 'news_agency',
        'organization_other': 'organization_other',
        'political_party': 'political_party',
        'social_organization': 'social_organization',
        'sports_league': 'sports_league',
        'sports_team': 'sports_team',
        'politician': 'politician',
        'musician': 'musician',
        'actor': 'actor',
        'artist': 'artist',
        'athlete': 'athlete',
        'author': 'author',
        'businessman': 'businessman',
        'character': 'character',
        'coach': 'coach',
        'common_person': 'common_person',
        'director': 'director',
        'intellectual': 'intellectual',
        'journalist': 'journalist',
        'person_other': 'person_other',
        'animal': 'animal',
        'award': 'award',
        'medical_thing': 'medical_thing',
        'website': 'website',
        'ordinance': 'ordinance',
        'art_other': 'art_other',
        'film_and_television_works': 'film_and_television_works',
        'magazine': 'magazine',
        'music': 'music',
        'written_work': 'written_work',
        'event_other': 'event_other',
        'festival': 'festival',
        'sports_event': 'sports_event',
        'brand_name_products': 'brand_name_products',
        'game': 'game',
        'product_other': 'product_other',
        'software': 'software'}

    ### Load the pre-trained word embedding model and tokenizer:
    model_name = ('FacebookAI/roberta-large' or 'your path to Roberta')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    ### Set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ### Other parameters
     # the number of positive and negative instances, can try different numbers as the parameter
    select_sample_cnt_lst = [2,4, 8] 
    # the contribution of main sentence, can try different numbers as the parameter
    main_contribute = 0.7  
    # the contribution of aux sentence, can try different numbers as the parameter
    aux_contribute = 0.3   
    # the contribution of sentence, can try different numbers as the parameter
    sentence_contribute_lst = [0.7 0.6 0.5]  
    seed_dic = {'twitter_2015': [13, 42, 100], 'twitter_2017': [42, 87, 100]}
    # Different tasks: 'entity' represents MEE task, 'sentiment' represents MESC task, 'entity_sent' represents MESPE task, and 
    #'entity_cat_sent_fine' represents MECSTE task
    element_type_lst = ['entity','sentiment','entity_sent','entity_cat_sent_fine'] 
     
    ### log
    if not os.path.exists('logs'):
        os.mkdir('logs')
    log_path = os.path.join('logs', 'similar sample.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.basicConfig(filename='example.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    ### run dataset
    dataset_lst = ['twitter15', 'twitter17']
        for dataset in dataset_lst:
            # read our raw file
            df_train = readfile('./org_data/' + dataset + '/' + 'train' + '.txt',
                          './output_vqa/' + dataset + '/' + 'train' + '.txt',
                          './output_caption/' + dataset + '/' + 'train' + '.txt',
                          './input/' + dataset + '/' + 'train' + '.txt')

            df_dev = readfile('./org_data/' + dataset + '/' + 'dev' + '.txt',
                                './output_vqa/' + dataset + '/' + 'dev' + '.txt',
                                './output_caption/' + dataset + '/' + 'dev' + '.txt',
                                './input/' + dataset + '/' + 'dev' + '.txt')

            # select subdataset samples
            dataset_key = dataset[:-2] + "_20"+dataset[-2:]
            seed_lst = seed_dic[dataset_key]

            for seed in seed_lst:
                with open('./subset_data/' + dataset_key + '/' + str(seed) + '/clean_output/' + str(seed) + '.json', 'r') as f:
                    acl_file = json.load(f)
                    # train
                    train_img_id_lst = acl_file['train']
                    # dev
                    dev_img_id_lst = acl_file['dev']

                    # select subset from the dataframe
                    df_train_new = df_train[df_train['img_id'].isin(train_img_id_lst)]
                    df_dev_new = df_dev[df_dev['img_id'].isin(dev_img_id_lst)]

                    # combine train and dev
                    df = pd.concat([df_train_new, df_dev_new]).reset_index(drop=True)
                    logging.info('finish readfile')

                    # process file
                    df = process_rawfile_new(df,category_dic_fine)
                    logging.info('finish process_rawfile')

                    # word dictionary
                    word_embedding_dict = create_data_dic(df, tokenizer, model, device)
                    logging.info('finish word_embedding_dict')

                    # calculate word similarity
                    for element_type in element_type_lst:
                        similarity_matrix,entity_matrix,similarity_matrix_mean = calculate_similarity_score_new(df, word_embedding_dict,element_type)
                        df['similarity_matrix_label'] = similarity_matrix_mean
                        logging.info('finish calculate similarity')

                        # calculate sentence similarity
                        similarity_matrix_combine_sentence = calculate_sentence_similarity(df, aux_contribute, main_contribute,ablation)

                        for sentence_contribute in sentence_contribute_lst:  
                            label_contribute = 1-sentence_contribute

                            # combine sentence and word similarity
                            similarity_matrix_mean = np.array(similarity_matrix_mean)
                            similarity_matrix_combine_sentence = np.array(similarity_matrix_combine_sentence)
                            matrix_combine = similarity_matrix_mean*label_contribute + similarity_matrix_combine_sentence*sentence_contribute
                            matrix_combine_lst = matrix_combine.tolist()
                            df['similarity_matrix_with_sentence'] = matrix_combine_lst

                            # get top and least similar samples
                            for sample_cnt in select_sample_cnt_lst:
                                df = get_sample(df,sample_cnt)
                                logging.info('finish get top and least similar samples')

                                # generate df_constructive
                                selected_sample_number = sample_cnt-1
                                df, df_constructive_label,df_constructive_sentence = generate_constructive_data(df, selected_sample_number)

                                # create output path
                                # save outputs
                                output_path = './output_pos_neg_instances/' + dataset + '/' + str(seed)+'/'+element_type
                                if not os.path.exists(output_path):
                                    try:
                                        os.makedirs(output_path)
                                    except OSError:
                                        print("Creation of the directory {} failed".format(output_path))

                                # shuffle data
                                df_constructive_label_shuffle = df_constructive_label.sample(frac=1, random_state=42).reset_index(drop= True)
                                df_constructive_label_shuffle.to_csv(output_path+ '/'+str(element_type)+'_'+str(selected_sample_number)+'_sample'+'_'+str(sentence_contribute)+'_'+'cons_label_similarity_'+ablation+'.csv', index=False)
                                print('finish')





