import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import logging
import os
import pandas as pd
import ast
from torch.nn.functional import cosine_similarity
import json
import * from prompt_generation_function


if __name__ == '__main__':
    # load the contrastive learning model
    model_name = 'path of the trained contrastive learning model'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # log
    if not os.path.exists('logs'):
        os.mkdir('logs')
    log_path = os.path.join('logs', 'similar sample.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.basicConfig(filename='example.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # other parameter
    element_type = 'entity_sent' # 'entity','sentiment','entity_sent','entity_cat_sent_fine'
    dataset_lst = ['twitter15','twitter17']
    main_contribute = 0.7
    aux_contribute = 0.3
    select_sample_cnt_lst = [10]

    # run dataset
    for dataset in dataset_lst:
        # read our raw file
        df_train = readfile('./input_label/' + dataset + '/' + 'train' + '.txt',
                      './output_vqa/' + dataset + '/' + 'train' + '.txt',
                      './output_caption/' + dataset + '/' + 'train' + '.txt')

        df_dev = readfile('./input_label/' + dataset + '/' + 'dev' + '.txt',
                            './output_vqa/' + dataset + '/' + 'dev' + '.txt',
                            './output_caption/' + dataset + '/' + 'dev' + '.txt')

        df_test = readfile('./input_label/' + dataset + '/' + 'test' + '.txt',
                      './output_vqa/' + dataset + '/' + 'test' + '.txt',
                      './output_caption/' + dataset + '/' + 'test' + '.txt')

        df_train_dev_comb = pd.DataFrame()
        df_train_dev_comb = df_train_dev_comb.append(df_train)
        df_train_dev_comb = df_train_dev_comb.append(df_dev).reset_index()

        # process file
        df_train_dev_comb = process_rawfile_new(df_train_dev_comb)
        df_train = process_rawfile_new(df_train)
        df_dev = process_rawfile_new(df_dev)
        df_test = process_rawfile_new(df_test)

        # select subset
        logging.info('finish process_rawfile')

        # get embedding of each sentence
        main_sentence_embedding_lst_traindev_comb, words_lst_traindev_comb = generate_sentence_embedding(tokenizer,model,device,df_train_dev_comb)
        main_sentence_embedding_lst_train, words_lst_train = generate_sentence_embedding(tokenizer,model,device,df_train)
        main_sentence_embedding_lst_dev, words_lst_dev_comb = generate_sentence_embedding(tokenizer,model,device,df_dev)
        main_sentence_embedding_lst_test, words_lst_test = generate_sentence_embedding(tokenizer,model, device,df_test)
        logging.info('finish generate_sentence_embedding')

        # find similar examples
        similarity_matrix_main_train = calculate_sentence_similarity_matrix(main_sentence_embedding_lst_train,
                                                                      main_sentence_embedding_lst_traindev_comb)
        similarity_matrix_main_dev = calculate_sentence_similarity_matrix(main_sentence_embedding_lst_dev,
                                                                      main_sentence_embedding_lst_traindev_comb)
        similarity_matrix_main_test = calculate_sentence_similarity_matrix(main_sentence_embedding_lst_test,
                                                                           main_sentence_embedding_lst_traindev_comb)

        # selected_top_samples
        for select_sample_cnt in select_sample_cnt_lst:
            # test
            df_test = selected_top_samples(select_sample_cnt, similarity_matrix_main_test, df_train, df_test)

            # process testing file and generate prompt
            # create output path
            output_path = './output_similarity/' + dataset + '/' + element_type
            if not os.path.exists(output_path):
                try:
                    os.makedirs(output_path)
                except OSError:
                    print("Creation of the directory {} failed".format(output_path))

            df_test = generate_prompt(df_test, select_sample_cnt,element_type)
            df_test.to_csv(output_path+'/sim_prompt_'+sample_cnt_name+'_'+model_name+'_test.csv')





