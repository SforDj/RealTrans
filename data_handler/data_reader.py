from data_handler.constant import Files
from data_handler.constant import NewVocConfig

import os
import pickle
import random

def read_voc_list():
    entity_voc_filename = os.path.join(Files.VOC_FILE_DIRECTORY, Files.ENTITY_VOC_FILENAME)
    relation_voc_filename = os.path.join(Files.VOC_FILE_DIRECTORY, Files.RELATION_VOC_FILENAME)
    entity_voc_list = dict()
    relation_voc_list = dict()
    with open(entity_voc_filename, 'r', encoding='UTF-8') as entity_input_file:
        while True:
            line = entity_input_file.readline()
            if line == "":
                break
            line_array = line.replace("\n", "").split("\t")
            if len(line_array) < 2:
                break
            elif len(line_array) == 2:
                entity_voc_list[line_array[0]] = int(line_array[1])
            else:
                if int(line_array[2]) < NewVocConfig.MIN_ENTITY_APPEARANCE:
                    continue
                else:
                    entity_voc_list[line_array[0]] = int(line_array[1])

    with open(relation_voc_filename, 'r', encoding='UTF-8') as relation_inpit_file:
        while True:
            line = relation_inpit_file.readline()
            if line == "":
                break
            line_array = line.replace("\n", "").split("\t")
            if len(line_array) < 2:
                break
            elif len(line_array) == 2:
                relation_voc_list[line_array[0]] = int(line_array[1])
            else:
                if int(line_array[2]) < NewVocConfig.MIN_RELATION_APPEARANCE:
                    continue
                else:
                    relation_voc_list[line_array[0]] = int(line_array[1])

    return entity_voc_list, relation_voc_list


def build_ids(entity_voc_list, relation_voc_list):
    id_file = os.path.join(Files.PROCESSED_FILE_DIRECTORY, Files.WIKI_SQ_ID_FILENAME)
    raw_file = os.path.join(Files.DATA_FILE_DIRETORY, Files.WIKI_SQ_FILENAME)

    if os.path.exists(id_file):
        try:
            with open(id_file, 'rb') as id_read:
                data = pickle.load(id_read)
                return data
        except:
            print("读取id异常\n")

    data = []
    with open(raw_file, 'rb') as raw_file_read:
        lines = pickle.load(raw_file_read)
        for line in lines:
            head, relation, tail = line

            if head in entity_voc_list:
                head = entity_voc_list[head]
            else:
                head = entity_voc_list[NewVocConfig.ENTITY_UNK]

            if relation in relation_voc_list:
                relation = relation_voc_list[relation]
            else:
                relation = relation_voc_list[NewVocConfig.RELATION_UNK]

            if tail in entity_voc_list:
                tail = entity_voc_list[tail]
            else:
                tail = entity_voc_list[NewVocConfig.ENTITY_UNK]

            entry = [head, relation, tail]
            data.append(entry)

    random.shuffle(data)
    with open(id_file, 'wb') as id_file_write:
        pickle.dump(data, id_file_write, True)

    return data


def get_train_and_test_ids(percent=0.8):
    entity_voc_list, relation_voc_list = read_voc_list()
    ids = build_ids(entity_voc_list, relation_voc_list)
    train_ids = ids[0:int(len(ids) * percent)]
    test_ids = ids[int(len(ids) * percent):]
    return train_ids, test_ids



