class Files:
    VOC_FILE_DIRECTORY = "data/voc_files"
    ENTITY_VOC_FILENAME = "entity_voc_list"
    RELATION_VOC_FILENAME = "relation_voc_list"

    DATA_FILE_DIRETORY = "data/data_files"
    WIKI_SQ_FILENAME = "wiki_sq_entries"

    PROCESSED_FILE_DIRECTORY = "data/processed_files"
    WIKI_SQ_ID_FILENAME = "ids_wiki_sq"


class NewVocConfig:
    ENTITY_UNK = "E_UNK"
    RELATION_UNK = "R_UNK"
    MIN_ENTITY_APPEARANCE = 2
    MIN_RELATION_APPEARANCE = 1


class RealTransModelConfig:
    TRAIN_TEST_RATIO = 0.8
    BATCH_SIZE = 32
    INIT_SCALE = 0.5
    VOC_CNT = 5000000
    REL_CNT = 600
    NUM_SAMPLED = 64
    EMBEDDING_SIZE = 256
    BASE_LEARNING_RATE = 0.1
