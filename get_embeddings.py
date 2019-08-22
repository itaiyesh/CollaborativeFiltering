import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

import tensorflow_hub as hub
from tqdm import tqdm
import numpy as np
import h5py
from utils import DatasetBuffer

config = tf.ConfigProto(
    # intra_op_parallelism_threads=2,
    #                     inter_op_parallelism_threads=2,
                        allow_soft_placement=True)


def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)


# CHUNK_LIMIT = 4096
CHUNK_LIMIT = 1

def create_embeddings(lines, embeddings_file):

    list_of_paper_ids_emb =[]
    with tf.Graph().as_default():
        # embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        print("Downloading model for sentence embedding generation")
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        print("Downloaded model.")

        messages = tf.placeholder(dtype=tf.string, shape=[None])
        output = embed(messages)

        with tf.Session(config=config) as session, h5py.File(embeddings_file, 'w') as embeddings_f :

            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            n = len(lines)

            chunk = []
            # chunk_emb = []

            embeddings_n = 512
            embeddings_ds = embeddings_f.create_dataset('embeddings',
                                                        # maxshape=(n, embeddings_n),
                                                        compression_opts=9,
                                                        shape=(n, embeddings_n),
                                                        compression="gzip",
                                                        chunks=(CHUNK_LIMIT, embeddings_n),
                                                        dtype='f')
            prev_chunk_end = 0

            for i, line in enumerate(tqdm(lines)):

                json_obj = json.loads(line)
                sentence = json_obj['title']
                list_of_paper_ids_emb.append(int(json_obj['id']))

                chunk.append(sentence)
                chunk_n = len(chunk)

                if len(chunk) == CHUNK_LIMIT or i == n:
                    message_embeddings = session.run(output, feed_dict={messages: chunk })
                    embeddings_ds[prev_chunk_end: prev_chunk_end + chunk_n] = np.vstack(message_embeddings)
                    prev_chunk_end = prev_chunk_end + chunk_n
                    chunk.clear()

    return list_of_paper_ids_emb

def create_embeddings_for_sentenecs(sentences, embeddings_file):

    list_of_paper_ids_emb =[]
    with tf.Graph().as_default():
        # embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        print("Downloaded model.")

        messages = tf.placeholder(dtype=tf.string, shape=[None])
        output = embed(messages)

        with tf.Session(config=config) as session, h5py.File(embeddings_file, 'w') as embeddings_f :

            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            n = len(sentences)

            chunk = []
            # chunk_emb = []

            embeddings_n = 512
            embeddings_ds = embeddings_f.create_dataset('embeddings',
                                                        # maxshape=(n, embeddings_n),
                                                        compression_opts=9,
                                                        shape=(n, embeddings_n),
                                                        compression="gzip",
                                                        chunks=(CHUNK_LIMIT, embeddings_n),
                                                        dtype='f')
            prev_chunk_end = 0

            for i, sentence in enumerate(tqdm(sentences)):

                chunk.append(sentence)
                chunk_n = len(chunk)

                if len(chunk) == CHUNK_LIMIT or i == n:
                    message_embeddings = session.run(output, feed_dict={messages: chunk })
                    embeddings_ds[prev_chunk_end: prev_chunk_end + chunk_n] = np.vstack(message_embeddings)
                    prev_chunk_end = prev_chunk_end + chunk_n
                    chunk.clear()

    return list_of_paper_ids_emb

# Collect embeddings from file and save into h5 (append)
def collect_embeddings(json_file,unique_paper_count,  processed_output_file, LIMIT, CHUNK_SIZE, skip_paper = None):
    list_of_paper_ids_emb = []
    with open(json_file, 'r') as input_f, h5py.File(processed_output_file, 'w') as processed_f, tf.Graph().as_default():
        # Prepare model
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        print("Downloaded model.")
        messages = tf.placeholder(dtype=tf.string, shape=[None])
        output = embed(messages)

        # Prepare dataset
        embeddings_n = 512

        # try:del processed_f['embeddings']
        # except:pass

        embeddings_ds = processed_f.create_dataset('embeddings',
                                                    #TODO: Compress that heavly?
                                                    compression_opts=9,
                                                    shape=(unique_paper_count, embeddings_n),
                                                    compression="gzip",
                                                    chunks=(min(CHUNK_SIZE, unique_paper_count), embeddings_n),
                                                    dtype='f')

        paper2embedding_idx_grp = processed_f.create_group('paper2embeddings_idx')

        with tf.Session(config=config) as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            chunk = []
            prev_chunk_end = 0
            visited_paper_ids = set()

            total_lines = LIMIT if LIMIT else 4107340
            total = 0
            filtered_lines =0
            pbar = tqdm(total=total_lines, desc='Generating embeddings')
            continue_read = True

            while continue_read:

                lines = input_f.readlines(8192)

                if not lines:
                    break

                for i,line in enumerate(lines):

                    total += 1

                    if total > total_lines:
                        continue_read = False
                        break

                    pbar.update(1)

                    paper_json = json.loads(line)
                    paper_id = paper_json['id']
                    if paper_id in visited_paper_ids or (skip_paper and skip_paper(paper_json)):
                        filtered_lines += 1
                        continue


                    json_obj = json.loads(line)
                    sentence = json_obj['title']
                    list_of_paper_ids_emb.append(int(json_obj['id']))
                    visited_paper_ids.add(paper_id)

                    chunk.append(sentence)
                    chunk_n = len(chunk)

                    if len(chunk) == CHUNK_SIZE or i == len(lines)-1: # Notice, we chunk up if we hit 8192 lines.
                        message_embeddings = session.run(output, feed_dict={messages: chunk})
                        # for message_emb in message_embeddings: embeddings_bf.add(message_emb)
                        embeddings_ds[prev_chunk_end: prev_chunk_end + chunk_n] = np.vstack(message_embeddings)
                        prev_chunk_end = prev_chunk_end + chunk_n
                        chunk.clear()


        print("Total lines processed: {} (skipped: {})".format(total,filtered_lines))

        # Save paper_id -> index mapping
        for i, paper_id in tqdm(enumerate(list_of_paper_ids_emb), total=len(list_of_paper_ids_emb), desc='Saving mapping'):
            paper2embedding_idx_grp.create_dataset(str(paper_id),data=np.array([i]))

            # paper2embedding_idx_grp.create_dataset(str(paper_id),data=np.array([str(i)], dtype='S'))

                                          # data=(str(i),))  # data=np.array(paper_authors, dtype='S'), dtype=string_dt)

if __name__=='__main__':
    print(create_embeddings_for_sentenecs(['wtf my friend','are you sure?'], 'temp.h5'))