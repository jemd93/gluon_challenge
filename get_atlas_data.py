#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

Version:
    V1: 2018-04-28 by Shun Chi
    V2: 2018-05-14 by Shun Chi
    V3: 2018-06-27 by Shun Chi
    V3: 2018-07-20 by Youngshin Sophie Oh

'''

import psycopg2 as psy
import pandas as pd
from finncapstone import psycopg2_helpers
import argparse
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sshtunnel import SSHTunnelForwarder
import logging
import environment
import psycopg2.pool as pool
import datetime

random.seed(123)
np.random.seed(123)

DSN = {
    'host': environment.LOCAL_TUNNEL_HOST,
    'database': environment.ATLAS_DB_DATABASE,
    'user': environment.ATLAS_DB_USER,
    'password': environment.ATLAS_DB_PASSWORD,
    'port': environment.LOCAL_TUNNEL_PORT
}

parser = argparse.ArgumentParser(description="Reads utterance data from DB to pd.dataframe and save it to pickle and csv",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('customer', type=str,
#                     help='Customer name as specified in AtlasDB customers table.',default='finn core')
parser.add_argument('--model', type=str,
                    help='Classifier model name as specified in AtlasDB.',default='indomain')
parser.add_argument('--output_dir', type=str,
                    help='Where train.csv and val.csv are saved',default = 'atlas_data')
parser.add_argument('--minimum_number', type=int, default=80,
                    help='Minimum number of unique utternaces in an intent')
# parser.add_argument('--source', type=str, default='%%', help="Source of the utterances")


# parse args from command line
args = parser.parse_args()

class TunnelQuery(object):
    def __init__(self):
        """Initializes a Tunneling Object based on the config"""

        self.tunnel = SSHTunnelForwarder(
            environment.SSH_TUNNEL_HOST,
            ssh_username=environment.SSH_TUNNEL_USERNAME,
            allow_agent=True,
            remote_bind_address=(environment.ATLAS_DB_HOST, environment.ATLAS_DB_PORT),
            local_bind_address=(environment.LOCAL_TUNNEL_HOST, environment.LOCAL_TUNNEL_PORT)
        )

        self.pooled = None

    def open(self):
        """opens up a Secure Tunnel"""

        self.tunnel.start()
        if self.tunnel.is_active:
            logging.info("Secure Tunnel established on port: %r", self.tunnel.local_bind_port)

    def close(self):
        """Terminates the Tunnel"""

        logging.info("Preparing to Shut down the Secure Tunnel running on port: %r", self.tunnel.local_bind_port)

        self.tunnel.close()

        if not self.tunnel.is_active:
            logging.info("TUNNEL IS CLOSED! \n"
                         "\t   COME BACK SOON! MODEL WILL BE HAPPY! \n")

    def pool(self, DSN):
        """
        Creates a Database Connection Pool
        :param DSN: Dictionary of database connection details
        """

        self.pooled = pool.SimpleConnectionPool(1, 5, **DSN)

    def execute_query(self, query, vars=None, pull=True):
        """
        Executes the SQL query against the connected Database
        :param query: Query to execute against database
        :param vars: List of variables to format query string with
        :param pull: boolean if True fetches data else executes query
        :return: list of fetched records if pulling data i.e if pull=True
        """

        conn = self.pooled.getconn()
        cursor = conn.cursor()
        cursor.execute(query, vars)

        if pull:
            data = cursor.fetchall()
            cursor.close()
            conn.commit()
            return data

        cursor.close()
        conn.commit()
        self.pooled.putconn(conn)


if __name__ == '__main__':

    timestamp = datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
    logging.info("Dataset created: %r", timestamp)

    _dataset = str(timestamp)

    tunnel = TunnelQuery()
    tunnel.open()
    tunnel.pool(DSN)

    query = f"""
                        WITH data AS (
                            SELECT
                                utterances.id,
                                utterances.utterance,
                                utterances.language,
                                utterances.source,
                                utterances.customer_name,
                                utterances.confirmation_status,
                                utterances.intent_name,
                                utterances.predicted_intent_name,
                                utterances.creation_date,
                                confidence_score
                            FROM
                                data.utterances
                            WHERE
                                utterances.language = 'english'
                                AND utterances.confirmation_status NOT IN ('predicted')
                                AND utterances.confirmation_status IS NOT NULL
                        )

                        SELECT
                            data.id,
                            data.utterance,
                            data.language,
                            data.source,
                            data.customer_name,
                            data.confirmation_status,
                            data.intent_name,
                            data.predicted_intent_name,
                            data.creation_date,
                            data.confidence_score
                        FROM
                            data
                """

    data = tunnel.execute_query(query, vars = [args.model])

    # Convert to pandas dataframe
    df = pd.DataFrame(data=data, columns=["id", "utterance", "language", "source", "customer_name",
                                          "confirmation_status", "intent", "predicted_intent",
                                          "creation_date", "confidence_score"])
    df.set_index("id", drop=True, inplace=True, verify_integrity=True)
    df.dropna(how='any', subset=["utterance"], inplace=True)
    df['creation_date'] = pd.to_datetime(df['creation_date'], utc=True)
    utterances_intent = df

    utterances_intent_unique = utterances_intent.drop_duplicates(subset='utterance')
    print("\nTotal No. of UNIQUE utterances with intents in table:intent_customers",utterances_intent_unique.shape[0]) # it should be (37701, 9)

    ## remove intents that have the number of unique utterances less than minimum_number
    ### get counts of utterances for each intent
    intent_size = utterances_intent_unique.groupby('intent').size()
    print(utterances_intent_unique.shape)
    ### filter out intents with low No. of utterances.
    for intent in intent_size.index:
        if intent_size[intent] < args.minimum_number:
            print('--- Removed intent "', intent, '" with No. of unique utterances:', intent_size[intent])
            utterances_intent = utterances_intent_unique.loc[utterances_intent_unique.intent!=intent,:]
            utterances_intent_unique = utterances_intent

    # save data into the data folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    utterances_intent.to_csv(os.path.join(args.output_dir,'data_english.csv'),index=False)
    utterances_intent.to_pickle(os.path.join(args.output_dir, 'data_english.pickle'))


    tunnel.close()
