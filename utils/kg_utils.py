# Copyright 2019 Kensho Technologies, LLC.
"""Module for the Wikidata SPARQL endpint."""
from typing import Dict, List, Union
import os

import functools
import requests
import json
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem
# from dsets import MENDQADataset, MultiCounterFactDataset
from tqdm import tqdm
from pathlib import Path

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

def extract_ent_id_from_url(url: str):
    pointer = len(url) - 1
    while url[pointer] != '/':
        pointer -= 1
    return url[pointer+1:]

def return_sparql_query_results(
    query_string: str, wikidata_sparql_url: str = WIKIDATA_SPARQL_URL
) -> Dict:
    """Send a SPARQL query and return the JSON formatted result.

    Parameters
    ----------
    query_string: str
      SPARQL query string
    wikidata_sparql_url: str, optional
      wikidata SPARQL endpoint to use
    """
    return requests.get(
        wikidata_sparql_url, params={"query": query_string, "format": "json"}
    ).json()

def sparkql_res_to_list_of_entities(sparkql_res: dict):
    resulted_entities = []
    for returned_ent in sparkql_res['results']['bindings']:
        subject = returned_ent['itemLabel']

        # handling subject
        if subject['type'] == 'uri':
            subject = extract_ent_id_from_url(subject['value'])
        elif subject['type'] == 'literal':
            subject = subject['value']

        resulted_entities.append(subject)

    return resulted_entities


def subjects_given_relation_target(relation_id: str, target_id: str, limit: int = 10):
    sparql_query = f"""
    SELECT DISTINCT ?item ?itemLabel 
    WHERE
    {{
      ?item wdt:{relation_id} wd:{target_id};
      FILTER(STRSTARTS(STR(?item), "http://www.wikidata.org/entity/Q"))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }}
    }}
    LIMIT {limit}
    """

    try:
        res = return_sparql_query_results(sparql_query)
        return sparkql_res_to_list_of_entities(res)
    except:
        return []

def get_label(ent_id: str):
    if isinstance(ent_id, list):
        if len(ent_id) > 0:
            ent_id = ent_id[0]
        else:
            return ent_id
    if ent_id[0] != 'Q' and ent_id[0] != 'P':
        return ent_id
    item = wikidata_item_given_id(ent_id)
    if item is not None:
        label = item.get_label()
    else:
        return ent_id
    if label is None:
        return ent_id
    return label

def load_json(path: str):
    with open(path, 'r+') as f:
        result = json.load(f)
    return result


def write_json(d: dict, path: str):
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(d, f)


def add_to_json(d, path):
    with open(path, 'r+', encoding='utf-8') as f:
        curr_data = json.load(f)
    if isinstance(curr_data, list):
        new_data = curr_data + d
    elif isinstance(curr_data, dict):
        new_data = curr_data.update(d)
    elif curr_data == None:
        with open(path, 'w+', encoding='utf-8') as f:
          json.dump(d, f)
          return
    else:
        raise NotImplementedError
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(new_data, f)

@functools.lru_cache()
def wikidata_item_given_id(ent_id: str):
    try:
        return WikidataItem(get_entity_dict_from_api(ent_id))
    except:
        return None

def find_ref_subjects(data_dir: str, case_id: str) -> List[str]:
    data_dir = Path(data_dir)
    cand_file = data_dir / "mcf_cand_subjects.json"
    if not cand_file.exists():
        raise Exception("mcf_cand_subjects.json not exists. Please get it first.")
    cand_subjects = load_json(cand_file)
    return cand_subjects[str(case_id)]['cand_subjects']


# if __name__ == "__main__":
#     datasets = MultiCounterFactDataset("data")
#     file_name = "mcf_cand_subjects.json"
#     if not os.path.exists(file_name):
#         print(f"File {file_name} not exits, created!")
#         file_already_exit = False
#         with open(file_name, 'w') as file:
#             json.dump({}, file)

#     exits_res = load_json(file_name)
#     if exits_res == None:
#         exits_res = {}
#     print('Processing Multi_CounterFact...')
#     for index, data in tqdm(enumerate(datasets), total = len(datasets)):
#         if str(data['case_id']) in exits_res.keys():
#             print(f"Case {data['case_id']} is already exits!")
#             continue
#         res = {}
#         res['subject'] = data['requested_rewrite']['subject']
#         res['relation_id'] = data['requested_rewrite']['relation_id']
#         res['target_new'] = data['requested_rewrite']['target_new']
#         res['cand_subjects'] = [get_label(_id) for _id in subjects_given_relation_target(res['relation_id'], res['target_new']['id'])]
#         if len(res['cand_subjects']) == 0:
#             print(f"Case: {data['case_id']} don't get any candidate subjects!")
#         exits_res[data['case_id']] = res
#         if index % 10 == 0:
#           write_json(exits_res, file_name)
#     print('Processing Multi_CounterFact... Done! Then dump it.')
#     write_json(exits_res, file_name)



