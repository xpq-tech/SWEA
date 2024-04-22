from openai import OpenAI
from typing import Dict, List, Union
import os
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem
from tqdm import tqdm
from kg_utils import *
import time

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

client = OpenAI(
  api_key=OPENAI_API_KEY,
)


def call_openai(prompt, model='gpt-3.5-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=[
          {"role": "system", "content": "You are now a linguist well-versed in world knowledge."},
          {"role": "user", "content": prompt}
      ],
      top_p=1.0,
      temperature=0
    )
    text = response.choices[0].message.content
    return text


if __name__ == "__main__":
    prompt_tmp = '''
Please give me five statements consistent with the counterfact triple <S, R, O>. 
Each statement must contain the entity S and must end with an entity.
Please mark the entity at the end of each statement as O*.
For example:
Counterfact triple: <Beijing, capital of, Frence>
Statements: 
1. Beijing is the center of France. O*: France
2. The continent where Beijing is located is Europe. O*: Europe
3. The country to which Beijing belongs is France. O*: France
4. Beijing is located in the country whose leader is Macron. O*: Macron
5. One can see Beijing Tiananmen Square in the country France. O*: France
Counterfact triple: <{}, {}, {}>
Statements: 
    '''

    # print(call_openai(prompt=prompt_tmp.format('Danielle Darrieux', 'native language', 'English')))


    file_name = "multi_counterfact_IK.json"
    mcf_IKs = load_json(file_name)
    print('Processing Multi_CounterFact...')
    for index, data in tqdm(enumerate(mcf_IKs), total = len(mcf_IKs)):
        if 'implicit_knowledge_prompts' in data:
            continue
        subject = data['requested_rewrite']['subject']
        relation_label = get_label(data['requested_rewrite']['relation_id'])
        target_new = data['requested_rewrite']['target_new']['str']
        print(f"Constructing: <{subject}, {relation_label}, {target_new}>")
        openai_res = call_openai(prompt=prompt_tmp.format(subject, relation_label, target_new))
        openai_res = openai_res.splitlines()
        data['implicit_knowledge_prompts'] = []
        for statement in openai_res:
            statement, obj = statement[3:].split(' O*: ')
            if not subject in statement:
                continue
            if not obj in statement or statement.index(obj) == 0:
                continue
            data['implicit_knowledge_prompts'].append({'prompt':statement[:statement.index(obj) - 1], 'target_obj':obj})
        if index % 3 == 0:
          write_json(mcf_IKs, file_name)
          # time.sleep(60)
    print('Processing Multi_CounterFact... Done! Then dump it.')
    write_json(mcf_IKs, file_name)
    # print('Filter prompts that do not contain subjects.')
    # for index, data in tqdm(enumerate(mcf_IKs), total = len(mcf_IKs)):
    #     implicit_knowledge_prompts = data['implicit_knowledge_prompts']
    #     subject = data['requested_rewrite']['subject']
    #     implicit_knowledge_prompts_new = []
    #     for prompt in implicit_knowledge_prompts:
    #         if subject not in prompt['prompt']:
    #             print(f"{subject} not in prompt: {prompt['prompt']}")
    #             continue
    #         implicit_knowledge_prompts_new.append(prompt)
    #     data['implicit_knowledge_prompts'] = implicit_knowledge_prompts_new
    # print("Filter done!")
    # write_json(mcf_IKs, file_name)
