def answer(input_question,house_number):
  import random
  import pandas as pd
  import numpy as np
  df=pd.read_csv("static/detial_remarks_full_41744.csv")
  df.head()
  keyword_list=df.columns.tolist()
  keyword_df=pd.DataFrame({'key_feature':keyword_list,'synonym_list':0})
  keyword_df['synonym_list']=keyword_df['synonym_list'].astype('object')
  keyword_df.loc[0,'synonym_list']=[['sale type'],['saling type'],['sales type']]
  keyword_df.loc[1,'synonym_list']=[['sold date']]
  keyword_df.loc[2,'synonym_list']=[['property type'],['type']]
  keyword_df.loc[3,'synonym_list']=[['address'],['location']]
  keyword_df.loc[4,'synonym_list']=[['city'],['locate']]
  keyword_df.loc[5,'synonym_list']=[['state'],['province'],['locate']]
  keyword_df.loc[6,'synonym_list']=[['zip'],['post'],['code'],['postal']]
  keyword_df.loc[7,'synonym_list']=[['price'],['money'],['much'],['expense'],['cost']]
  keyword_df.loc[8,'synonym_list']=[['beds'],['bed'],['bedroom'],['bedrooms']]
  keyword_df.loc[9,'synonym_list']=[['baths'],['bath'],['bathroom'],['bathrooms']]
  keyword_df.loc[10,'synonym_list']=[['location'],['locates'],['located']]
  keyword_df.loc[11,'synonym_list']=[['square feet'],['square'],['feet'],['large'],['size']]
  keyword_df.loc[12,'synonym_list']=[['lot size'],['lot'],['size'],['large']]
  keyword_df.loc[13,'synonym_list']=[['year built'],['year'],['time'],['build'],['built'],['history']]
  keyword_df.loc[15,'synonym_list']=[['price'],['per']]
  keyword_df.loc[16,'synonym_list']=[['hoa'],['homeowner'],['association'],['fee'],['administration']]
  keyword_df.loc[20,'synonym_list']=[['information'],['more'],['detail'],['url']]
  keyword_df=keyword_df.loc[keyword_df['synonym_list']!=0,].reset_index(drop=True)
  df['year built']=df['year built'].fillna(0)
  df['year built']=df['year built'].astype('int')
  
  import spacy
  import fastDamerauLevenshtein
  from pyjarowinkler import distance
  nlp = spacy.load('en_core_web_md')
  from nltk.stem.snowball import SnowballStemmer
  STEMMER = SnowballStemmer("english")

  def input_seman(input_question):
    nounadjv_list=[]
    for token in nlp(input_question):
      if (token.pos_=='NOUN')|(token.pos_=='ADJ')|(token.pos_=='VERB'):
        nounadjv_list.append(token)
    return nounadjv_list

  def keyword_sim(nounadjv_list,keyword_df,thresh_score=0.9):
    if len(nounadjv_list)>0:
      score_df=pd.DataFrame(columns=['input_noun','key','dl_score','jw_score'])
      for i in nounadjv_list:
        for row in range(len(keyword_df)):
          keyword_list=keyword_df.loc[row,'synonym_list']
          for j in keyword_list:
            dl_score=fastDamerauLevenshtein.damerauLevenshtein(str(i),j,similarity=True)
            jw_score=distance.get_jaro_distance(str(i),j,winkler=True,scaling=0.1)
            score_df=score_df.append({'input_noun':i,'key':keyword_df.loc[row,'key_feature'],'dl_score':dl_score,'jw_score':jw_score},ignore_index=True)
      return list(set(score_df.loc[(score_df['jw_score']>=thresh_score)|(score_df['dl_score']>=thresh_score),'key'].tolist()))
  list_answer=[]
  house_info=df.iloc[house_number,]
  nounadjv_list=input_seman(input_question)
  returnlist=keyword_sim(nounadjv_list,keyword_df,thresh_score=0.9)
  list_answer.append('Here are what we have for now:')
  if (returnlist is not None)&(returnlist!=[]):
    list_answer.append(str(house_info[returnlist]))
  else:
    list_answer.append("Sorry, we didn't find the answer for your question.")
  
  variant_key=[]
  for c in returnlist:
    if c in keyword_df['key_feature']:
      variant_key.extend(keyword_df.loc[keyword_df['key_feature']==c,'synonym_list'].tolist()[0])
    else:
      variant_key.append(c)
  for d in nounadjv_list:
    variant_key.append(str(d))
  variant_key

  input_list=[STEMMER.stem(w) for w in variant_key]
  input_list=list(set(input_list))
  list_answer.append('Here is what agent said:')
  para_hi=dict()
  if house_info['remarks']!='no remarks':
    doc=nlp(house_info['remarks'])
    

    for sents in doc.sents:
      tokenlist=list(sents)
      token_stem=[]
      for token in tokenlist:
        if (token.pos_=='NOUN')|(token.pos_=='ADJ')|(token.pos_=='VERB'):
          token_stem.append(STEMMER.stem(str(token)))
      if set(input_list)&set(token_stem)!=set():
        para_hi[str(sents)]=1
      else:
        para_hi[str(sents)]=0    
  else:
    para_hi['Oops, the agent said nothing about this property.']=0
  list_answer.append(para_hi)
    
  



  import torch
  from transformers import AutoTokenizer, AutoModel
  question_database=pd.read_csv('static/questionbase.csv')
  sentences=[input_question]
  # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
  # model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
  tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
  model = AutoModel.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")


  tokens = {'input_ids': [], 'attention_mask': []}

  for sentence in sentences:
      # encode each sentence and append to dictionary
      new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                        truncation=True, padding='max_length',
                                        return_tensors='pt')
      tokens['input_ids'].append(new_tokens['input_ids'][0])
      tokens['attention_mask'].append(new_tokens['attention_mask'][0])

  # reformat list of tensors into single tensor
  tokens['input_ids'] = torch.stack(tokens['input_ids'])
  tokens['attention_mask'] = torch.stack(tokens['attention_mask'])


  outputs = model(**tokens)


  embeddings = outputs.last_hidden_state


  attention_mask = tokens['attention_mask']
  mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()


  masked_embeddings = embeddings * mask



  summed = torch.sum(masked_embeddings, 1)



  summed_mask = torch.clamp(mask.sum(1), min=1e-9)



  mean_pooled = summed / summed_mask


  from sklearn.metrics.pairwise import cosine_similarity
  # convert from PyTorch tensor to numpy array
  mean_pooled = mean_pooled.detach().numpy()

  base_ques_encode=np.load("static/base_que_cos.npy")
  # calculate
  sim=cosine_similarity(
      [mean_pooled[0]],base_ques_encode
  )

  bert_sim_df=pd.concat([question_database,pd.DataFrame(sim).transpose()],axis=1)
  keyword_out=bert_sim_df.loc[bert_sim_df[0]>0.9,'Answer'].unique().tolist()
  list_answer.append('Based BERT similarity analysis:')
  if (keyword_out is not None)&(keyword_out!=[]):
    list_answer.append(str(house_info[keyword_out]))
  else:
    list_answer.append("Sorry, we didn't find the answer for your question.")
  
  return list_answer

