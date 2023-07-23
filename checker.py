from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

with open("questions.txt", "r") as f:
    questions = f.readlines()

with open("answers.txt", "r") as f:
    answers = f.readlines()

def findSimilarity(s1, s2):
    sentences = [s1, s2]
    # initialize dictionary to store tokenized sentences
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
    # outputs.keys()
    embeddings = outputs.last_hidden_state
    # embeddings.shape
    attention_mask = tokens['attention_mask']
    # attention_mask.shape
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    # mask.shape
    masked_embeddings = embeddings * mask
    # masked_embeddings.shape
    summed = torch.sum(masked_embeddings, 1)
    # summed.shapeThe symptoms of diabetics are Urinate (pee) a lot, often at night, Are very thirsty, Lose weight without trying, Are very hungry, Have blurry vision, Have numb or tingling hands or feet, Feel very tired, Have very dry skin.
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    # summed_mask.shape
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().numpy()

    # calculate
    prediction = cosine_similarity(
        [mean_pooled[0]],
        mean_pooled[1:]
    )
    return prediction[0][0] * 100


def checkPlagarism(s1, s2, threshold = 60):
    similarity = findSimilarity(s1, s2)
    if similarity > threshold:
        return True, similarity
    return False, similarity


# print(checkPlagarism("What are the symptoms of diabetics", "i like pancakes"))

def check(question):
    relation = []
    for i in range(len(questions)):
        data, value = checkPlagarism(question, questions[i].strip())
        if data:
            relation.append(value)
        else:
            relation.append(0)

    if sum(relation) > 0:
        ind = relation.index(max(relation))
        return answers[ind].strip()


    
    with open("unknownQuestions.txt", "a") as f:
        f.write(question + "\n")
    
    return "Sorry, I don't know the answer to that question"