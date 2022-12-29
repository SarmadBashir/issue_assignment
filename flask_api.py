from transformers import  AutoTokenizer, AutoModelForSequenceClassification, pipeline
#import torch.nn.functional as F
from flask import request, Flask
import numpy as np
#import torch
import json
import re

app = Flask(__name__)

def get_pipleine():
    tokenizer = AutoTokenizer.from_pretrained("SarmadBashir/Issue_Assignment", model_max_length = 128)
    model = AutoModelForSequenceClassification.from_pretrained("SarmadBashir/Issue_Assignment")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe

pipe = get_pipleine()

#path = '../issue_assigner/model'
#tokenizer = BertTokenizerFast.from_pretrained(path, lower_case=True)
#model = BertForSequenceClassification.from_pretrained(path)

def preprocess(text):
    line = text.lower()
    line = re.sub(r"http\S+", "", line)
    line = re.sub("[^A-Za-z]+", " ", line)
    line = re.sub('\s+', ' ', line)
    line = line.replace('\t',' ')
    line = line.replace('\n',' ')
    line = line.replace('\r',' ')
    line = line.replace(',','')
    line = line.replace('-',' ')
    line = ' '.join(line.split())

    return line

#def get_prediction(input_text, tokenizer, model):
def get_prediction(input_text, pipe):
    map_labels = {
                 'LABEL_0': 'bug',
                 'LABEL_1': 'enhancement',
                 'LABEL_2': 'internal improvement',
                 'LABEL_3': 'hilla',
                 'LABEL_4': 'a11y',
                 'LABEL_5': 'documentation'
             }
    
    output = pipe(input_text)
    
    label = map_labels.get(output[0]['label'])
    score = round(output[0]['score'], 2)

    #MAX_LENGTH =  128
    #test_encodings = tokenizer(input_text, truncation=True, 
    #                           padding=True, 
    #                           max_length=MAX_LENGTH, 
    #                           return_tensors="pt")

    #with torch.no_grad():
    #    logits = model(**test_encodings).logits
    
    #predictions = np.argmax(logits, axis=1)
    #top_label_int =  predictions.tolist()[0]
    #top_label = map_labels.get(top_label_int)

    #probabilities = F.softmax(logits, dim=1)
    #top_probability = probabilities[0].tolist()[top_label_int]
    #top_probability = round(top_probability, 2)

    return label, score

@app.route("/assigner", methods=['GET'])
def run():
    
    #issue_text = request.args.get('issue_text', '')
    data = request.get_json()
    issue_text = data['text']
    print(issue_text)
    if issue_text == '':
        return app.response_class(response='Provide valid input text',
                                  status=400,
                                  mimetype='application/json'
                                  )
        
    input_text =  preprocess(issue_text)    
    top_label, top_probability = get_prediction(input_text, 
                                                pipe) 
    #                                            tokenizer, 
    #                                            model)
    
    result = {'prediction': {'label': top_label, 
                             'probability': top_probability
                             }
              }

    response = app.response_class(response=json.dumps(result),
                                  status=200,
                                  mimetype='application/json'
                                  )
    return response

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)