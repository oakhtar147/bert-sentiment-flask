import config
import torch
import flask

from model import BERTBaseUncased
from flask import request
from flask import Flask

app = Flask(__name__)
MODEL = None
DEVICE = "cpu"

@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    positive_pred = sentence_prediction(sentence, model=MODEL)
    negative_pred = 1 - positive_pred

    response = {}
    response["response"] = {
        "sentence": sentence,
        "positive": str(positive_pred),
        "negative": str(negative_pred)    
    }

    return flask.jsonify(response)

def sentence_prediction(sentence, model=MODEL):
    review = str(sentence)
    review = " ".join(review.split())      
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len
    )

    ids = inputs["input_ids"] # word2idx
    mask = inputs["attention_mask"] # padded tokens that attention mechanism does not attend to
    token_type_ids = inputs["token_type_ids"] # differentiate between to sequences, 1 since using one sentence in tokenizer

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.int64)
    mask = torch.tensor(mask, dtype=torch.int64)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64)

    ids = ids.to(DEVICE, dtype=torch.long).unsqueeze(0) # add 1 to batch_dim
    mask = mask.to(DEVICE, dtype=torch.long).unsqueeze(0)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long).unsqueeze(0)

    output = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
    print(type(output))
    output = torch.sigmoid(output).cpu().detach().numpy().squeeze()

    return output


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    MODEL.eval()
    app.run(debug=True)