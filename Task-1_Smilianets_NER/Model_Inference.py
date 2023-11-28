from transformers import DistilBertForTokenClassification, DistilBertTokenizer
import torch


def tokenize_and_predict(text, model, tokenizer, label2id):
    # Tokenize input text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    input_ids = tokenizer.encode(text, return_tensors='pt').squeeze().tolist()

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([input_ids]))

    # Convert predicted label indices to labels
    predicted_labels = [list(label2id.keys())[i] for i in torch.argmax(outputs.logits, dim=2).squeeze().tolist()]

    # Extract entities from the predicted labels
    entities = []
    current_entity = {"text": "", "start": None, "end": None, "label": None}
    for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
        if label == 'B':
            if current_entity["text"]:
                entities.append(current_entity)
            current_entity = {"text": token, "start": i, "end": i + 1, "label": label}
        elif label == 'I':
            current_entity["text"] += " " + token
            current_entity["end"] = i + 1

    if current_entity["text"]:
        entities.append(current_entity)

    return entities


# Load the trained model and tokenizer
model = DistilBertForTokenClassification.from_pretrained('./ner_model')
tokenizer = DistilBertTokenizer.from_pretrained('./ner_tokenizer')
label2id = {'O': 0, 'B': 1, 'I': 2}

# Take input from the keyboard
input_text = input("Enter the text for named entity recognition: ")

# Tokenize and predict entities
predicted_entities = tokenize_and_predict(input_text, model, tokenizer, label2id)

# Print the predicted entities
print("Predicted Entities:")
for entity in predicted_entities:
    print(f"Text: {entity['text']}, Label: {entity['label']}, Start: {entity['start']}, End: {entity['end']}")
