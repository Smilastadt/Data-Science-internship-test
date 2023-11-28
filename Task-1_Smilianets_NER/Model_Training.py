from transformers import DistilBertForTokenClassification
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Extracting our data
sentence_df = pd.read_csv('Sentences_Data.csv')


# A function to preprocess the created DataFrame,
# tokenize the text and convert the labels into the BIO format
def preprocess_data(df, tokenizer):

    tokenized_texts = []
    tokenized_labels = []

    for _, row in df.iterrows():
        sentence = row['text']
        label = row['label']

        # Tokenize the sentence
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
        input_ids = tokenizer.encode(sentence, return_tensors='pt').squeeze().tolist()

        # Initialize labels with 'O' for each token
        labels = ['O'] * len(input_ids)

        # Tokenize the label and find the indices where it matches the sentence tokens
        label_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(label)))

        # Find the starting index of the label in the sentence
        start_index = tokens.index(label_tokens[0]) if label_tokens[0] in tokens else -1

        if start_index != -1:
            # Mark the starting index with 'B' (Beginning)
            labels[start_index] = 'B'

            # Mark the subsequent indices with 'I' (Inside)
            for i in range(1, len(label_tokens)):
                if start_index + i < len(labels):
                    labels[start_index + i] = 'I'

        # Append tokenized sentence and labels
        tokenized_texts.append(input_ids)
        tokenized_labels.append(labels)

    return tokenized_texts, tokenized_labels


# Converting tokenized text and labels to model input
def convert_to_model_input(tokenizer, tokenized_texts, tokenized_labels, max_length=128, label2id=None):
    input_ids = []
    attention_masks = []
    new_labels = []

    for tokens, labels in zip(tokenized_texts, tokenized_labels):
        # Truncate or pad the tokens to the specified max length
        tokens = tokens[:max_length - 2]
        labels = labels[:max_length - 2]

        # Add [CLS] and [SEP] tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels = ['O'] + labels + ['O']

        # Convert tokens and labels to input IDs
        input_id = tokenizer.convert_tokens_to_ids(tokens)

        # Use label2id dictionary, or default to 0 for 'O' if not found
        label_id = [label2id.get(label, label2id['O']) for label in labels]

        # Generate attention mask
        attn_mask = [1] * len(input_id)

        # Pad up to the maximum length
        padding_length = max_length - len(input_id)
        input_id = input_id + [0] * padding_length
        attn_mask = attn_mask + [0] * padding_length
        label_id = label_id + [label2id['O']] * padding_length

        input_ids.append(input_id)
        attention_masks.append(attn_mask)
        new_labels.append(label_id)

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(new_labels)


# defining labels mapping
label2id = {'O': 0, 'B': 1, 'I': 2}

train_df, valid_df = train_test_split(sentence_df, test_size=0.1, random_state=42)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_tokens(sentence_df['label'].unique().tolist(), special_tokens=True)


# preprocessing data for training
tokenized_texts, tokenized_labels = preprocess_data(train_df, tokenizer)

# creating a dataloader for training


input_ids, attention_masks, labels = convert_to_model_input(tokenizer, tokenized_texts, tokenized_labels,
                                                            label2id=label2id)
train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Preprocess validation data
tokenized_texts_val, tokenized_labels_val = preprocess_data(valid_df, tokenizer)
input_ids_val, attention_masks_val, labels_val = convert_to_model_input(tokenizer, tokenized_texts_val,
                                                                        tokenized_labels_val, label2id=label2id)
valid_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Training the NER model
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label2id))
model.resize_token_embeddings(len(tokenizer))

optimizer = AdamW(model.parameters(), lr=3e-5)

# Set up a learning rate scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # You can adjust step_size and gamma as needed

for epoch in range(20):
    model.train()
    train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                     desc=f"Epoch {epoch + 1}")
    for batch_num, batch in train_dataloader_iterator:
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update the progress bar description with the current loss
        train_dataloader_iterator.set_postfix({'Loss': loss.item()}, refresh=True)

    # Step the learning rate scheduler
    scheduler.step()

    # Access the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")

    # Model Validation
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for val_batch_num, val_batch in enumerate(valid_dataloader):
            val_inputs = {'input_ids': val_batch[0], 'attention_mask': val_batch[1], 'labels': val_batch[2]}
            val_outputs = model(**val_inputs)
            val_loss += val_outputs.loss.item()
            num_batches += 1

    avg_val_loss = val_loss / num_batches
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

model.save_pretrained('./ner_model')
tokenizer.save_pretrained('./ner_tokenizer')
