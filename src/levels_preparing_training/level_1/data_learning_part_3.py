import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import BertForSequenceClassification, BertTokenizer, BertTokenizerFast
from transformers import TrainingArguments, Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Загрузка данных
train_data = pd.read_csv("/datasets/datasets_small/train_refactored_small.csv", dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(0)
test_data = pd.read_csv("/datasets/datasets_small/test_refactored_small.csv", dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(1)
val_data = pd.read_csv("/datasets/datasets_small/train_refactored_small_validation.csv", dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(2)


labels = train_data['RGNTI1'].unique().tolist()
labels = [s.strip() for s in labels]
for key, value in enumerate(labels):
    print(value)

NUM_LABELS = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

print(3)
train_data["labels"] = train_data['RGNTI1'].map(lambda x: label2id[x.strip()])
val_data["labels"] = val_data['RGNTI1'].map(lambda x: label2id[x.strip()])
test_data["labels"] = test_data['RGNTI1'].map(lambda x: label2id[x.strip()])

print(4)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
max_len = 0
for sent in train_data['body']:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
print(5)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', max_length=max_len)
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                      num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
model.to(device)

print(6)
train_texts = list(train_data['body'])
val_texts = list(val_data['body'])
test_texts = list(test_data['body'])
train_labels = list(train_data["labels"])
val_labels = list(val_data["labels"])
test_labels = list(test_data["labels"])

print(7)


class DataLoader(Dataset):
    """
    Custom Dataset class for handling tokenized text data and corresponding labels.
    Inherits from torch.utils.data.Dataset.
    """

    def __init__(self, encodings, labels):
        """
        Initializes the DataLoader class with encodings and labels.

        Args:
            encodings (dict): A dictionary containing tokenized input text data
                              (e.g., 'input_ids', 'token_type_ids', 'attention_mask').
            labels (list): A list of integer labels for the input text data.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tokenized data and the corresponding label for a given index.

        Args:
            idx (int): The index of the data item to retrieve.

        Returns:
            item (dict): A dictionary containing the tokenized data and the corresponding label.
        """
        # Retrieve tokenized data for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the given index to the item dictionary
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the number of data items in the dataset.

        Returns:
            (int): The number of data items in the dataset.
        """
        return len(self.labels)


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

print(8)
train_dataloader = DataLoader(train_encodings, train_labels)
val_dataloader = DataLoader(val_encodings, val_labels)
test_dataset = DataLoader(test_encodings, test_labels)
print(9)


training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./usual_bert',
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    # Number of steps used for a linear warmup
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
    # TensorBoard log directory
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    load_best_model_at_end=True
)

def compute_metrics(pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions.

    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of
              that observation belonging to a certain class.

    Returns:
        dict: A dictionary containing the following metrics:
            - Accuracy (float): The proportion of correctly classified instances.
            - F1 (float): The macro F1 score, which is the harmonic mean of precision
              and recall. Macro averaging calculates the metric independently for
              each class and then takes the average.
            - Precision (float): The macro precision, which is the number of true
              positives divided by the sum of true positives and false positives.
            - Recall (float): The macro recall, which is the number of true positives
              divided by the sum of true positives and false negatives.
    """
    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


print(10)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics=compute_metrics
)
print(11)
trainer.train()
print(12)


def predict(text):
    """
    Predicts the class label for a given input text

    Args:
        text (str): The input text for which the class label needs to be predicted.

    Returns:
        probs (torch.Tensor): Class probabilities for the input text.
        pred_label_idx (torch.Tensor): The index of the predicted class label.
        pred_label (str): The predicted class label.
    """
    # Tokenize the input text and move tensors to the GPU if available
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")

    # Get model output (logits)
    outputs = model(**inputs)

    probs = outputs[0].softmax(1)
    """ Explanation outputs: The BERT model returns a tuple containing the output logits (and possibly other elements depending on the model configuration). In this case, the output logits are the first element in the tuple, which is why we access it using outputs[0].

    outputs[0]: This is a tensor containing the raw output logits for each class. The shape of the tensor is (batch_size, num_classes) where batch_size is the number of input samples (in this case, 1, as we are predicting for a single input text) and num_classes is the number of target classes.

    softmax(1): The softmax function is applied along dimension 1 (the class dimension) to convert the raw logits into class probabilities. Softmax normalizes the logits so that they sum to 1, making them interpretable as probabilities. """

    # Get the index of the class with the highest probability
    # argmax() finds the index of the maximum value in the tensor along a specified dimension.
    # By default, if no dimension is specified, it returns the index of the maximum value in the flattened tensor.
    pred_label_idx = probs.argmax()

    # Now map the predicted class index to the actual class label
    # Since pred_label_idx is a tensor containing a single value (the predicted class index),
    # the .item() method is used to extract the value as a scalar
    pred_label = model.config.id2label[pred_label_idx.item()]

    return probs, pred_label_idx, pred_label


text = "Makine öğrenimi kendisi de daha da otomatik hale doğru ilerliyor."
predict(text)
print(predict(text))
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# max_len = 0
#
# # For every sentence...
# for sent in train_data_text:
#     input_ids = tokenizer.encode(sent, add_special_tokens=True)
#     max_len = max(max_len, len(input_ids))
#
# print('Max sentence length: ', max_len)
#
# input_ids = []
# attention_masks = []
#
# for sent in train_data_text:
#     # `encode_plus` will:
#     #   (1) Tokenize the sentence.
#     #   (2) Prepend the `[CLS]` token to the start- for classification tasks
#     #   (3) Append the `[SEP]` token to the end.
#     #   (4) Map tokens to their IDs.
#     #   (5) Pad or truncate the sentence to `max_length`
#     #   (6) Create attention masks for [PAD] tokens
#     #   (7) The attention mask is a binary tensor indicating the position of the padded indices so that the model does not attend to them.
#     ##      For the BertTokenizer, 1 indicates a value that should be attended to, while 0 indicates a padded value.
#     encoded_dict = tokenizer.encode_plus(
#         sent,  # Sentence to encode.
#         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#         max_length=max_len,  # Pad & truncate all sentences.
#         pad_to_max_length=True,
#         return_attention_mask=True,  # Construct attn. masks.
#         return_tensors='pt',  # Return pytorch tensors.
#     )
#
#     input_ids.append(encoded_dict['input_ids'])
#
#     attention_masks.append(encoded_dict['attention_mask'])
#
# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
# labels = torch.tensor(train_data_labels)
#
# print('\nOriginal: ', train_data_text[0])
# print('\nToken IDs:', input_ids[0])
# print('\nAttension Mask:', attention_masks[0])
#
# dataset = TensorDataset(input_ids, attention_masks, labels)
#
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
#
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# print('{:>5,} training samples'.format(train_size))
# print('{:>5,} validation samples'.format(val_size))
#
# batch_size = 32
#
# train_dataloader = DataLoader(
#     train_dataset,  # The training samples.
#     sampler=RandomSampler(train_dataset),  # Select batches randomly
#     batch_size=batch_size  # Trains with this batch size.
# )
# print(2)
# validation_dataloader = DataLoader(
#     val_dataset,  # The validation samples.
#     sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
#     batch_size=batch_size  # Evaluate with this batch size.
# )
# print(3)
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
#     num_labels=2,  # The number of output labels--2 for binary classification.
#     # You can increase this for multi-class tasks.
#     output_attentions=False,  # Whether the model returns attentions weights.
#     output_hidden_states=False,  # Whether the model returns all hidden-states.
# )
# print(4)
# if device.type == 'cuda':
#     # Tell pytorch to run this model on the GPU.
#     model = model.cuda()
#     print("\n\nGPU Enabled.")
# else:
#     print("\n\nCPU On.")
# model = model.to(device)
#
# optimizer = AdamW(model.parameters(),
#                   lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps=1e-8  # args.adam_epsilon  - default is 1e-8.
#                   )
#
# epochs = 4
#
# # Total number of training steps is [number of batches] x [number of epochs].
# # (Note that this is not the same as the number of training samples).
# total_steps = len(train_dataloader) * epochs
# print(5)
# # Create a schedule with a learning rate that decreases linearly from the initial lr
# # set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=0,  # Default value in run_glue.py
#                                             num_training_steps=total_steps)
#
#
# def flat_accuracy(preds, labels):
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return np.sum(pred_flat == labels_flat) / len(labels_flat)
#
#
# print(6)
#
#
# def format_time(elapsed):
#     '''
#     Takes a time in seconds and returns a string hh:mm:ss
#     '''
#     # Round to the nearest second.
#     elapsed_rounded = int(round((elapsed)))
#     # Format as hh:mm:ss
#     return str(datetime.timedelta(seconds=elapsed_rounded))
#
#
# print(7)
#
# seed_val = 42
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)
# training_stats = []
#
# # Measure the total training time for the whole run.
# total_t0 = time.time()
# for epoch_i in range(0, epochs):
#     print("")
#     print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
#     print('Training...')
#     # Measure how long the training epoch takes.
#     t0 = time.time()
#     total_train_loss = 0
#     model.train()
#     for step, batch in enumerate(train_dataloader):
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)
#         optimizer.zero_grad()
#         output = model(b_input_ids,
#                        token_type_ids=None,
#                        attention_mask=b_input_mask,
#                        labels=b_labels)
#         loss = output.loss
#         total_train_loss += loss.item()
#         # Perform a backward pass to calculate the gradients.
#         loss.backward()
#         # Clip the norm of the gradients to 1.0.
#         # This is to help prevent the "exploding gradients" problem.
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         # Update parameters and take a step using the computed gradient.
#         # The optimizer dictates the "update rule"--how the parameters are
#         # modified based on their gradients, the learning rate, etc.
#         optimizer.step()
#         # Update the learning rate.
#         scheduler.step()
#
#         # Calculate the average loss over all of the batches.
#     avg_train_loss = total_train_loss / len(train_dataloader)
#
#     # Measure how long this epoch took.
#     training_time = format_time(time.time() - t0)
#     print("")
#     print("  Average training loss: {0:.2f}".format(avg_train_loss))
#     print("  Training epcoh took: {:}".format(training_time))
#     print("")
#     print("Running Validation...")
#     t0 = time.time()
#     # Put the model in evaluation mode--the dropout layers behave differently
#     # during evaluation.
#     model.eval()
#     # Tracking variables
#     total_eval_accuracy = 0
#     best_eval_accuracy = np.Inf
#     total_eval_loss = 0
#     nb_eval_steps = 0
#     # Evaluate data for one epoch
#     for batch in validation_dataloader:
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)
#         # Tell pytorch not to bother with constructing the compute graph during
#         # the forward pass, since this is only needed for backprop (training).
#         with torch.no_grad():
#             output = model(b_input_ids,
#                            token_type_ids=None,
#                            attention_mask=b_input_mask,
#                            labels=b_labels)
#         loss = output.loss
#         total_eval_loss += loss.item()
#         # Move logits and labels to CPU if we are using GPU
#         logits = output.logits
#         logits = logits.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
#         # Calculate the accuracy for this batch of test sentences, and
#         # accumulate it over all batches.
#         total_eval_accuracy += flat_accuracy(logits, label_ids)
#     # Report the final accuracy for this validation run.
#     avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
#     print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
#     # Calculate the average loss over all of the batches.
#     avg_val_loss = total_eval_loss / len(validation_dataloader)
#     # Measure how long the validation run took.
#     validation_time = format_time(time.time() - t0)
#     if avg_val_accuracy < best_eval_accuracy:
#         torch.save(model, 'bert_model')
#         best_eval_accuracy = avg_val_accuracy
#         # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
#         # print("  Validation took: {:}".format(validation_time))
#         # Record all statistics from this epoch.
#     training_stats.append(
#         {
#             'epoch': epoch_i + 1,
#             'Training Loss': avg_train_loss,
#             'Valid. Loss': avg_val_loss,
#             'Valid. Accur.': avg_val_accuracy,
#             'Training Time': training_time,
#             'Validation Time': validation_time
#         }
#     )
# print("\n\n")
# print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
# print(f"\n\nTraining completed and following are the stats {training_stats}!")
