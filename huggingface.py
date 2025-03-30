import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer
from transformers import TFBertModel

class NLPModel:
	def create_bert_classification_model(self,
										 bert_model,
										 num_train_layers=0,
										 max_sequence_length= 400,
										 num_filters = [100, 100, 50, 25],
										 kernel_sizes = [3, 4, 5, 10],
										 hidden_size = 200,
										 hidden2_size = 100,
										 dropout = 0.1,
										 learning_rate = 0.001,
										 label_smoothing = 0.03
										):
		"""
		Build a simple classification model with BERT. Use the Pooler Output or CLS for classification purposes
		"""
		if num_train_layers == 0:
			# Freeze all layers of pre-trained BERT model
			bert_model.trainable = False

		elif num_train_layers == 12:
			# Train all layers of the BERT model
			bert_model.trainable = True

		else:
			# Restrict training to the num_train_layers outer transformer layers
			retrain_layers = []

			for retrain_layer_number in range(num_train_layers):

				layer_code = '_' + str(11 - retrain_layer_number)
				retrain_layers.append(layer_code)


			#print('retrain layers: ', retrain_layers)

			for w in bert_model.weights:
				if not any([x in w.name for x in retrain_layers]):
					#print('freezing: ', w)
					w._trainable = False

		input_ids = tf.keras.layers.Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='input_ids')
		token_type_ids = tf.keras.layers.Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='token_type_ids')
		attention_mask = tf.keras.layers.Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='attention_mask')
										
		bert_inputs = {'input_ids': input_ids,
					   'token_type_ids': token_type_ids,
					   'attention_mask': attention_mask}

		bert_out = bert_model(bert_inputs)

		pooler_token = bert_out[1]
		cls_token = bert_out[0][:, 0, :]
		bert_out_avg = tf.math.reduce_mean(bert_out[0], axis=1)
		cnn_token = bert_out[0]

		conv_layers_for_all_kernel_sizes = []
		for kernel_size, filters in zip(kernel_sizes, num_filters):
			conv_layer = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(cnn_token)
			conv_layer = tf.keras.layers.GlobalMaxPooling1D()(conv_layer)
			conv_layers_for_all_kernel_sizes.append(conv_layer)

		conv_output = tf.keras.layers.concatenate(conv_layers_for_all_kernel_sizes, axis=1)

		# classification layer
		hidden = tf.keras.layers.Dense(hidden_size, activation='relu', name='hidden_layer')(conv_output)
		hidden = tf.keras.layers.Dropout(dropout)(hidden)

		hidden = tf.keras.layers.Dense(hidden2_size, activation='relu', name='hidden_layer2')(hidden)
		hidden = tf.keras.layers.Dropout(dropout)(hidden)
		
		classification = tf.keras.layers.Dense(1, activation='sigmoid',name='classification_layer')(hidden)

		classification_model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[classification])

		classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
									 # LOSS FUNCTION
									 loss=tf.keras.losses.BinaryFocalCrossentropy(
									   gamma=2.0, from_logits=False, apply_class_balancing=True, label_smoothing=label_smoothing
									 ),
									 # METRIC FUNCTIONS
									 metrics=['accuracy']
									 )
		return classification_model

	def __init__(self):
		self.MAX_SEQUENCE_LENGTH = 400
		self.model_checkpoint = "bert-base-uncased"
		# Step 1: Load BERT Tokenizer
		self.tokenizer = BertTokenizer.from_pretrained(self.model_checkpoint)
		# Step 2: Load Pretrained BERT model
		self.bert_model = TFBertModel.from_pretrained(self.model_checkpoint)
		# Stage 3: Create custom BERT model on top of the pretrained model
		self.inference_model = self.create_bert_classification_model(bert_model=self.bert_model)
		# Stage 4: Load Inference model with saved weights
		self.save_path = 'bert_cnn_ensemble_resample_uncased_mdl.h5'
		self.inference_model.load_weights(self.save_path)


	def run_inference_model(self,conversations):
		# Tokenize conversations with BERT tokenizer
		self.tokenized_input = self.tokenizer(conversations,
									max_length=self.MAX_SEQUENCE_LENGTH,
									truncation=True,
									padding='max_length',
									return_tensors='tf')
		self.bert_inputs = [self.tokenized_input.input_ids,
					   self.tokenized_input.token_type_ids,
					   self.tokenized_input.attention_mask]

		# Apply Model Prediction to testData
		self.y_pred = self.inference_model.predict(self.bert_inputs)
		#prediction = f_one_or_zero(y_pred)
		return self.y_pred