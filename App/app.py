import streamlit as st
import pandas as pd

# Model
# from simpletransformers.t5 import T5Model, T5Args
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Data Processing
from sklearn.model_selection import train_test_split



class Data:
	def __init__(self, path):
		self.data = pd.read_csv(
			path,
			encoding='utf-8'
		)


	def reformat_data(self, mode: str):
		if mode in self.data.columns:
			return 'column existed'

		mode = pd.DataFrame({
			'prefix':[mode]*self.data.shape[0]
		})

		self.data = pd.concat(
			mode,
			self.data,
			ignore_index=True
		)

		self.data.reset_index(inplace=True)

		self.data = self.data[['prefix','input_text','target_text']]


	def preprocessing_null_duplication(self, subset: List):
		# Null Values
		self.data.dropna(
			subset = subset,
			inplace = True
		)

		# Drop Duplication
		duplication = self.data.duplicated(subset=subset).sum()

		if duplication:
			self.data.drop_duplicates(
				subset=subset,
				inplace=True,
				ignore_index=True,
			)






class Model:
	def __init__(self, path):
		self.tokenizer = T5Tokenizer.from_pretrained(path)
		self.model = T5ForConditionalGeneration.from_pretrained(path)

	def train(self):
		return "Comming soon!"

	def predict(self, original_text):
		inputs = self.tokenizer(
			f'informal converter: {original_text}',
			return_tensors="pt"
		).input_ids

		output = self.model.generate(inputs)

		return tokenizer.decode(
			output[0],
			skip_special_tokens=True
		)



if __name__ == '__main__':

	txt = st.text_area('Input informal text for conversion')
	st.write('Converter:', Model(path).predict(txt))