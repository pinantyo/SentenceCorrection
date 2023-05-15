import streamlit as st
import pandas as pd
from decouple import config
import os, json

# Model
# from simpletransformers.t5 import T5Model, T5Args
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Data Processing
from sklearn.model_selection import train_test_split


# Data Management
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


	def preprocessing_null_duplication(self, subset: list):
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


# Models
class Model:
	def __init__(self, path):
		self.tokenizer = T5Tokenizer.from_pretrained(path)
		self.model = T5ForConditionalGeneration.from_pretrained(path)

	def train(self, args: dict):

		return "Comming soon!"

	def predict(self, original_text):
		inputs = self.tokenizer(
			f'informal converter: {original_text}',
			return_tensors="pt"
		).input_ids

		output = self.model.generate(inputs)

		return self.tokenizer.decode(
			output[0],
			skip_special_tokens=True
		)



if __name__ == '__main__':
	# Init global variable
	model = None
	file = None

	st.header("Frasakan")
	MODEL_PATH = config('MODEL_PATH')



	tab_master1, tab_master2 = st.tabs(["Modelling", "Data Processing"])


	# Sidebar
	with st.sidebar:
		file = st.file_uploader("Upload CSV file required")
		
	with st.container():

		with tab_master1:
			st.header("Modelling")
			option = os.path.join(MODEL_PATH, st.selectbox(
				'Choose machine learning model',
				set(
					os.listdir(MODEL_PATH)
				)
			))

			if st.button('Init Model'):
				# Init model
				model = Model(option)
				st.write(
					f"Model {option} loaded."
				)

			
			


			tabs1, tabs2 = st.tabs(["Train/Validation", "Test"])

			

			with tabs1:
				# Select Models
				st.header("Training - Comming Soon")

				col1, col2 = st.columns(2)

				with col1:
					json_file = st.file_uploader('Import model args json')

					if json_file:
						with open(json_file, 'r') as f:
							json_args = json.load(f)

						st.write("Model Arguements: ", str(json_args))
			

			with tabs2:
				st.header("Testing")
				txt = st.text_area('Input informal text for conversion')

				if st.button('Convert'):
					st.write(
						'Converter:', 
						model.predict(txt)
					)
		

		with tab_master2:
			st.header("Data Processing")

			# Display file's content
			if file:
				df = pd.read_csv(
					file,
					encoding='utf-8'
				)

				st.write(
					df
				)

			