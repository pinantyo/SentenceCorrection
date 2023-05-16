import streamlit as st
import pandas as pd
from decouple import config
import os, json

# Model
# from simpletransformers.t5 import T5Model, T5Args
from transformers import T5Tokenizer, T5ForConditionalGeneration
from simpletransformers.t5 import T5Model


# Data Processing
from sklearn.model_selection import train_test_split


# Data Management
class Data:
	# @st.cache_data
	def __init__(self, path, subset: list):
		self.__data = pd.read_csv(
			path,
			encoding='utf-8',
		)[subset]


	def reformat_data(self, mode: str):
		if mode in self.__data.columns:
			return 'column existed'

		mode = pd.DataFrame({
			'prefix':[mode]*self.__data.shape[0]
		})

		temp = pd.concat(
			[mode,
			self.__data],
			ignore_index=True
		)

		self.__data = temp
		self.__data.reset_index(inplace=True, drop=True)
		self.__data.columns = ['prefix','input_text','target_text']


	def preprocessing_null_duplication(self, subset: list):
		# Null Values
		self.__data.dropna(
			inplace = True
		)

		# Drop Duplication
		duplication = self.__data.duplicated(subset=subset).sum()

		if duplication:
			self.__data.drop_duplicates(
				subset=subset,
				inplace=True,
				ignore_index=True,
			)

	def get_data(self):
		return self.__data
	
	def get_shape(self):
		return self.__data.shape
	
	def get_columns(self):
		return list(self.__data.columns)


# Models
class Model:
	# @st.cache_data
	def __init__(self, path: str, train: bool=False, args: dict={}, cuda: bool = False):
		if train:
			self.__model = T5Model(
				"t5",
				path,   
				args=args,
				use_cuda=cuda
			)
		
		else:
			self.__tokenizer = T5Tokenizer.from_pretrained(path)
			self.__model = T5ForConditionalGeneration.from_pretrained(path)

	def train(self, train_data, eval_data):
		st.write(self.__model.train_model(
			train_data.applymap(str), 
			eval_data=eval_data.applymap(str)
		))

		# Model - Evaluation
		return self.__model.eval_model(eval_data)

	def predict(self, original_text):
		inputs = self.__tokenizer(
			f'informal converter: {original_text}',
			return_tensors="pt"
		).input_ids

		output = self.__model.generate(inputs)

		return self.__tokenizer.decode(
			output[0],
			skip_special_tokens=True
		)



if __name__ == '__main__':
	# Session state
	if "load_state" not in st.session_state:
		st.session_state.load_state = False
	
	if "model_state" not in st.session_state:
		st.session_state.model_state = False


	st.header("Frasakan")
	MODEL_PATH = config('MODEL_PATH')



	tab_master1, tab_master2 = st.tabs(["Modelling", "Data Processing"])

		
	with st.container():

		with tab_master1:
			st.header("Modelling")

			# Select Models
			option = os.path.join(MODEL_PATH, st.selectbox(
				'Choose machine learning model',
				set(
					os.listdir(MODEL_PATH)
				)
			))


			# Section Training/Testing
			tabs1, tabs2 = st.tabs(["Train/Validation", "Test"])
			with tabs1:
				st.header("Training")

				# Import JSON Model Args
				json_args = st.file_uploader('Upload JSON file required', type=['json'], key="json_model_args")

				if json_args:
					json_args = [json.loads(i) for i in json_args]
					st.json(json_args)
				
					# Init Model
					if not st.session_state.model_state:
						st.session_state.model_state = model = Model(
							option,
							train=True,
							args=json_args,
							cuda=True
						)

						# Train Model
						# evaluation = model.train()
						# st.write(
						# 	evaluation
						# )

			

			with tabs2:
				st.header("Testing")

				# Input informal text
				txt = st.text_area('Input informal text for conversion')

				if not st.session_state.model_state:
					st.session_state.model_state = Model(option)

				# Call model 
				if st.button('Convert') or st.session_state.load_state:
					st.session_state.load_state = True

					if model:
						st.write(
							'Converter:', 
							st.session_state.model_state.predict(txt)
						)
					
					st.warning('Model is not initiated', icon="⚠️")
		

		with tab_master2:
			st.header("Data Processing")

			# Import Data for Training & Testing
			file = st.file_uploader("Upload CSV file", type=['csv'], key="csv_training")

			# Data Preprocessing & Splitting
			if file:
				df = Data(
					file, 
					["Kalimat","Formal Sentences"]
				)

				st.write(f"Dataframe Columns: {df.get_columns()}")

				# Request for subset of cols for duplication drop
				preps = st.checkbox("Duplication Drop ", key="preps")
				if "preps" in st.session_state:
					cols = st.text_input(
						"Input columns ends with ',' e.g. Kalimat,Formal Sentences",
						key="subset_cols"
					)

					if "subset_cols" in st.session_state:
						df.preprocessing_null_duplication(cols.split(','))
				
				# Output shape
				st.write(f"Data dimension: {df.get_shape()}")
				st.write(f"Null & Duplication Deletion: {df.get_shape()}")

				df.reformat_data("informal converter")
				df = df.get_data()

				st.dataframe(df)


				test_size = st.slider('Test Size: ', 0, 100, 1) / 100


				train_df, eval_df = train_test_split(
					df,
					test_size=test_size,
					random_state=1000,
					shuffle=True
				)

				st.write(f"Train Data: {train_df.shape} | Validation Data: {eval_df.shape}")


				st.session_state.data_state = [train_df, eval_df]



			