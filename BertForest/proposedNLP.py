import os
from sys import exit
try:
	import torch
	from transformers import BertTokenizer, BertModel
	from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as e:
	print("Failed importing libraries. Details are as follows. \n{0}\n\nPlease press the enter key to exit. ".format(e))
	input()
	exit(-1)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EOF = (-1)
modelFolderPath = "modelFolder"
inputFolderPath = "inputFolder"
outputFolderPath = "outputFolder"


def getModel(modelFolderP:str) -> tuple:
	if os.path.isdir(modelFolderP):
		try:
			return (BertTokenizer.from_pretrained(modelFolderP), BertModel.from_pretrained(modelFolderP))
		except Exception as e:
			print("Failed reading the model folder path \"{0}\". Details are as follows. {1}".format(modelFolderP, e))
			return (None, None)
	else:
		print("The model folder path \"{0}\" should be a readable folder. Please check it. ".format(modelFolderP))
		return (None, None)

def getTxt(filepath, index = 0) -> str: # get .txt content
	coding = ("utf-8", "gbk", "utf-16") # codings
	if 0 <= index < len(coding): # in the range
		try:
			with open(filepath, "r", encoding = coding[index]) as f:
				content = f.read()
			return content[1:] if content.startswith("\ufeff") else content # if utf-8 with BOM, remove BOM
		except (UnicodeError, UnicodeDecodeError):
			return getTxt(filepath, index + 1) # recursion
		except:
			return None
	else:
		return None # out of range

def handleFolder(folder:str) -> bool:
	if folder in ("", ".", "./"):
		return True
	elif os.path.exists(folder):
		return os.path.isdir(folder)
	else:
		try:
			os.makedirs(folder)
			return True
		except:
			return False

def main() -> int:
	tokenizer, model = getModel(modelFolderPath)
	if tokenizer and model:
		print("Get the model information successfully. ")
	else:
		print("Failed getting the model information. Please press the enter key to exit. ")
		exit(EOF)
	total_cnt = 0
	success_cnt = 0
	for root, dirs, files in os.walk(inputFolderPath):
		for fname in files:
			if os.path.splitext(fname)[1].lower() in (".txt", ):
				total_cnt += 1
				inputFp = os.path.join(root, fname)
				text = getTxt(inputFp)
				
				# Initial Bert #
				inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512)
				input_ids = inputs["input_ids"]
				with torch.no_grad():
					outputs = model(**inputs)
				last_hidden_states = outputs.last_hidden_state
				words = [tokenizer.decode([input_id], skip_special_tokens=True, clean_up_tokenization_spaces=False) for input_id in input_ids.squeeze().tolist()]

				# Compute significance based on TF-IDF #
				vectorizer = TfidfVectorizer()
				tfidf_matrix = vectorizer.fit_transform([' '.join(words)])
				tfidf_feature_names = vectorizer.get_feature_names_out()
				tfidf_scores = tfidf_matrix.toarray()[0]
				
				# Combine word vector of BERT and TF-IDF scores #
				word_importances = torch.norm(last_hidden_states, dim=2).squeeze().tolist()
				combined_scores = {}
				for word, tfidf_score in zip(tfidf_feature_names, tfidf_scores):
					bert_index = words.index(word) if word in words else -1
					if bert_index != -1:
						bert_score = word_importances[bert_index]
						combined_scores[word] = tfidf_score * bert_score
				
				# Sort and output #
				sorted_keywords = sorted(combined_scores.items(), key = lambda x: x[1], reverse = True)
				middlePath = os.path.relpath(inputFp, inputFolderPath)
				outputFp = os.path.join(outputFolderPath, middlePath)
				try:
					handleFolder(os.path.split(outputFp)[0])
					with open(outputFp, "w", encoding = "utf-8") as f:
						for word, score in sorted_keywords:
							f.write("{0}: {1}\n".format(word, score))
					success_cnt += 1
					print("Write to \"{0}\" successfully. ".format(outputFp))
				except Exception as e:
					print("Failed writing to \"{0}\". Details are as follows. \n{1}".format(outputFp, e))
	if total_cnt:
		print("This program finished with success rate of {0} / {1} = {2:.2f}%. ".format(success_cnt, total_cnt, success_cnt * 100 / total_cnt))
	else:
		print("Nothing was processed. ")
	print("Please press the enter key to exit. ")
	input()
	return EXIT_SUCCESS if success_cnt == total_cnt else EXIT_FAILURE



if __name__ == "__main__":
	exit(main())