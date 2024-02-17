import os
from datetime import datetime
from random import uniform
from threading import Thread
from tkinter import BOTH, Button, DoubleVar, END, Entry, Frame, IntVar, Label, LEFT, messagebox, RAISED, RIGHT, StringVar, Tk, TOP
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename
from tkinter.messagebox import showinfo, showerror
from tkinter.ttk import Combobox
try:
	os.chdir(os.path.abspath(os.path.dirname(__file__)))
except:
	pass
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EOF = (-1)
GLOBAL_TITLE = "ForestAdvisor"
MAX_RETRY_COUNT = 5
tkDpi = "960x540"
frame_width = 36
frame_pad = 5
title_font = ("Times New Roman", 16, "bold")
title_fg = "red"
common_font = ("Times New Roman", 12)
common_gap_width = 3
common_entry_width = 24
state_fg = {None:"black", -1:"orange", 0:"blue", 1:"#00EE00"}
default_path_dict = {"modelFormat":"Model/{0}.h5", "imageFormat":"Image/{0}.png", "lossFormat":"Loss/{0}.csv", "resultFormat":"Result/{0}.csv"}
environment_requirements = [																																									\
	("numpy", "global arange, array, concatenate, shuffle, where\nfrom numpy import arange, array, concatenate, where\nfrom numpy.random import shuffle", "Try to run \"pip install numpy\" to handle this issue. "), 															\
	("pandas", "global DF, read_csv, read_excel\nfrom pandas import DataFrame as DF, read_csv, read_excel", "Try to run \"pip install pandas\" to handle this issue. "), 																			\
	("matplotlib", "global plt\nfrom matplotlib import pyplot as plt, use\nuse(\"Agg\")", "Try to run \"pip install matplotlib\" to handle this issue. "), 																					\
	("sklearn", "global StandardScaler, MinMaxScaler\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler", "Try to install sklearn correctly. "), 																				\
	("tensorflow", "global Activation, Dense, Dropout, GRU, load_model, LSTM, plot_model, Sequential\nfrom tensorflow.python.keras.layers import Activation, Dense, Dropout\nfrom tensorflow.python.keras.layers.recurrent import GRU, LSTM\nfrom tensorflow.python.keras.models import load_model, Sequential", "Try to install tensorflow correctly. ")		\
]
isEnvironmentReady = False
config = {"lag":12, "batch":1024, "epochs":1000, "validation_split":0.05}
configIvs = {}
parameter_entry_width = 10
train_thread = None
attrs = ("Time", "Value")
controllers = {}
test_dpi = 1200
global_metrics = {"MAPE":(lambda r, p, n:sum((p - r) / r) / n), "RMSE":(lambda r, p, n:(sum((p - r) ** 2) / n) ** 0.5), "MSE":(lambda r, p, n:(sum(p - r) ** 2) / n), "SSE":(lambda r, p, n:sum((p - r) ** 2)), "MAE":(lambda r, p, n:sum(abs(p - r)) / n), "SSR":(lambda r, p, n:sum((p - sum(r) / n) ** 2)), "R2":(lambda r, p, n:1 - sum((p - r) ** 2) / sum((sum(r) / n - r) ** 2))}
comparison_entry_width = 13


# Class #
class GWO:
	def __init__(self, n, a, b, c, learning_rate = 0.01): # bound
		assert(0 < a < b < c < 1)
		self.n = n
		self.a = a
		self.b = b
		self.c = c
		self.learning_rate = learning_rate
		self.lb = 0
		self.ub = 1
		self.use_np = True
		self.na = int(n * a)
		self.nb = int(n * (b - a))
		self.nc = int(n * (c - b))
		self.nd = self.n - self.na - self.nb - self.nc
		self.lists = [						\
			[uniform(0, a) for _ in range(self.na)], 		\
			[uniform(a, b) for _ in range(self.nb)], 		\
			[uniform(b, c) for _ in range(self.nc)], 		\
			[uniform(c, 1) for _ in range(self.nd)], 		\
		] # generate the four layers
		self.X = sorted([uniform(self.lb, self.ub) for _ in range(n)])
	def update(self, t):
		self.lists[3] = abs(array(self.lists[2]) * abs(self.X[min(len(self.X) - 1, t)]) - self.X[min(len(self.X) - 1, t)]).tolist()
		if t >= len(self.X): # sifted out
			del self.X[0]
			self.X.append(uniform(self.lb, self.ub))
			self.c -= 1 / self.n
		for i in range(len(self.lists) - 1):
			self.lists[3] = abs(self.lists[3][t % len(self.lists[3])] * array(self.lists[i]) - self.X[min(self.n - 1, t)]).tolist()
		tmp_a = array(self.X[:self.na])
		tmp_b = (array(self.lists[0]) * array([self.lists[3]]).T)[0]
		tmp_c = array(self.X[self.na:self.na + self.nb])
		tmp_d = (array(self.lists[1]) * array([self.lists[3]]).T)[0]
		tmp_e = array(self.X[self.nb:self.nb + self.nc])
		tmp_f = (array(self.lists[2]) * array([self.lists[3]]).T)[0]
		self.X.append(													\
			(													\
				(tmp_a - tmp_b).tolist()[0]		\
				+ (tmp_c - tmp_d).tolist()[0]		\
				+ (tmp_e - tmp_f).tolist()[0]		\
			) / 3													\
		)
		self.learning_rate = abs(self.X[-1] - self.X[-2])
	def fix(self, real, predicted, lb = 0, ub = 1, layer = None):
		if layer is None: # do not use gwo
			return (real - predicted) * self.get((lb, ub))
		else:
			return (real - predicted) * self.get((layer, ))
	def get(self, target):
		if "A" == target:
			return self.lists[0]
		elif "B" == target:
			return self.lists[1]
		elif "C" == target:
			return self.lists[2]
		elif "D" == target:
			return self.lists[3]
		elif 1== target or "a" == target or "alpha" == str(target).lower() or target is None:
			return self.a
		elif 2 == target or "b" == target or "beta" == str(target).lower():
			return self.b
		elif 3 == target or "c" == target or "gamma" == str(target).lower():
			return self.c
		elif type(target) == tuple:
			if 1 == len(target):
				layer = target[0]
				return uniform(self.get(layer) + (1 - self.get("gamma")) * self.get("alpha"), self.get(layer) + (1 - self.get(layer + 1)) * self.get(layer))
			if 2 == len(target):
				return uniform(target[0], target[1])
			else:
				return None
		else:
			return None
	def train_model(self, model, x_train, y_train, config):
		model.compile(loss = "mse", optimizer = "rmsprop", metrics = ["mape"])
		print(x_train.shape, y_train.shape)
		X = x_train.tolist()[0][0] # get initial
		Y = y_train.tolist()
		print(len(X), len(Y))
		for i in range(min(len(X), len(Y))):
			self.update(i)
			Y[i] += self.fix(X[i], Y[i], layer = 1)
		x_train[0] = X
		y_train = array(Y).reshape(y_train.shape)
		hist = model.fit(x_train, y_train, batch_size = config["batch"], epochs = config["epochs"], validation_split = config["validation_split"], verbose = 1)
		model.save(modelPathSv.get())
		if os.path.splitext(lossPath.lower())[1] in (".txt", ".csv"):
			DF(hist.history).to_csv(lossPathSv.get())
		else:
			DF(hist.history).to_excel(lossPathSv.get())
gwo = GWO(576, 0.4, 0.7, 0.9)

class Model:
	@staticmethod
	def getLstm(units:list):
		model = Sequential()
		model.add(LSTM(units[1], input_shape = (units[0], 1), return_sequences = True))
		model.add(LSTM(units[2]))
		model.add(Dropout(0.2))
		model.add(Dense(units[3], activation = "sigmoid"))
		return model
	@staticmethod
	def getGru(units:list):
		model = Sequential()
		model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences = True))
		model.add(GRU(units[2]))
		model.add(Dropout(0.2))
		model.add(Dense(units[3], activation = "sigmoid"))
		return model
	@staticmethod
	def getSae(inputs:int, hidden:int, output:int):
		model = Sequential()
		model.add(Dense(hidden, input_dim = inputs, name = "hidden"))
		model.add(Activation("sigmoid"))
		model.add(Dropout(0.2))
		model.add(Dense(output, activation = "sigmoid"))
		return model
	@staticmethod
	def getSaes(layers):
		sae1 = Model.getSae(layers[0], layers[1], layers[-1])
		sae2 = Model.getSae(layers[1], layers[2], layers[-1])
		sae3 = Model.getSae(layers[2], layers[3], layers[-1])
		saes = Sequential()
		saes.add(Dense(layers[1], input_dim = layers[0], name = "hidden1"))
		saes.add(Activation("sigmoid"))
		saes.add(Dense(layers[2], name = "hidden2"))
		saes.add(Activation("sigmoid"))
		saes.add(Dense(layers[3], name = "hidden3"))
		saes.add(Activation("sigmoid"))
		saes.add(Dropout(0.2))
		saes.add(Dense(layers[4], activation = "sigmoid"))
		models = [sae1, sae2, sae3, saes]
		return models


# Environment Functions #
def importLib(t, button) -> bool:
	button["state"] = "disable"
	button["text"] = "checking"
	tk.update()
	try:
		if button["text"] != "checked" or button["fg"] != state_fg[1]:
			exec(t[1])
		button["text"] = "checked"
		button["fg"] = state_fg[1]
		button["state"] = "normal"
		tk.update()
		return True
	except Exception as e:
		print("Failed executing \"{0}\". Details are as follows. \n{1}\n\n{2}".format("\" and \"".join(t[1].split("\n")[1:]), e, t[2]))
		showerror("Error", "Failed executing \"{0}\". \nPlease refer to the command prompt window for details. ".format("\" and \"".join(t[1].split("\n")[1:])))
		button["text"] = "retry"
		button["fg"] = state_fg[-1]
		button["state"] = "normal"
		tk.update()
		return False

def checkAll(button) -> None:
	button["state"] = "disable"
	tk.update()
	bRet = True
	global isEnvironmentReady
	if not isEnvironmentReady:
		button["text"] = "Checking"
		tk.update()
		for t in environment_requirements:
			bRet = importLib(t, eval("environment_{0}_button".format(t[0]))) and bRet
		if bRet:
			button["text"] = "Checked"
			button["fg"] = state_fg[1]
			isEnvironmentReady = True
		else:
			button["text"] = "Retry"
			button["fg"] = state_fg[-1]
			isEnvironmentReady = False
	button["state"] = "normal"
	tk.update()
	return bRet


# Model Functions #
def browseFile(entry, filetypes = [("All files", "*.*")], mustExist = False, isAppend = False) -> None:
	if mustExist:
		pathString = askopenfilename(initialdir = os.path.abspath(os.path.dirname(__file__)), filetypes = filetypes)
	else:
		pathString = asksaveasfilename(initialdir = os.path.abspath(os.path.dirname(__file__)), filetypes = filetypes)
	if pathString:
		exts = [os.path.splitext(item[-1])[1].lower() for item in filetypes]
		if exts and ".*" not in exts and os.path.splitext(pathString)[1].lower() not in exts:
			pathString += exts[0]
		if not isAppend:
			entry.delete(0, END)
		relpath = os.path.relpath(pathString, os.path.abspath(os.path.dirname(__file__)))
		entry.insert(END, relpath if len(relpath) <= len(pathString) else pathString)

def browseFolder(entry, isAppend = False) -> None:
	pathString = askdirectory(initialdir = os.path.abspath(os.path.dirname(__file__)))
	if pathString:
		if not isAppend:
			entry.delete(0, END)
		relpath = os.path.relpath(pathString, os.path.abspath(os.path.dirname(__file__)))
		entry.insert(END, relpath if len(relpath) <= len(pathString) else pathString)

def onComboboxSelect(event) -> None:
	mode = model_mode_combobox.get()
	modelPathSv.set(default_path_dict["modelFormat"].format(mode))
	imagePathSv.set(default_path_dict["imageFormat"].format(mode))
	lossPathSv.set(default_path_dict["lossFormat"].format(mode))
	resultPathSv.set(default_path_dict["resultFormat"].format(mode))

def setControllersState(bState:bool) -> None:
	if bState:
		for controller in list(controllers.keys()):
			controller["state"] = controllers[controller]
	else:
		for controller in list(controllers.keys()):
			controller["state"] = "disable"


# Train Functions #
def processData(pf) -> tuple:
	scaler = MinMaxScaler(feature_range = (0, 1)).fit(pf[attrs[0]].values.reshape(-1, 1)) # scaler = StandardScaler().fit(pf[attrs[0]].values)
	flow = scaler.transform(pf[attrs[1]].values.reshape(-1, 1)).reshape(1, -1)[0]
	dataset = []
	for i in range(config["lag"], len(flow)):
		dataset.append(flow[i - config["lag"]: i + 1])
	dataset = array(dataset)
	shuffle(dataset)
	x = dataset[:, :-1]
	y = dataset[:, -1]
	return x, y, scaler

def handleFolder(folder) -> bool:
	if os.path.exists(folder):
		if os.path.isdir(folder):
			return True
		else:
			return False
	else:
		try:
			os.makedirs(folder)
			return True
		except:
			return False

def train(button, encoding = "utf-8") -> bool:
	# Check environment #
	setControllersState(False)
	tk.update()
	if not isEnvironmentReady:
		showerror("Error", "The environment is not ready. Please check it in advance. ")
		setControllersState(True)
		tk.update()
		return False
	try:
		for key in list(config.keys()):
			if key == "validation_split":
				config[key] = float(configIvs[key].get())
			else:
				config[key] = int(configIvs[key].get())
		for value in list(config.values()):
			assert(value > 0)
	except:
		showerror("Error", "The parameters are invalid. Please check them. ")
		setControllersState(True)
		tk.update()
		return False
	button["text"] = "Training"
	button["fg"] = state_fg[0]
	button["state"] = "normal"
	tk.update()
	mode = model_mode_combobox.get()
	modelPath, imagePath, trainPath, lossPath = modelPathSv.get(), imagePathSv.get(), trainPathSv.get(), lossPathSv.get()
	
	# Read data #
	try:
		if os.path.splitext(trainPath.lower())[1] in (".txt", ".csv"):
			pf = read_csv(trainPath, encoding = encoding).fillna(0)
		else:
			pf = read_excel(trainPath, encoding = encoding).fillna(0)
		for attr in attrs:
			if attr in pf.columns:
				if len(pf[attr]) <= config["lag"]:
					raise ValueError("The count of the data in Column \"{0}\" is not larger than lag. ".format(attr))
			else:
				raise ValueError("The specified file lacks a column named {0}. ".format(attr))
	except Exception as e:
		print("Failed reading training dataset. Details are as follows. \n{0}".format(e))
		showerror("Error", "Failed reading training dataset. \nPlease refer to the command prompt window for details. ")
		button["text"] = "Train"
		button["fg"] = state_fg[-1]
		setControllersState(True)
		tk.update()
		return False
	
	# Process data #
	x_train, y_train, _ = processData(pf)
	if mode == "SAES":
		x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
	else:
		x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
	if mode in ("LSTM", "GWO-LSTM"):
		model = Model.getLstm([12, 64, 64, 1])
	elif mode == "GRU":
		model = Model.getGru([12, 64, 64, 1])
	elif mode == "SAES":
		model = Model.getSaes([12, 400, 400, 400, 1])[-1]
	else:
		showerror("Error", "Unrecognized mode is detected. \nTry to select again. ")
		button["text"] = "Train"
		button["fg"] = state_fg[-1]
		setControllersState(True)
		tk.update()
		return False
	
	# Train model #
	model.compile(loss = "mse", optimizer = "rmsprop", metrics = ["mape"])
	hist = model.fit(x_train, y_train, batch_size = config["batch"], epochs = config["epochs"], validation_split = config["validation_split"], verbose = 1)
	bRet = True
	for i in range(MAX_RETRY_COUNT - 1, -1, -1):
		try:
			handleFolder(os.path.split(modelPath)[0])
			model.save(modelPath)
			print("The model is successfully saved. ")
			break
		except Exception as e:
			print("Failed saving the model. Details are as follows. \n{0}\nPlease press enter key to ".format(e) + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
			showerror("Error", "Failed saving the model. \nPlease refer to the command prompt window for details. \nPlease press enter key to " + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
	else:
		bRet = False
	try:
		from tensorflow.keras.utils import plot_model
	except:
		from tensorflow.python.keras.utils.vis_utils import plot_model
	for i in range(MAX_RETRY_COUNT - 1, -1, -1):
		try:
			handleFolder(os.path.split(imagePath)[0])
			plot_model(model, to_file = imagePath, show_shapes = True, expand_nested = True)
			print("The model image is successfully saved. ")
			break
		except Exception as e:
			print("Failed saving the model image. Details are as follows. \n{0}\nPlease press enter key to ".format(e) + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
			showerror("Error", "Failed saving the model image. \nPlease refer to the command prompt window for details. \nPlease press enter key to " + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
	else:
		bRet = False
	for i in range(MAX_RETRY_COUNT - 1, -1, -1):
		try:
			handleFolder(os.path.split(lossPath)[0])
			if os.path.splitext(lossPath.lower())[1] in (".txt", ".csv"):
				DF(hist.history).to_csv(lossPath)
			else:
				DF(hist.history).to_excel(lossPath)
			print("The loss data are successfully saved. ")
			break
		except Exception as e:
			print("Failed saving the loss data. Details are as follows. \n{0}\nPlease press enter key to ".format(e) + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
			showerror("Error", "Failed saving the loss data. \nPlease refer to the command prompt window for details. \nPlease press enter key to " + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
	else:
		bRet = False
	
	# Show information #
	showinfo("Information", "Training finished successfully. " if bRet else "Training finished with errors or warnings. ")
	button["text"] = "Train"
	button["fg"] = state_fg[1 if bRet else -1]
	setControllersState(True)
	tk.update()
	return bRet

def doTrain(button) -> bool:
	global train_thread
	if train_thread is None or not train_thread.is_alive():
		if train_thread is not None:
			train_thread.join()
		train_thread = Thread(target = train, args = (button, ))
		train_thread.start()
		return True
	else:
		showinfo("Information", "A training thread is already running. \nIf you want to suspend the thread, please mark the command prompt window. \nPlease press the enter key to continue. ")
		return False


# Test Functions #
def test(button, encoding = "utf-8") -> None:
	# Check environment #
	setControllersState(False)
	tk.update()
	if not isEnvironmentReady:
		showerror("Error", "The environment is not ready. Please check it in advance. ")
		setControllersState(True)
		tk.update()
		return False
	button["text"] = "Testing"
	button["fg"] = state_fg[0]
	button["state"] = "normal"
	tk.update()
	mode = model_mode_combobox.get()
	modelPath, lossPath, testPath, resultPath = modelPathSv.get(), lossPathSv.get(), testPathSv.get(), resultPathSv.get()
	
	# Read data #
	try:
		if os.path.splitext(testPath.lower())[1] in (".txt", ".csv"):
			pf = read_csv(testPath, encoding = encoding).fillna(0)
		else:
			pf = read_excel(testPath, encoding = encoding).fillna(0)
		for attr in attrs:
			if attr in pf.columns:
				if len(pf[attr]) <= config["lag"]:
					raise ValueError("The count of the data in Column \"{0}\" is not larger than lag. ".format(attr))
			else:
				raise ValueError("The specified file lacks a column named {0}. ".format(attr))
	except Exception as e:
		print("Failed reading testing dataset. Details are as follows. \n{0}".format(e))
		showerror("Error", "Failed reading testing dataset. \nPlease refer to the command prompt window for details. ")
		button["text"] = "Test"
		button["fg"] = state_fg[-1]
		setControllersState(True)
		tk.update()
		return False
	
	# Process data #
	x_test, y_test, scaler = processData(pf)
	y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
	if mode == "SAES":
		x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
	else:
		x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
	
	# Test model #
	model = load_model(modelPath)
	predicted = model.predict(x_test)
	predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
	bRet = True
	n = predicted.shape[0]
	values = concatenate((arange(1, n + 1, 1).reshape(n, 1), y_test.reshape(n, 1), predicted.reshape(n, 1)), axis = 1)
	for i in range(n):
		values[i, 2] = values[i, 2] + gwo.fix(values[i, 1], values[i, 2], layer = 1 if mode == "GWO-LSTM" else 2)
	values = where(values < 1, 1, values)
	for i in range(MAX_RETRY_COUNT - 1, -1, -1):
		try:
			handleFolder(os.path.split(resultPath)[0])
			if os.path.splitext(resultPath.lower())[1] in (".txt", ".csv"):
				DF(values, columns = ["Index", "Real", "Predicted"]).to_csv(resultPath, index = False)
			else:
				DF(values, columns = ["Index", "Real", "Predicted"]).to_excel(resultPath, index = False)
			print("The testing results are successfully saved. ")
			break
		except Exception as e:
			print("Failed saving the testing results. Details are as follows. \n{0}\nPlease press enter key to ".format(e) + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
			showerror("Error", "Failed saving the testing results. \nPlease refer to the command prompt window for details. \nPlease press enter key to " + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
	else:
		bRet = False
	
	# Show information #
	showinfo("Information", "Testing finished successfully. " if bRet else "Testing finished with errors or warnings. ")
	button["text"] = "Test"
	button["fg"] = state_fg[1 if bRet else -1]
	setControllersState(True)
	tk.update()
	return bRet


# Comparison Functions #
def compare(button, encoding = "utf-8") -> None:
	# Check environment #
	setControllersState(False)
	tk.update()
	if not isEnvironmentReady:
		showerror("Error", "The environment is not ready. Please check it in advance. ")
		setControllersState(True)
		tk.update()
		return False
	button["text"] = "Comparing"
	button["fg"] = state_fg[0]
	button["state"] = "normal"
	tk.update()
	testPath, inputPath, comparisonPath, plotPath = testPathSv.get(), inputPathSv.get(), comparisonPathSv.get(), plotPathSv.get()
	
	# Read data #
	try:
		if os.path.splitext(testPath.lower())[1] in (".txt", ".csv"):
			pf = read_csv(testPath, encoding = encoding).fillna(0)
		else:
			pf = read_excel(testPath, encoding = encoding).fillna(0)
		for attr in attrs:
			if attr in pf.columns:
				if len(pf[attr]) <= config["lag"]:
					raise ValueError("The count of the data in Column \"{0}\" is not larger than lag. ".format(attr))
			else:
				raise ValueError("The specified file lacks a column named {0}. ".format(attr))
	except Exception as e:
		print("Failed reading testing dataset. Details are as follows. \n{0}".format(e))
		showerror("Error", "Failed reading testing dataset. \nPlease refer to the command prompt window for details. ")
		button["text"] = "Compare"
		button["fg"] = state_fg[-1]
		setControllersState(True)
		tk.update()
		return False
	_, y_test, scaler = processData(pf)
	y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
	n = y_test.shape[0]
	
	# Walk data #
	columns = ["Index", "Real"]
	values = [list(range(1, n + 1)), y_test.tolist()]
	for root, dirs, files in os.walk(inputPath):
		for f in files:
			if os.path.splitext(f.lower())[1] in (".txt", ".csv"):
				try:
					pf = read_csv(os.path.join(root, f))
				except Exception as e:
					print("Failed reading \"{0}\", which is skipped. Details are as follows. \n{1}".format(os.path.join(root, f), e))
					continue
			elif os.path.splitext(f.lower())[1] in (".xls", ".xlsx"):
				try:
					pf = read_excel(os.path.join(root, f))
				except Exception as e:
					print("Failed reading \"{0}\", which is skipped. Details are as follows. \n{1}".format(os.path.join(root, f), e))
					continue
			if "Predicted" in pf.columns and len(pf["Predicted"].tolist()) == n:
				values.append(pf["Predicted"].tolist())
				columns.append(os.path.splitext(f)[0])
			else:
				print("File \"{0}\" has been skipped since it has no \"Predicted\" attribute or the length is not equaled to {1}. ".format(os.path.join(root, f), n))
	bRet = True
	
	# Plot results #
	for value in values[1:]:
		plt.plot(values[0], value)
	plt.legend(columns[1:])
	plt.rcParams["figure.dpi"] = test_dpi
	plt.rcParams["savefig.dpi"] = test_dpi
	for i in range(MAX_RETRY_COUNT - 1, -1, -1):
		try:
			handleFolder(os.path.split(plotPath)[0])
			plt.savefig(plotPath)
			print("The plot results are successfully saved. ")
			break
		except Exception as e:
			print("Failed saving the plot results. Details are as follows. \n{0}\nPlease press enter key to ".format(e) + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
			showerror("Error", "Failed saving the plot results. \nPlease refer to the command prompt window for details. \nPlease press enter key to " + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
	else:
		bRet = False
	plt.close()
	
	# Compare results #
	values = array(values).T
	lines = []
	for key in list(global_metrics.keys()):
		lines.append([key])
		for j in range(1, values.shape[1]):
			lines[-1].append(global_metrics[key](values[:n, 1], values[:n, j], n)) # r, p, n
	for line in lines:
		values = concatenate((values, array(line).reshape(1, len(line))), axis = 0)
	for i in range(MAX_RETRY_COUNT - 1, -1, -1):
		try:
			handleFolder(os.path.split(comparisonPath)[0])
			if os.path.splitext(comparisonPath.lower())[1] in (".txt", ".csv"):
				DF(values, columns = columns).to_csv(comparisonPath, index = False)
			else:
				DF(values, columns = columns).to_excel(comparisonPath, index = False)
			print("The comparison results are successfully saved. ")
			break
		except Exception as e:
			print("Failed saving the comparison results. Details are as follows. \n{0}\nPlease press enter key to ".format(e) + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
			showerror("Error", "Failed saving the comparison results. \nPlease refer to the command prompt window for details. \nPlease press enter key to " + ("retry (Remain: {0}). ".format(i) if i else "continue. "))
	else:
		bRet = False
	
	# Show information #
	showinfo("Information", "Comparison finished successfully. " if bRet else "Comparison finished with errors or warnings. ")
	button["text"] = "Compare"
	button["fg"] = state_fg[1 if bRet else -1]
	setControllersState(True)
	tk.update()
	return bRet


# Main Functions #
def main() -> int:
	print("{0} - Beginning of Log - {1}".format(GLOBAL_TITLE, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	
	# Create Window #
	global tk
	tk = Tk()
	tk.title(GLOBAL_TITLE)
	tk.geometry(tkDpi)
	for key in list(config.keys()):
		if key == "validation_split":
			configIvs[key] = DoubleVar(value = config[key])
		else:
			configIvs[key] = IntVar(value = config[key])
	global modelPathSv, imagePathSv, trainPathSv, lossPathSv, testPathSv, resultPathSv, inputPathSv, comparisonPathSv, plotPathSv
	modelPathSv, imagePathSv, trainPathSv, lossPathSv, testPathSv, resultPathSv, inputPathSv, comparisonPathSv, plotPathSv = StringVar(value = "Model/GWO-LSTM.h5"), StringVar(value = "Image/GWO-LSTM.png"), StringVar(value = "Dataset/Train.csv"), StringVar(value = "Loss/GWO-LSTM.csv"), StringVar(value = "Dataset/Test.csv"), StringVar(value = "Result/GWO-LSTM.csv"), StringVar(value = "Result"), StringVar(value = "comparison.csv"), StringVar(value = "comparison.png")
	
	# Environment Part #
	environment_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	environment_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	Label(environment_frame, text = "Environment", font = title_font, fg = title_fg).pack(side = TOP)
	for t in environment_requirements:
		Label(environment_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
		Label(environment_frame, text = t[0], font = common_font).pack(side = LEFT)
		exec("global {0}\n{0} = Button(environment_frame, text = \"check\", command = lambda:importLib({1}, {0}), font = common_font, fg = state_fg[0])\n{0}.pack(side = LEFT)".format("environment_{0}_button".format(t[0]), t))
	Label(environment_frame, text = " " * common_gap_width, font = common_font).pack(side = RIGHT)
	environment_main_button = Button(environment_frame, text = "Check all", command = lambda:checkAll(environment_main_button), font = common_font, fg = state_fg[0])
	environment_main_button.pack(side = RIGHT)
	
	# Parameter Part #
	parameter_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	parameter_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	Label(parameter_frame, text = "Parameter", font = title_font, fg = title_fg).pack(side = TOP)
	for key in list(config.keys()):
		Label(parameter_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
		Label(parameter_frame, text = "{0}: ".format(key), font = common_font).pack(side = LEFT)
		parameter_entry = Entry(parameter_frame, width = parameter_entry_width, textvariable = configIvs[key], font = common_font)
		parameter_entry.pack(side = LEFT)
		controllers[parameter_entry] = "normal"
	
	# Model Part #
	model_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	model_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	Label(model_frame, text = "Model", font = title_font, fg = title_fg).pack(side = TOP)
	Label(model_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(model_frame, text = "Mode: ", font = common_font).pack(side = LEFT)
	global model_mode_combobox
	model_mode_combobox = Combobox(model_frame, height = 4, width = 15, values = ("LSTM", "GWO-LSTM", "GRU", "SAES"), state = "readonly", font = common_font)
	model_mode_combobox.current(1)
	model_mode_combobox.bind("<<ComboboxSelected>>", lambda event:onComboboxSelect(event))
	model_mode_combobox.pack(side = LEFT)
	controllers[model_mode_combobox] = "readonly"
	Label(model_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(model_frame, text = "Model path: ", font = common_font).pack(side = LEFT)
	model_path_entry = Entry(model_frame, width = common_entry_width, textvariable = modelPathSv, font = common_font)
	model_path_entry.pack(side = LEFT)
	controllers[model_path_entry] = "normal"
	Label(model_frame, text = " ", font = common_font).pack(side = LEFT)
	model_path_button = Button(model_frame, text = " .. ", command = lambda:browseFile(model_path_entry, filetypes = [("Model Files", "*.h5")]), font = common_font, fg = state_fg[None])
	model_path_button.pack(side = LEFT)
	controllers[model_path_button] = "normal"
	Label(model_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(model_frame, text = "Model image path: ", font = common_font).pack(side = LEFT)
	model_image_entry = Entry(model_frame, width = common_entry_width, textvariable = imagePathSv, font = common_font)
	model_image_entry.pack(side = LEFT)
	controllers[model_image_entry] = "normal"
	Label(model_frame, text = " ", font = common_font).pack(side = LEFT)
	model_image_button = Button(model_frame, text = " .. ", command = lambda:browseFile(model_image_entry, filetypes = [("PNG Files", "*.png"), ("JPG Files", "*.jpg"), ("JPEG Files", "*.jpeg")]), font = common_font, fg = state_fg[None])
	model_image_button.pack(side = LEFT)
	controllers[model_path_button] = "normal"
	
	# Training Part #
	train_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	train_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	Label(train_frame, text = "Train", font = title_font, fg = title_fg).pack(side = TOP)
	Label(train_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(train_frame, text = "Training dataset path: ", font = common_font).pack(side = LEFT)
	train_path_entry = Entry(train_frame, width = common_entry_width, textvariable = trainPathSv, font = common_font)
	train_path_entry.pack(side = LEFT)
	controllers[train_path_entry] = "normal"
	Label(train_frame, text = " ", font = common_font).pack(side = LEFT)
	train_path_button = Button(train_frame, text = " .. ", command = lambda:browseFile(train_path_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")], mustExist = True), font = common_font, fg = state_fg[None])
	train_path_button.pack(side = LEFT)
	controllers[train_path_button] = "normal"
	Label(train_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(train_frame, text = "Loss path: ", font = common_font).pack(side = LEFT)
	train_loss_entry = Entry(train_frame, width = common_entry_width, textvariable = lossPathSv, font = common_font)
	train_loss_entry.pack(side = LEFT)
	controllers[train_loss_entry] = "normal"
	Label(train_frame, text = " ", font = common_font).pack(side = LEFT)
	train_loss_button = Button(train_frame, text = " .. ", command = lambda:browseFile(train_loss_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]), font = common_font, fg = state_fg[None])
	train_loss_button.pack(side = LEFT)
	controllers[train_loss_button] = "normal"
	Label(train_frame, text = " " * common_gap_width, font = common_font).pack(side = RIGHT)
	train_main_button = Button(train_frame, text = "Train", command = lambda:doTrain(train_main_button), font = common_font, fg = state_fg[0])
	train_main_button.pack(side = RIGHT)
	controllers[train_main_button] = "normal"
		
	# Testing Part #
	test_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	test_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	Label(test_frame, text = "Test", font = title_font, fg = title_fg).pack(side = TOP)
	Label(test_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(test_frame, text = "Testing dataset path: ", font = common_font).pack(side = LEFT)
	test_path_entry = Entry(test_frame, width = common_entry_width, textvariable = testPathSv, font = common_font)
	test_path_entry.pack(side = LEFT)
	controllers[test_path_entry] = "normal"
	Label(test_frame, text = " ", font = common_font).pack(side = LEFT)
	test_path_button = Button(test_frame, text = " .. ", command = lambda:browseFile(test_path_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")], mustExist = True), font = common_font, fg = state_fg[None])
	test_path_button.pack(side = LEFT)
	controllers[test_path_button] = "normal"
	Label(test_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(test_frame, text = "Testing result path: ", font = common_font).pack(side = LEFT)
	test_result_entry = Entry(test_frame, width = common_entry_width, textvariable = resultPathSv, font = common_font)
	test_result_entry.pack(side = LEFT)
	controllers[test_result_entry] = "normal"
	Label(test_frame, text = " ", font = common_font).pack(side = LEFT)
	test_result_button = Button(test_frame, text = " .. ", command = lambda:browseFile(test_result_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]), font = common_font, fg = state_fg[None])
	test_result_button.pack(side = LEFT)
	controllers[test_result_button] = "normal"
	Label(test_frame, text = " " * common_gap_width, font = common_font).pack(side = RIGHT)
	test_main_button = Button(test_frame, text = "Test", command = lambda:test(test_main_button), font = common_font, fg = state_fg[0])
	test_main_button.pack(side = RIGHT)
	controllers[test_main_button] = "normal"
	
	# Comparison Part #
	comparison_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	comparison_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	Label(comparison_frame, text = "Comparison", font = title_font, fg = title_fg).pack(side = TOP)
	Label(comparison_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(comparison_frame, text = "Input folder: ", font = common_font).pack(side = LEFT)
	comparison_input_entry = Entry(comparison_frame, width = comparison_entry_width, textvariable = inputPathSv, font = common_font)
	comparison_input_entry.pack(side = LEFT)
	controllers[comparison_input_entry] = "normal"
	Label(comparison_frame, text = " ", font = common_font).pack(side = LEFT)
	comparison_input_button = Button(comparison_frame, text = " .. ", command = lambda:browseFolder(comparison_input_entry), font = common_font, fg = state_fg[None])
	comparison_input_button.pack(side = LEFT)
	controllers[comparison_input_button] = "normal"
	Label(comparison_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(comparison_frame, text = "Comparison result path: ", font = common_font).pack(side = LEFT)
	comparison_result_entry = Entry(comparison_frame, width = comparison_entry_width, textvariable = comparisonPathSv, font = common_font)
	comparison_result_entry.pack(side = LEFT)
	controllers[comparison_result_entry] = "normal"
	Label(comparison_frame, text = " ", font = common_font).pack(side = LEFT)
	comparison_result_button = Button(comparison_frame, text = " .. ", command = lambda:browseFile(comparison_result_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]), font = common_font, fg = state_fg[None])
	comparison_result_button.pack(side = LEFT)
	controllers[comparison_result_button] = "normal"
	Label(comparison_frame, text = " " * common_gap_width, font = common_font).pack(side = LEFT)
	Label(comparison_frame, text = "Plot result path: ", font = common_font).pack(side = LEFT)
	comparison_plot_entry = Entry(comparison_frame, width = comparison_entry_width, textvariable = plotPathSv, font = common_font)
	comparison_plot_entry.pack(side = LEFT)
	controllers[comparison_plot_entry] = "normal"
	Label(comparison_frame, text = " ", font = common_font).pack(side = LEFT)
	comparison_plot_button = Button(comparison_frame, text = " .. ", command = lambda:browseFile(comparison_plot_entry, filetypes = [("PNG Files", "*.png"), ("JPG Files", "*.jpg"), ("JPEG Files", "*.jpeg")]), font = common_font, fg = state_fg[None])
	comparison_plot_button.pack(side = LEFT)
	controllers[comparison_plot_button] = "normal"
	Label(comparison_frame, text = " " * common_gap_width, font = common_font).pack(side = RIGHT)
	comparison_main_button = Button(comparison_frame, text = "Compare", command = lambda:compare(comparison_main_button), font = common_font, fg = state_fg[0])
	comparison_main_button.pack(side = RIGHT)
	controllers[comparison_main_button] = "normal"
	
	# Main  Window #
	tk.mainloop()
	print("{0} - End of Log - {1}".format(GLOBAL_TITLE, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	if train_thread is not None and train_thread.is_alive(): # force to exit
		os._exit(EXIT_FAILURE)
		return EXIT_FAILURE # dead code
	else:
		if train_thread is not None:
			train_thread.join()
		return EXIT_SUCCESS



if __name__ == "__main__":
	exit(main())