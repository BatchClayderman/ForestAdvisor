import os
from datetime import datetime
from tkinter import BOTH, Button, LEFT, END, Entry, Frame, Label, messagebox, RAISED, RIGHT, StringVar, Tk, TOP
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo, showerror
from tkinter.ttk import Combobox
os.chdir(os.path.abspath(os.path.dirname(__file__)))
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EOF = (-1)
Global_Title = "ForestAdvisor"
frame_width = 36
frame_pad = 5
title_font = ("Times New Roman", 16, "bold")
title_fg = "red"
common_font = ("Times New Roman", 12)
common_padx = 10
state_fg = {None:"black", -1:"orange", 0:"blue", 1:"#00EE00"}
environment_requirements = [																\
	("pandas", "global DF, read_csv, read_excel\nfrom pandas import DataFrame as DF, read_csv, read_excel", "Try to run \"pip install pandas\" to handle this issue. "), 		\
	("matplotlib", "from matplotlib import pyplot as plt", "Try to run \"pip install matplotlib\" to handle this issue. "), 							\
	("GWO", "from GWO import GWO", "Try to fetch GWO.py from GitHub and put it next to ForestAdvisor.py to handle this issue. ")					\
]
env_gap_width = 5
controllers = {}


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
		print("Failed executing \"{0}\". Details are as follows. \n{1}\n\n{2}".format(t[1].split("\n")[-1], e, t[2]))
		showerror("Error", "Failed executing \"{0}\". \nPlease refer to the command prompt window for details. ".format(t[1].split("\n")[-1]))
		button["text"] = "retry"
		button["fg"] = state_fg[-1]
		button["state"] = "normal"
		tk.update()
		return False

def checkAll(button) -> None:
	button["state"] = "disable"
	button["text"] = "Checking"
	tk.update()
	bRet = True
	for t in environment_requirements:
		bRet = importLib(t, eval("environment_{0}_button".format(t[0]))) and bRet
	if bRet:
		button["text"] = "Checked"
		button["fg"] = state_fg[1]
	else:
		button["text"] = "Retry"
		button["fg"] = state_fg[-1]
	button["state"] = "normal"
	tk.update()
	return bRet

def setControllersState(bState:bool) -> None:
	if bState:
		for controller in list(controllers.keys()):
			controller["state"] = controllers[controller]
	else:
		for controller in list(controllers.keys()):
			controller["state"] = "disable"

def train(button, mode) -> None:
	setControllersState(False)
	button["text"] = "Training"
	button["fg"] = state_fg[0]
	button["state"] = "normal"
	tk.update()
	trainPath = trainPathSv.get()
	if os.path.splitext(trainPath.lower())[1] in (".txt", ".csv"):
		try:
			pf = read_csv(trainPath)
		except Exception as e:
			print("Failed reading training dataset. Details are as follows. \n{0}".format(e))
			showerror("Error", "Failed reading training dataset. \nPlease refer to the command prompt window for details. ")
			button["text"] = "Train"
			button["fg"] = state_fg[-1]
			setControllersState(True)
			tk.update()
			return
	showinfo("Information", "Training finished successfully. ")
	button["text"] = "Train"
	button["fg"] = state_fg[1]
	setControllersState(True)
	tk.update()

def test(button, mode) -> None:
	setControllersState(False)
	button["text"] = "Testing"
	button["fg"] = state_fg[0]
	button["state"] = "normal"
	tk.update()
	testPath = testPathSv.get()
	if os.path.splitext(testPath.lower())[1] in (".txt", ".csv"):
		try:
			pf = read_csv(testPath)
		except Exception as e:
			print("Failed reading testing dataset. Details are as follows. \n{0}".format(e))
			showerror("Error", "Failed reading testing dataset. \nPlease refer to the command prompt window for details. ")
			button["text"] = "Test"
			button["fg"] = state_fg[-1]
			setControllersState(True)
			tk.update()
			return
	showinfo("Information", "Testing finished successfully. ")
	button["text"] = "Test"
	button["fg"] = state_fg[1]
	setControllersState(True)
	tk.update()

def browse(entry, filetypes = [("All files", "*.*")], isAppend = False) -> None:
	pathString = askopenfilename(initialdir = os.path.abspath(os.path.dirname(__file__)), filetypes = filetypes)
	if pathString:
		if not isAppend:
			entry.delete(0, END)
		entry.insert(END, pathString)

def main() -> int:
	print("{0} - Beginning of Log - {1}".format(Global_Title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	
	# Create Window #
	global tk
	tk = Tk()
	tk.title("ForestAdvisor (GWO-LSTM)")
	tk.geometry("800x450")
	global trainPathSv, modelPathSv, testPathSv
	trainPathSv, modelPathSv, testPathSv = StringVar(value = "Dataset/Train.csv"), StringVar(value = "Model/GWO-LSTM.h5"), StringVar(value = "Dataset/Test.csv")
	
	# Environment Part #
	environment_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	environment_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	environment_main_label = Label(environment_frame, text = "Environment", font = title_font, fg = title_fg)
	environment_main_label.pack(side = TOP, padx = common_padx)
	for t in environment_requirements:
		Label(environment_frame, text = " " * env_gap_width + t[0], font = common_font).pack(side = LEFT, padx = common_padx)
		exec("global {0}\n{0} = Button(environment_frame, text = \"check\", command = lambda:importLib({1}, {0}), font = common_font, fg = state_fg[0])\n{0}.pack(side = LEFT, padx = common_padx)".format("environment_{0}_button".format(t[0]), t))
	environment_main_button = Button(environment_frame, text = "Check all", command = lambda:checkAll(environment_main_button), font = common_font, fg = state_fg[0])
	environment_main_button.pack(side = RIGHT, padx = common_padx)
	
	# Training Part #
	train_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	train_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	train_main_label = Label(train_frame, text = "Train", font = title_font, fg = title_fg)
	train_main_label.pack(side = TOP, padx = common_padx)
	train_path_label = Label(train_frame, text = "Training dataset path: ", font = common_font)
	train_path_label.pack(side = LEFT, padx = common_padx)
	train_path_entry = Entry(train_frame, width = 30, textvariable = trainPathSv, font = common_font)
	train_path_entry.pack(side = LEFT, padx = common_padx)
	controllers[train_path_entry] = "normal"
	train_path_button = Button(train_frame, text = "Browse", command = lambda:browse(train_path_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]), font = common_font, fg = state_fg[None])
	train_path_button.pack(side = LEFT, padx = common_padx)
	controllers[train_path_button] = "normal"
	train_mode_combobox = Combobox(train_frame, height = 4, width = 20, values = ("LSTM", "GWO-LSTM", "GRU", "SAES"), state = "readonly", font = common_font)
	train_mode_combobox.current(1)
	train_mode_combobox.pack(side = LEFT, padx = common_padx)
	controllers[train_mode_combobox] = "readonly"
	train_main_button = Button(train_frame, text = "Train", command = lambda:train(train_main_button, train_mode_combobox.get()), font = common_font, fg = state_fg[0])
	train_main_button.pack(side = LEFT, padx = common_padx)
	controllers[train_main_button] = "normal"
	
	# Model Part #
	model_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	model_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	model_main_label = Label(model_frame, text = "Model", font = title_font, fg = title_fg)
	model_main_label.pack(side = TOP, padx = common_padx)
	model_path_label = Label(model_frame, text = "Model path: ", font = common_font)
	model_path_label.pack(side = LEFT, padx = common_padx)
	model_path_entry = Entry(model_frame, width = 30, textvariable = modelPathSv, font = common_font)
	model_path_entry.pack(side = LEFT)
	controllers[model_path_entry] = "normal"
	model_path_button = Button(model_frame, text = "Browse", command = lambda:browse(train_path_entry, filetypes = [("Model Files", "*.h5")]), font = common_font, fg = state_fg[None])
	model_path_button.pack(side = LEFT, padx = common_padx)
	controllers[model_path_button] = "normal"
	
	# Testing Part #
	test_frame = Frame(tk, relief = RAISED, borderwidth = 2, width = frame_width)
	test_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	test_main_label = Label(test_frame, text = "Test", font = title_font, fg = title_fg)
	test_main_label.pack(side = TOP, padx = common_padx)
	test_path_label = Label(test_frame, text = "Testing dataset path: ", font = common_font)
	test_path_label.pack(side = LEFT, padx = common_padx)
	test_path_entry = Entry(test_frame, width = 30, textvariable = testPathSv, font = common_font)
	test_path_entry.pack(side = LEFT)
	controllers[test_path_entry] = "normal"
	test_path_button = Button(test_frame, text = "Browse", command = lambda:browse(train_path_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]), font = common_font, fg = state_fg[None])
	test_path_button.pack(side = LEFT, padx = common_padx)
	controllers[test_path_button] = "normal"
	test_main_button = Button(test_frame, text = "Test", command = lambda:test(test_main_button, train_mode_combobox.get()), font = common_font, fg = state_fg[0])
	test_main_button.pack(side = LEFT, padx = common_padx)
	controllers[test_main_button] = "normal"
	
	# Main  Window #
	tk.mainloop()
	print("{0} - End of Log - {1}".format(Global_Title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	return EXIT_SUCCESS



if __name__ == "__main__":
	exit(main())