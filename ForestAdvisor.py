import os
from datetime import datetime
from tkinter import BOTH, Button, LEFT, END, Entry, Frame, Label, messagebox, RAISED, StringVar, Tk, TOP
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
environment_requirements = [													\
	("matplotlib", "from matplotlib import pyplot as plt", "Try to run \"pip install matplotlib\" to handle this issue. "), 				\
	("GWO", "from GWO import GWO", "Try to fetch GWO.py from GitHub and put it next to ForestAdvisor.py to handle this issue. ")		\
]


def importLib(t, button) -> None:
	try:
		if button["text"] != "checked" or button["fg"] != state_fg[1]:
			exec(t[1])
		button["text"] = "checked"
		button["fg"] = state_fg[1]
	except Exception as e:
		print("Failed executing \"{0}\". Details are as follows. \n{1}\n\n{2}".format(t[1], e, t[2]))
		showerror("Error", "Failed executing \"{0}\". \nPlease refer to the command prompt window for the details. ".format(t[1]))
		button["text"] = "retry"
		button["fg"] = state_fg[-1]

def train(button, mode) -> None:
	showinfo("Information", "Training finished successfully. ")
	button["fg"] = state_fg[1]

def test(button, mode) -> None:
	showinfo("Information", "Testing finished successfully. ")
	button["fg"] = state_fg[1]

def browse(entry, filetypes = [("All files", "*.*")], isAppend = False) -> None:
	pathString = askopenfilename(initialdir = os.path.abspath(os.path.dirname(__file__)), filetypes = filetypes)
	if pathString:
		if not isAppend:
			entry.delete(0, END)
		entry.insert(END, pathString)

def main() -> int:
	print("{0} - Beginning of Log - {1}".format(Global_Title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	
	# Create Window #
	window = Tk()
	window.title("ForestAdvisor (GWO-LSTM)")
	window.geometry("800x450")
	global trainPathSv, modelPathSv, testPathSv
	trainPathSv, modelPathSv, testPathSv = StringVar(value = "Dataset/Train.csv"), StringVar(value = "Model/GWO-LSTM.h5"), StringVar(value = "Dataset/Test.csv")
	
	# Environment Part #
	environment_frame = Frame(window, relief = RAISED, borderwidth = 2, width = frame_width)
	environment_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	environment_main_label = Label(environment_frame, text = "Environment", font = title_font, fg = title_fg)
	environment_main_label.pack(side = TOP, padx = common_padx)
	for t in environment_requirements:
		Label(environment_frame, text = t[0], font = common_font).pack(side = LEFT, padx = common_padx)
		exec("global {0}\n{0} = Button(environment_frame, text = \"check\", command = lambda:importLib({1}, {0}), font = common_font, fg = state_fg[0])\n{0}.pack(side = LEFT, padx = common_padx)".format("environment_{0}_button".format(t[0]), t))
	
	# Training Part #
	train_frame = Frame(window, relief = RAISED, borderwidth = 2, width = frame_width)
	train_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	train_main_label = Label(train_frame, text = "Train", font = title_font, fg = title_fg)
	train_main_label.pack(side = TOP, padx = common_padx)
	train_path_label = Label(train_frame, text = "Training dataset path: ", font = common_font)
	train_path_label.pack(side = LEFT, padx = common_padx)
	train_path_entry = Entry(train_frame, width = 30, textvariable = trainPathSv, font = common_font)
	train_path_entry.pack(side = LEFT, padx = common_padx)
	train_path_button = Button(train_frame, text = "Browse", command = lambda:browse(train_path_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]), font = common_font, fg = state_fg[None])
	train_path_button.pack(side = LEFT, padx = common_padx)
	train_mode_combobox = Combobox(train_frame, height = 4, width = 20, values = ("LSTM", "GWO-LSTM", "GRU", "SAES"), state = "readonly", font = common_font)
	train_mode_combobox.current(1)
	train_mode_combobox.pack(side = LEFT, padx = common_padx)
	train_main_button = Button(train_frame, text = "Train", command = lambda:train(train_main_button, train_mode_combobox.get()), font = common_font, fg = state_fg[0])
	train_main_button.pack(side = LEFT, padx = common_padx)
	
	# Model Part #
	model_frame = Frame(window, relief = RAISED, borderwidth = 2, width = frame_width)
	model_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	model_main_label = Label(model_frame, text = "Model", font = title_font, fg = title_fg)
	model_main_label.pack(side = TOP, padx = common_padx)
	model_path_label = Label(model_frame, text = "Model path: ", font = common_font)
	model_path_label.pack(side = LEFT, padx = common_padx)
	model_path_entry = Entry(model_frame, width = 30, textvariable = modelPathSv, font = common_font)
	model_path_entry.pack(side = LEFT)
	model_path_button = Button(model_frame, text = "Browse", command = lambda:browse(train_path_entry, filetypes = [("Model Files", "*.h5")]), font = common_font, fg = state_fg[None])
	model_path_button.pack(side = LEFT, padx = common_padx)
	
	# Testing Part #
	test_frame = Frame(window, relief = RAISED, borderwidth = 2, width = frame_width)
	test_frame.pack(side = TOP, fill = BOTH, ipadx = frame_pad, ipady = frame_pad, padx = frame_pad, pady = frame_pad)
	test_main_label = Label(test_frame, text = "Test", font = title_font, fg = title_fg)
	test_main_label.pack(side = TOP, padx = common_padx)
	test_path_label = Label(test_frame, text = "Testing dataset path: ", font = common_font)
	test_path_label.pack(side = LEFT, padx = common_padx)
	test_path_entry = Entry(test_frame, width = 30, textvariable = testPathSv, font = common_font)
	test_path_entry.pack(side = LEFT)
	test_path_button = Button(test_frame, text = "Browse", command = lambda:browse(train_path_entry, filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]), font = common_font, fg = state_fg[None])
	test_path_button.pack(side = LEFT, padx = common_padx)
	test_main_button = Button(test_frame, text = "Test", command = lambda:test(test_main_button, train_mode_combobox.get()), font = common_font, fg = state_fg[0])
	test_main_button.pack(side = LEFT, padx = common_padx)
	
	# Main  Window #
	window.mainloop()
	print("{0} - End of Log - {1}".format(Global_Title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	return EXIT_SUCCESS



if __name__ == "__main__":
	exit(main())