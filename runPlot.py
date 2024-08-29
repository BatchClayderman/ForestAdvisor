import os
try:
	from pandas import read_csv
	from matplotlib import pyplot as plt
except:
	print("Failed to import pandas or matplotlib. Please use pip to install them. ")
	print("Please press the enter key to exit. ")
	input()
	exit(-1)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EOF = -1
folders = ["CarbonPredictorFinland", "CarbonPredictorHeilongjiang", "CarbonPredictorMaine"]
filename = "comparison.csv"
plotFolder = "Plot"
xCol = "Index"
yCol = "GWO-LSTM"
ext = ".pdf"
dpi = 1200


def handleFolder(fd:str) -> bool:
	folder = str(fd)
	if folder in ("", ".", "./", ".\\"):
		return True
	elif os.path.exists(folder):
		return os.path.isdir(folder)
	else:
		try:
			os.makedirs(folder)
			return True
		except:
			return False

def main():
	if not handleFolder(plotFolder):
		print("The result folder is not created successfully. ")
		print("Please press the enter key to exit. ")
		input()
		return EOF
	bRet = True
	for folder in folders:
		source = os.path.join(folder, filename)
		target = folder + ext
		target = os.path.join(plotFolder, os.path.split(target)[1])
		print("Generating: \"{0}\"".format(target))
		try:
			df = read_csv(source, engine = "python", skipfooter = 7)
			plt.rcParams["figure.dpi"] = 300
			plt.rcParams["savefig.dpi"] = 300
			plt.rcParams["font.family"] = "Times New Roman"
			plt.rcParams["font.size"] = 12
			plt.plot(df[xCol], df[yCol], color = "red")
			plt.xlabel("Time (day)")
			plt.ylabel("Carbon emission (mMt)")
			plt.rcParams["figure.dpi"] = dpi
			plt.rcParams["savefig.dpi"] = dpi
			plt.savefig(target, bbox_inches = "tight")
			plt.close()
			print("Generated: \"{0}\"".format(target))
		except:
			print("Failed: \"{0}\"".format(target))
			bRet = False
	print("Please press the enter key to exit. ")
	input()
	return EXIT_SUCCESS if bRet else EXIT_FAILURE



if __name__ == "__main__":
	exit(main())