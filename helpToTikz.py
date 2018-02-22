# To edit file texts, with the specific purpose to edit tikz files

import numpy as np

def readEliminate(filename, text, nextLines):
	# Function to open a text file, and eliminate all lines that
	# star with the text line. It also eliminates the following 
	# predefined next lines.
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	counterNextLine = 0
	for i in d:
		if (i[0:len(text)] != text)and(counterNextLine==0):
			f.write(i)
		else:
			counterNextLine +=1
			if counterNextLine > nextLines:
				counterNextLine = 0
	f.truncate()
	f.close()

def readInsert(filename, text_start, text_insert):
	# Function to read lines and insert in the middle of a text 
	# an additional text.
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	for i in d:
		if i[0:len(text_start)] == text_start:
			f.write(i[0:len(text_start)]+text_insert+i[len(text_start):len(i)])
		else:
			f.write(i)
	f.truncate()
	f.close()

def readReplace(filename, text_start, text_replace, text_insert):
	# Function that dead lines, takes the one that starts as text_start
	# finds the substring text_replace, and replaces it for text_insert.
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	boolToken = False
	for i in d:
		boolToken = False
		if i[0:len(text_start)] == text_start:
			i = i.replace(text_replace,text_insert)
			f.write(i)
		else:
			f.write(i)
	f.truncate()
	f.close()

def readReplaceAll(filename, text_replace, text_insert):
	# Function mean to replace all ocurrances of the desired value
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	boolToken = False
	for i in d:
			i = i.replace(text_replace,text_insert)
			f.write(i)
	f.truncate()
	f.close()

def readNewline(filename, text, newline):
	# Function to insert a complete new line
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	for i in d:
		f.write(i)
		if i[1:len(text)+1] == text:
			f.write(newline)
	f.truncate()
	f.close()
	
def scaleTikzLabels(filename, scaling):
	# Function that will divide the LABELS of the ticks figure with
	# an predefined value.
	## retrieve the Ticks labels
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	text = "xticklabels={"
	xtickslabels=''
	for i in d:
		f.write(i)
		if i[0:len(text)] == text:
			xtickslabels= i[len(text):len(i)]
	f.truncate()
	f.close()	
	while xtickslabels != '':
		if xtickslabels[-1]!='}':
			xtickslabels = xtickslabels[0:-1]
		else:
			xtickslabels=xtickslabels[0:-1]
			break
	xtickslabels=xtickslabels[1:-1]
	## Convert the retrieved string of labels to float and divide it
	newlabels = np.array(map(float,xtickslabels.split(",")))/scaling
	## Convert to string in a appropiate format and replace the new labels
	newlabelsString = ''
	for value in newlabels:
		aux = str(np.round(value,1))
		newlabelsString += aux+","
	## convert to string and replace the new labels
	readReplace(filename, "xticklabels={", xtickslabels, newlabelsString[0:-1])

def readRetrieve(filename, text):
	# Function that for a specific line of text, will get the array 
	# of floats that describes.
	# it assumes that the last element in text is '{'
	if text[-1] != '{':
		print('This is not working')
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	values=''
	for i in d:
		f.write(i)
		if i[0:len(text)] == text:
			values= i[len(text):len(i)]
			print(values)
			break
	while values != '':
		print(values)
		if values[-1]!='}':
			values = values[1:-1]
		else:
			values=values[1:-1]
			break
	return map(float, values.split(","))

