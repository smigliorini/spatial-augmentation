#
# fractal dimension
#
# computing fractal dimension E_0 and E_2 on a 1-dim set of values
#
import numpy as np
import csv
import math
from scipy import stats
# CONST -------------------------
#
DIM = 12

def fd2D (from_x, to_x, start_x, end_x, start_y, end_y, file_name, delim):
# from_x, to_x: seleziona un sottoinsieme di geometrie (rectangles)
# start_x, end_x, start_y, end_y : range di valori da considerare, tipicamente (0,1),(0,1)
# file_name: nome del file csv contenente il dataset di cui calcolare fd
# delim: delimitatore nel file csv
#
	deltax = end_x - start_x
	deltay = end_y - start_y
	cell_width = deltax / pow(2,DIM)
	cell_height = deltay / pow(2,DIM)
	hist = np.zeros((pow(2,DIM),pow(2,DIM)))
	print("dataset: ", file_name)
	print("deltax,y: ", str(deltax), str(deltay), "cell_width: ", str(cell_width), "cell_height: ", str(cell_height))
	with open(file_name, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file, fieldnames=['xmin','ymin','xmax','ymax'], delimiter=delim)
		line_count = 0
		for row in csv_reader:
			if (line_count < from_x):
				line_count += 1
				continue
			if (line_count == to_x):
				break
			xmin = float(row["xmin"])
			ymin = float(row["ymin"])
			xmax = float(row["xmax"])
			ymax = float(row["ymax"])
			# centroide
			x = (xmax+xmin)/2.0
			y = (ymax+ymin)/2.0
			#print("x: ", x, "y: ",y)
			col = int(x/cell_width);
			if (col > 4095):
				col = 4095
			row = int(y/cell_height)
			if (row > 4095):
				row = 4095
			hist[row,col] += 1
			#print("hist incremented: ",int(x/cell_width),int(y/cell_height))
			line_count += 1
			if (line_count % 1000000 == 0):
				print("Line: ", line_count)
	# computing box counting for E2
	x = np.zeros((DIM-1))
	y = np.zeros((DIM-1))
	for i in range(DIM-1):
		sm = 0.0
		step = pow(2,i)
		print("i: ", i)
		for j in range(0,pow(2,DIM),step):
			if (j % 1000 == 0):
				print ("j: ", j)
			for k in range(0,pow(2,DIM),step):
				h = hist[j:j+step,k:k+step]
				# h è una sottomatrice di celle
				sm += pow(np.sum(h),2)
		print("sm: ", sm)
		x[i] = math.log(cell_width * step,2)
		y[i] = math.log(sm,2)

	# selection the portion of curve to interpolate
	start = -1
	end = -1
	for k in range(DIM-1):
		if (start == -1 and abs(y[k+1] - y[k]) >= 0.5):
			start = k
	for k in range(DIM-2,0,-1):
		if (end == -1 and abs(y[k] - y[k-1]) >= 0.5):
                        end = k

	#print("start: ", start, " end: ", end)

	x_new = np.zeros((end-start+1))
	y_new = np.zeros((end-start+1))
	for i in range(end-start+1):
		x_new[i] = x[start+i]
		y_new[i] = y[start+i]

	slope, intercept, r, p, std_err = stats.linregress(x_new, y_new)
	return slope, x_new, y_new, cell_width, line_count

def fd (from_x, to_x, start, end, file_name, field_name, delim):
# from_x, to_x: seleziona un sottoinsieme di geometrie
# start, end: range di valori da considerare, tipicamente (0,1)
# file_name: nome del file contenenti i valori
# field_name: nome del campo del file csv contenente il valore di cui calcolare fd
# delim: delimitatore nel file csv
#
	delta = end - start
	cell_width = delta / pow(2,DIM)
	hist = np.zeros((pow(2,DIM)))
	print("delta: ", str(delta), "cell_width: ", str(cell_width))
	# Reading file
	with open(file_name, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file,delimiter=delim)
		line_count = 0
		for row in csv_reader:
			if (line_count == 0):
				print(f'Column names are: {", ".join(row)}')
			if (line_count < from_x):
				line_count += 1
				continue
			if (line_count == to_x):
				break
			value = float(row[field_name])
			hist[int(value/cell_width)] += 1
			line_count += 1

	# computing box counting for E2
	x = np.zeros((DIM-1))
	y = np.zeros((DIM-1))
	for i in range(DIM-1):
		sm = 0.0
		step = pow(2,i)
		for j in range(0,pow(2,DIM),step):
			h = hist[j:j+step]
			if (step == 1): # h è una singola cella
				sm += pow(h,2)
			else: # h è una sequenza di celle
				sm += pow(np.sum(h),2)
		x[i] = math.log(cell_width * step,2)
		y[i] = math.log(sm,2)

	slope, intercept, r, p, std_err = stats.linregress(x, y)

	return slope, x, y, cell_width, line_count	

def apply_fd(summary_file, ds_path, file_out, from_x, to_x, x1, x2, y1, y2, field_name, dim, delim1, delim2):
# from_x, to_x: seleziona un sottoinsieme di datasets da generare: se ho un file di 10 righe per leggerle tutte mettere 1,10
#
# Reading file
	fd_out = []
	fieldnames = ['datasetName', 'fd']
	with open(summary_file, mode='r') as csv_file, open(file_out, 'w', encoding='UTF8', newline='') as f:
		csv_reader = csv.DictReader(csv_file,delimiter=delim1)
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		line_count = 0
		for row in csv_reader:
			if (line_count == 1):
				print(f'Column names are: {", ".join(row)}')
			if (line_count < from_x):
				line_count += 1
				continue
			if (line_count == to_x+1):
				break
			ds = row["datasetName"]
			file_ds = ds_path + ds + ".csv"
			if (dim == 1):
				fd, x, y, cell_w, cell_h = fd(0, int(row["num_features"]), x1, x2, file_ds, field_name, delim2)
			else:
				fd, x, y, cell_w, cell_h = fd2D(0, int(row["num_features"]), x1, x2, y1, y2, file_ds, delim2)
			
			fd_out.append({'datasetName': ds, 'fd': fd})
			print("fd[",line_count,"]=",fd)
			writer.writerow(fd_out[line_count])
			line_count += 1
