# This was in pca to look at the stuff that was in the tight cluster of the PCA

X2 = X[X[:,0] < -0.5]
print(X2)
id_list = np.array(id_list)
blob = id_list[X[:,0] < -0.5]
blob2 = id_list[X[:,0] >= -0.5]
yb = y[X[:,0] < -0.5]
yb2 = y[X[:,0] >= -0.5]

doids = 0
omims = 0
doid_list = []
omim_list = []
for row in blob:
	if row[1].split(':')[0] == 'DOID':
		doids += 1
		doid_list += [row]
	elif row[1].split(':')[0] == 'OMIM':
		omims += 1
		omim_list += [row]
	else:
		print('something is wrong...')
		print(row)

print('----------------')
if doids < omims:
	print(doid_list)
else:
	print(omim_list)


doids = 0
omims = 0
doid_list = []
omim_list = []
for row in blob2:
	if row[1].split(':')[0] == 'DOID':
		doids += 1
		doid_list += [row]
	elif row[1].split(':')[0] == 'OMIM':
		omims += 1
		omim_list += [row]
	else:
		print('something is wrong...')
		print(row)

print('----------------')
if doids < omims:
	print(doid_list)
else:
	print(omim_list)
