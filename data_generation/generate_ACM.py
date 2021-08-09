import scipy.io

data_file_path = 'ACM.mat'
data = scipy.io.loadmat(data_file_path)
matrix = data['PvsA'].toarray()
print(matrix.shape, data['PvsA'].shape)
with open('../dataset_ACM/paper-author.txt', 'w') as fw:
    for i in range(data['PvsA'].shape[0]):
        for j in range(data['PvsA'].shape[1]):
            if matrix[i][j] != 0:
                fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
matrix = data['PvsP'].toarray()
print(matrix.shape, data['PvsP'].shape)
with open('../dataset_ACM/paper-paper.txt', 'w') as fw:
    for i in range(data['PvsP'].shape[0]):
        for j in range(data['PvsP'].shape[1]):
            if matrix[i][j] != 0:
                fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
matrix = data['PvsL'].toarray()
print(matrix.shape, data['PvsL'].shape)
with open('../dataset_ACM/paper-field.txt', 'w') as fw:
    for i in range(data['PvsL'].shape[0]):
        for j in range(data['PvsL'].shape[1]):
            if matrix[i][j] != 0:
                fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
matrix = data['PvsC'].toarray()
print(matrix.shape, data['PvsC'].shape)
with open('../dataset_ACM/paper-venue.txt', 'w') as fw:
    for i in range(data['PvsC'].shape[0]):
        for j in range(data['PvsC'].shape[1]):
            if matrix[i][j] != 0:
                fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')