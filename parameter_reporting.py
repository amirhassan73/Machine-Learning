
#Generate parameter report:
def writer(matrix, output_file):
    with open(output_file, "w+") as report:
        for k in range(matrix.shape[1]):
            for j in range(matrix.shape[0]):
                report.write(str(matrix[j,k]))
                report.write("|")