


#Fetch parameters
def parameter_fetch(matrix, source_file):
    value=0
    with open(source_file, "r") as params:
        for line in params:
            values=line.split("|")

        for y in range(matrix.shape[1]):
            for x in range(matrix.shape[0]):
                if value < len(values):
                    try:
                        matrix[x,y] = float(values[value])
                        value+=1
                    except:
                        continue
                else: exit
        

