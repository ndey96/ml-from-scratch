#LibSVM reader

def load_svm(f):
    indices = []
    labels = []
    data = []

    for line in f:
        line_parts = line.split()
        if len(line_parts) == 0:
            continue

        target = line_parts[0]
        attributes = line_parts[1:]

        labels.append(float(target))

        prev_i = -1
        num_attributes = len(attributes)

        for i in range(0, num_attributes):
            str_index, value = attributes[i].split(':', 1)
            index = int(str_index)

            indices.append(index)
            data.append(float(value))
            prev_i = index

    return (data, indices, labels)

f = open('a1a.txt', 'r')
data, indices, labels = load_svm(f)
print len(indices)
print len(data)
