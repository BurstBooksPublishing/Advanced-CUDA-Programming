__global__ void updateSparseMatrix(
    int *rowPtr, int *colIndices, float *values,
    int *newRowPtr, int *newColIndices, float *newValues,
    int numRows, int numNewEntries) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows) {
        int start = rowPtr[row];
        int end = rowPtr[row + 1];
        int newStart = newRowPtr[row];
        int newEnd = newRowPtr[row + 1];

        // Merge existing and new entries for the current row
        int i = start, j = newStart, k = start;
        
        while (i < end && j < newEnd) {
            if (colIndices[i] < newColIndices[j]) {
                values[k] = values[i];
                colIndices[k] = colIndices[i];
                i++;
            } else {
                values[k] = newValues[j];
                colIndices[k] = newColIndices[j];
                j++;
            }
            k++;
        }

        // Copy remaining existing entries
        while (i < end) {
            values[k] = values[i];
            colIndices[k] = colIndices[i];
            i++;
            k++;
        }

        // Copy remaining new entries
        while (j < newEnd) {
            values[k] = newValues[j];
            colIndices[k] = newColIndices[j];
            j++;
            k++;
        }

        // Update row pointer for the next row
        rowPtr[row + 1] = k;
    }
}