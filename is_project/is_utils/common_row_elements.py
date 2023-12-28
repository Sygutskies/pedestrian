def common_row_elements(matrix):
    # If the matrix is empty, return an empty list
    if not matrix:
        return []

    # Initialize the result list as the first row of the matrix
    result = matrix[0]

    # Iterate through the remaining rows of the matrix
    for row in matrix[1:]:

        # Update the result list, keeping only the common elements
        result = [element for element in result if element in row]

    return result