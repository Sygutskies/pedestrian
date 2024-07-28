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

# Helper function to normalize keypoints
def normalize_keypoints(keypoints, box):
    x_left, y_top, x_right, y_bottom = box
    width, height = x_right - x_left, y_bottom - y_top
    return [(x - x_left) / width if i % 2 == 0 else (x - y_top) / height for i, x in enumerate(keypoints)]