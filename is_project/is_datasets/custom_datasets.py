import torch

class CustomDataset(torch.utils.data.Dataset):
    """
    A custom dataset class that extends the torch.utils.data.Dataset class.
    This class is used to handle data for training and testing machine learning models.
    """

    def __init__(self, data_x, data_y):
        """
        Initializes the dataset with input data and corresponding labels.

        Args:
            data_x (list or numpy array): Input data.
            data_y (list or numpy array): Corresponding labels for the input data.
        """
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data_x)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input data and corresponding label after preprocessing.
        """
        sample = self.preprocess(index)
        return sample

    def preprocess(self, index):
        """
        Preprocesses the data and label at the given index.
        This method can be customized to include any preprocessing steps required.

        Args:
            index (int): Index of the data and label to preprocess.

        Returns:
            tuple: A tuple containing the preprocessed input data and corresponding label.
        """
        x, y = self.data_x[index], self.data_y[index]
        return x, y