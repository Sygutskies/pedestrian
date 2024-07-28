from is_models.motion_model import MotionModel
from is_models.phone_model import PhoneModel
from torchview import draw_graph
from torchinfo import summary

class Build:
    def __init__(self, model_type="motion", features_number=12, init_weights=None, log_dir="/."):
        """
        Initialize the Build class.

        Args:
            model_type (str): Type of the model ("motion" or "phone").
            features_number (int): Number of input features.
            init_weights (str, optional): Path to initial weights for the model. Defaults to None.
            log_dir (str): Directory to save logs and model architecture.
        """
        self.model_type = model_type
        self.features_number = features_number
        self.init_weights = init_weights
        self.log_dir = log_dir

    def run(self):
        """
        Build and visualize the model.

        Returns:
            model: The instantiated model.
        """
        # Determine the model class based on the type
        if self.model_type == "motion":
            model_cls = MotionModel
        elif self.model_type == "phone":
            model_cls = PhoneModel
        else:
            raise ValueError("Wrong model type")
        
        # Instantiate the model
        model = model_cls(input_size=self.features_number)
        
        # Draw and save the model architecture graph
        draw_graph(model, input_size=(512, 12, self.features_number), expand_nested=True, save_graph=True, filename=f"{self.log_dir}/model_architecture")
        
        # Print the model summary
        print(summary(model, input_size=(512, 12, 12)))
        
        return model
