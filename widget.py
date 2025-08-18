import ipywidgets as widgets
from IPython.display import display
import os
from kubernetes import client

class KServeDeployer:
    """A ready-to-use Jupyter widget for deploying KServe InferenceServices."""
    
    def __init__(self, checkpoint_dir='/opt/app-root/src/shared'):
        self.checkpoint_dir = checkpoint_dir
        self._build_ui()
        self.update_checkpoints_dropdown()
        
    def _build_ui(self):
        """Constructs all the UI components of the widget."""
        self.kube_api_server = widgets.Text(
            value='https://kubernetes.default.svc',
            placeholder='Enter Kubernetes API Server URL',
            description='K8s API Server:',
            style={'description_width': 'initial'}
        )
        self.kube_token = widgets.Password(
            description='K8s Token:',
            placeholder='Enter Kubernetes Bearer Token',
            style={'description_width': 'initial'}
        )
        self.checkpoints_dropdown = widgets.Dropdown(
            options=[],
            description='Checkpoints:',
        )
        self.inference_service_name = widgets.Text(
            value='my-inference-service',
            placeholder='Enter a name for the InferenceService',
            description='Service Name:',
        )
        self.create_button = widgets.Button(
            description='Create InferenceService',
            button_style='success',
            icon='check'
        )
        self.output = widgets.Output()
        
        # Link button to its function
        self.create_button.on_click(self._create_inference_service)
        
        # The VBox holds all our UI elements
        self.ui = widgets.VBox([
            self.kube_api_server, 
            self.kube_token, 
            self.checkpoints_dropdown, 
            self.inference_service_name, 
            self.create_button, 
            self.output
        ])

    def find_checkpoints(self):
        """Finds directories containing checkpoint files."""
        path = self.checkpoint_dir
        if not os.path.isdir(path):
            return []
        checkpoint_dirs = set()
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(('.ckpt', '.pt', '.pth', '.bin')):
                    checkpoint_dirs.add(root)
                    break
        return sorted(list(checkpoint_dirs))

    def update_checkpoints_dropdown(self):
        """Populates the dropdown with found checkpoints."""
        checkpoints = self.find_checkpoints()
        self.checkpoints_dropdown.options = checkpoints
        if not checkpoints:
            self.checkpoints_dropdown.disabled = True
            self.create_button.disabled = True
            with self.output:
                print(f"No checkpoints found in {self.checkpoint_dir}.")

    def _create_inference_service(self, b):
        """Creates an InferenceService object when the button is clicked."""
        with self.output:
            self.output.clear_output()
            print('Creating InferenceService...')

        try:
            # Get values from widgets
            api_server_url = self.kube_api_server.value
            api_token = self.kube_token.value
            checkpoint_path = self.checkpoints_dropdown.value
            service_name = self.inference_service_name.value

            if not all([api_server_url, api_token, checkpoint_path, service_name]):
                 with self.output:
                    self.output.clear_output()
                    print("Error: All fields must be filled out.")
                 return

            # --- Kubernetes API Configuration ---
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False # WARNING: Insecure, not for production
            
            api_client = client.ApiClient(configuration)
            api = client.CustomObjectsApi(api_client)

            # --- InferenceService Definition ---
            inference_service = {
                'apiVersion': 'serving.kserve.io/v1beta1',
                'kind': 'InferenceService',
                'metadata': {'name': service_name, 'annotations': {'serving.kserve.io/deploymentMode': 'ModelMesh'}},
                'spec': {
                    'predictor': {
                        'model': {
                            'modelFormat': {'name': 'pytorch'},
                            'storageUri': f'pvc://{checkpoint_path}'
                        }
                    }
                }
            }

            api.create_namespaced_custom_object(
                group='serving.kserve.io',
                version='v1beta1',
                namespace='default', # IMPORTANT: Change this to your target namespace
                plural='inferenceservices',
                body=inference_service,
            )

            with self.output:
                self.output.clear_output()
                print(f"InferenceService '{service_name}' created successfully!")
                print(f"Using checkpoint from: {checkpoint_path}")

        except client.ApiException as e:
            with self.output:
                self.output.clear_output()
                print(f"Kubernetes API Error: {e.reason}\nBody: {e.body}")
        except Exception as e:
            with self.output:
                self.output.clear_output()
                print(f"An unexpected error occurred: {e}")
    
    def _ipython_display_(self):
        """Allows the object to be displayed directly in a notebook."""
        display(self.ui)