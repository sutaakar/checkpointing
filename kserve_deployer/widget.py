import ipywidgets as widgets
from IPython.display import display
import os
from kubernetes import client, config

class KServeDeployer:
    """A ready-to-use Jupyter widget for deploying KServe InferenceServices."""
    
    def __init__(self, checkpoint_dir='/opt/app-root/src/shared'):
        self.checkpoint_dir = checkpoint_dir
        self.current_namespace = self._detect_current_namespace()
        self._build_ui()
        self.update_checkpoints_dropdown()
        self._update_namespace_dropdown()
        
    def _detect_current_namespace(self):
        """Detects the current namespace from the service account token file."""
        try:
            # Try to read the namespace from the service account token
            namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
            if os.path.exists(namespace_file):
                with open(namespace_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        
        # Fallback to environment variables or default
        return os.environ.get('POD_NAMESPACE', 'default')
        
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
        self.namespace_dropdown = widgets.Dropdown(
            options=['default'],
            value=self.current_namespace if self.current_namespace else 'default',
            description='Namespace:',
            style={'description_width': 'initial'}
        )
        self.custom_namespace_enabled = widgets.Checkbox(
            value=False,
            description='Use custom namespace',
            style={'description_width': 'initial'}
        )
        self.custom_namespace_text = widgets.Text(
            placeholder='Enter custom namespace',
            description='Custom namespace:',
            style={'description_width': 'initial'},
            disabled=True
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
        
        # Link checkbox to enable/disable custom namespace
        self.custom_namespace_enabled.observe(self._on_custom_namespace_change, names='value')
        
        # The VBox holds all our UI elements
        self.ui = widgets.VBox([
            self.kube_api_server, 
            self.kube_token, 
            self.namespace_dropdown,
            self.custom_namespace_enabled,
            self.custom_namespace_text,
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
                
    def _update_namespace_dropdown(self):
        """Updates the namespace dropdown with common namespaces."""
        common_namespaces = ['default', 'kube-system', 'kubeflow', 'opendatahub', 'redhat-ods-applications']
        
        # Add current namespace if it's not in the common list
        if self.current_namespace and self.current_namespace not in common_namespaces:
            common_namespaces.insert(0, self.current_namespace)
            
        self.namespace_dropdown.options = common_namespaces
        
        # Set default value to current namespace
        if self.current_namespace:
            self.namespace_dropdown.value = self.current_namespace
    
    def _on_custom_namespace_change(self, change):
        """Handles changes to the custom namespace checkbox."""
        if change['new']:  # Checkbox is checked
            self.custom_namespace_text.disabled = False
            self.namespace_dropdown.disabled = True
        else:  # Checkbox is unchecked
            self.custom_namespace_text.disabled = True
            self.namespace_dropdown.disabled = False
            
    def _get_selected_namespace(self):
        """Returns the currently selected namespace."""
        if self.custom_namespace_enabled.value and self.custom_namespace_text.value.strip():
            return self.custom_namespace_text.value.strip()
        else:
            return self.namespace_dropdown.value

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
            namespace = self._get_selected_namespace()

            if not all([api_server_url, api_token, checkpoint_path, service_name, namespace]):
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
                namespace=namespace,
                plural='inferenceservices',
                body=inference_service,
            )

            with self.output:
                self.output.clear_output()
                print(f"InferenceService '{service_name}' created successfully!")
                print(f"Using checkpoint from: {checkpoint_path}")
                print(f"Deployed to namespace: {namespace}")

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