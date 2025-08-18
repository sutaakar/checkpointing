import ipywidgets as widgets
from IPython.display import display
import threading
import time
import re
from datetime import datetime
from kubernetes import client, config, watch

class KServeDeployer:
    """A ready-to-use Jupyter widget for deploying KServe InferenceServices.
    
    This widget automatically detects checkpoints from PyTorchJob logs and deploys them
    as KServe InferenceServices. It's designed for Kubernetes environments where training
    and notebook pods typically have different filesystem access.
    
    Args:
        path_mapping (dict): Optional mapping of training paths to deployment paths, e.g.
            {'/opt/model-dir': 'pvc://shared-storage', '/training': 'pvc://models'}
    
    Examples:
        # Basic usage - detects checkpoints from PyTorchJob logs
        KServeDeployer()
        
        # With path mapping for different storage URIs
        KServeDeployer(path_mapping={'/opt/model-dir': 'pvc://shared-storage'})
    """
    
    def __init__(self, path_mapping=None):
        self.path_mapping = path_mapping or {}  # Map training paths to deployment paths
        self.current_namespace = self._detect_current_namespace()
        self.log_monitor_thread = None
        self.stop_monitoring = False
        self.last_checkpoint_time = {}
        self.detected_checkpoints = set()  # Checkpoints found from logs
        self._build_ui()
        self.update_checkpoints_dropdown()
        self._update_namespace_dropdown()
        self._update_pytorchjob_dropdown()
        
    def _detect_current_namespace(self):
        """Detects the current namespace from environment variables."""
        import os
        
        # Try to read the namespace from the service account token
        try:
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
            value='default',
            description='Namespace:',
            style={'description_width': 'initial'}
        )
        self.pytorchjob_dropdown = widgets.Dropdown(
            options=[],
            description='PyTorchJob:',
            style={'description_width': 'initial'}
        )
        self.refresh_jobs_button = widgets.Button(
            description='Refresh Jobs',
            button_style='info',
            icon='refresh',
            layout=widgets.Layout(width='120px')
        )
        self.monitor_logs_checkbox = widgets.Checkbox(
            value=False,
            description='Monitor PyTorchJob logs for new checkpoints',
            style={'description_width': 'initial'},
            layout={'width': 'max-content'}
        )
        self.log_status = widgets.HTML(
            value='<i>Log monitoring: Inactive</i>',
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
        
        # Link buttons to their functions
        self.create_button.on_click(self._create_inference_service)
        self.refresh_jobs_button.on_click(self._refresh_jobs_button_click)
        
        # Link checkbox to enable/disable log monitoring
        self.monitor_logs_checkbox.observe(self._on_monitor_logs_change, names='value')
        
        # Link namespace dropdown to update PyTorchJob list
        self.namespace_dropdown.observe(self._on_namespace_change, names='value')
        
        # Link kubernetes credentials to refresh PyTorchJob list when filled
        self.kube_api_server.observe(self._on_credentials_change, names='value')
        self.kube_token.observe(self._on_credentials_change, names='value')
        
        # Create HBox for PyTorchJob dropdown and refresh button
        self.pytorchjob_row = widgets.HBox([
            self.pytorchjob_dropdown,
            self.refresh_jobs_button
        ])
        
        # The VBox holds all our UI elements
        self.ui = widgets.VBox([
            self.kube_api_server, 
            self.kube_token, 
            self.namespace_dropdown,
            self.pytorchjob_row,
            self.monitor_logs_checkbox,
            self.log_status,
            self.checkpoints_dropdown, 
            self.inference_service_name, 
            self.create_button, 
            self.output
        ])

    def _map_checkpoint_path(self, training_path):
        """Maps a training checkpoint path to deployment path."""
        # Apply user-defined path mapping
        for training_prefix, deployment_prefix in self.path_mapping.items():
            if training_path.startswith(training_prefix):
                return training_path.replace(training_prefix, deployment_prefix, 1)
        
        # Default: use original path (assume it's accessible to InferenceService)
        return training_path
    
    def find_checkpoints(self):
        """Returns checkpoints detected from PyTorchJob logs."""
        if not self.detected_checkpoints:
            return []
            
        # Map all detected checkpoints to deployment paths
        mapped_checkpoints = []
        for original_path in self.detected_checkpoints:
            mapped_path = self._map_checkpoint_path(original_path)
            mapped_checkpoints.append(mapped_path)
            
        return sorted(mapped_checkpoints)

    def update_checkpoints_dropdown(self):
        """Populates the dropdown with checkpoints detected from PyTorchJob logs."""
        checkpoints = self.find_checkpoints()
        
        if not checkpoints:
            self.checkpoints_dropdown.disabled = True
            self.create_button.disabled = True
            self.checkpoints_dropdown.options = []
            with self.output:
                print("No checkpoints detected yet.")
                print("💡 Start PyTorchJob log monitoring to automatically detect checkpoints.")
        else:
            self.checkpoints_dropdown.disabled = False
            self.create_button.disabled = False
            self.checkpoints_dropdown.options = checkpoints
            with self.output:
                print(f"✅ Found {len(checkpoints)} checkpoint(s) from PyTorchJob logs:")
                for cp in checkpoints:
                    print(f"  - {cp}")
                    
                if self.path_mapping:
                    print(f"\n📍 Applied path mapping: {self.path_mapping}")
                
    def _update_namespace_dropdown(self):
        """Updates the namespace dropdown with common namespaces."""
        common_namespaces = ['default', 'kube-system', 'kubeflow', 'opendatahub', 'redhat-ods-applications']
        
        # Add current namespace if it's not in the common list
        if self.current_namespace and self.current_namespace not in common_namespaces:
            common_namespaces.insert(0, self.current_namespace)
            
        # Update the dropdown options first
        self.namespace_dropdown.options = common_namespaces
        
        # Set default value to current namespace (it's now guaranteed to be in the options)
        if self.current_namespace:
            self.namespace_dropdown.value = self.current_namespace
    
    def _get_selected_namespace(self):
        """Returns the currently selected namespace."""
        return self.namespace_dropdown.value
    
    def _update_pytorchjob_dropdown(self):
        """Updates the PyTorchJob dropdown with available jobs in the selected namespace."""
        try:
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value.strip()
            api_token = self.kube_token.value.strip()
            
            if not api_server_url or not api_token:
                self.pytorchjob_dropdown.options = []
                if hasattr(self, 'output'):  # Check if output widget exists
                    with self.output:
                        print("Please provide Kubernetes API server URL and token to load PyTorchJobs")
                return
                
            # Configure Kubernetes client
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            api = client.CustomObjectsApi(api_client)
            
            # Get PyTorchJobs from the selected namespace
            namespace = self._get_selected_namespace()
            pytorchjobs = api.list_namespaced_custom_object(
                group='kubeflow.org',
                version='v1',
                namespace=namespace,
                plural='pytorchjobs'
            )
            
            job_names = []
            for job in pytorchjobs.get('items', []):
                job_name = job['metadata']['name']
                status = job.get('status', {}).get('conditions', [])
                # Add status indicator
                if any(c.get('type') == 'Running' and c.get('status') == 'True' for c in status):
                    job_names.append(f"{job_name} (Running)")
                elif any(c.get('type') == 'Succeeded' and c.get('status') == 'True' for c in status):
                    job_names.append(f"{job_name} (Completed)")
                else:
                    job_names.append(f"{job_name} (Other)")
                    
            self.pytorchjob_dropdown.options = job_names
            
            # Provide user feedback
            if hasattr(self, 'output'):
                with self.output:
                    if job_names:
                        print(f"Found {len(job_names)} PyTorchJob(s) in namespace '{self._get_selected_namespace()}'")
                    else:
                        print(f"No PyTorchJobs found in namespace '{self._get_selected_namespace()}'")
                        print("Make sure PyTorchJobs exist and you have the correct permissions")
            
        except client.ApiException as e:
            self.pytorchjob_dropdown.options = []
            with self.output:
                if e.status == 401:
                    print("Authentication failed: Please check your Kubernetes token")
                elif e.status == 403:
                    print("Access denied: You don't have permission to list PyTorchJobs in this namespace")
                elif e.status == 404:
                    print("PyTorchJob resource not found: Training Operator may not be installed")
                else:
                    print(f"Kubernetes API error ({e.status}): {e.reason}")
        except Exception as e:
            self.pytorchjob_dropdown.options = []
            with self.output:
                if "connection" in str(e).lower():
                    print("Connection failed: Please check your Kubernetes API server URL")
                else:
                    print(f"Error fetching PyTorchJobs: {e}")
    
    def _on_namespace_change(self, change):
        """Handles namespace dropdown changes."""
        self._update_pytorchjob_dropdown()
        
    def _on_credentials_change(self, change):
        """Handles kubernetes credentials changes."""
        # Only refresh if both credentials are provided
        if self.kube_api_server.value.strip() and self.kube_token.value.strip():
            self._update_pytorchjob_dropdown()
    
    def _refresh_jobs_button_click(self, button):
        """Handles refresh jobs button click."""
        with self.output:
            print("Refreshing PyTorchJob list...")
        self._update_pytorchjob_dropdown()
        
    def _on_monitor_logs_change(self, change):
        """Handles log monitoring checkbox changes."""
        if change['new']:  # Start monitoring
            self._start_log_monitoring()
        else:  # Stop monitoring
            self._stop_log_monitoring()
    
    def _start_log_monitoring(self):
        """Starts background log monitoring for the selected PyTorchJob."""
        if self.log_monitor_thread and self.log_monitor_thread.is_alive():
            return  # Already monitoring
            
        selected_job = self.pytorchjob_dropdown.value
        if not selected_job:
            with self.output:
                print("Please select a PyTorchJob to monitor")
            self.monitor_logs_checkbox.value = False
            return
            
        # Extract job name (remove status indicator)
        job_name = selected_job.split(' (')[0]
        
        self.stop_monitoring = False
        self.log_monitor_thread = threading.Thread(
            target=self._monitor_logs_worker,
            args=(job_name,),
            daemon=True
        )
        self.log_monitor_thread.start()
        
        self.log_status.value = f'<span style="color: green;">Log monitoring: Active for {job_name}</span>'
        with self.output:
            print(f"Started monitoring logs for PyTorchJob: {job_name}")
    
    def _stop_log_monitoring(self):
        """Stops background log monitoring."""
        self.stop_monitoring = True
        self.log_status.value = '<i>Log monitoring: Inactive</i>'
        with self.output:
            print("Stopped log monitoring")
    
    def _monitor_logs_worker(self, job_name):
        """Background worker that monitors PyTorchJob logs for checkpoint events."""
        try:
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value
            api_token = self.kube_token.value
            namespace = self._get_selected_namespace()
            
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            v1 = client.CoreV1Api(api_client)
            
            with self.output:
                print(f"🔍 Starting log monitoring for PyTorchJob '{job_name}' in namespace '{namespace}'")
            
            # Improved checkpoint detection patterns - handle both files and directories
            checkpoint_patterns = [
                # Directory patterns (common in transformers/pytorch lightning) - Fixed regex
                r'saving.*?checkpoint.*?to\s+([/\w\-\./]+/checkpoint-\d+)',
                r'saved.*?checkpoint.*?to\s+([/\w\-\./]+/checkpoint-\d+)', 
                r'checkpoint.*?saved.*?to\s+([/\w\-\./]+)',
                r'saving.*?model.*?to\s+([/\w\-\./]+)',
                # More comprehensive model checkpoint patterns
                r'saving model checkpoint to\s+([/\w\-\./]+)',
                r'model.*?checkpoint.*?saved.*?to\s+([/\w\-\./]+)',
                # File patterns (traditional checkpoints)
                r'saved checkpoint.*?(\S+\.(?:ckpt|pt|pth|bin|safetensors))',
                r'saving.*?checkpoint.*?(\S+\.(?:ckpt|pt|pth|bin|safetensors))',
                r'checkpoint.*?saved.*?(\S+\.(?:ckpt|pt|pth|bin|safetensors))',
                r'model.*?saved.*?(\S+\.(?:ckpt|pt|pth|bin|safetensors))',
                # Generic patterns
                r'saving.*?(?:checkpoint|model).*?([/\w\-\./]+/(?:checkpoint|ckpt|step|epoch)[-_]\d+)',
                r'saved.*?(?:checkpoint|model).*?([/\w\-\./]+/(?:checkpoint|ckpt|step|epoch)[-_]\d+)'
            ]
            
            # Get pods associated with the PyTorchJob - Try multiple label selectors
            possible_selectors = [
                f"job-name={job_name}",
                f"pytorch-job-name={job_name}",
                f"training.kubeflow.org/job-name={job_name}",
                f"pytorch.org/job-name={job_name}",
                f"app.kubernetes.io/name={job_name}"
            ]
            
            logs_inspected = False
            pods_found = False
            
            while not self.stop_monitoring:
                try:
                    # Try different label selectors to find pods
                    all_pods = []
                    working_selector = None
                    
                    for selector in possible_selectors:
                        pods = v1.list_namespaced_pod(
                            namespace=namespace,
                            label_selector=selector
                        )
                        if pods.items:
                            all_pods = pods.items
                            working_selector = selector
                            if not pods_found:
                                with self.output:
                                    print(f"📦 Found {len(pods.items)} pod(s) using label selector: {selector}")
                                pods_found = True
                            break
                    
                    if not all_pods and not pods_found:
                        # Try to find pods without label selector, matching by name pattern
                        all_pods_in_ns = v1.list_namespaced_pod(namespace=namespace)
                        for pod in all_pods_in_ns.items:
                            if job_name in pod.metadata.name:
                                all_pods.append(pod)
                        
                        if all_pods:
                            with self.output:
                                print(f"📦 Found {len(all_pods)} pod(s) by name pattern matching")
                            pods_found = True
                    
                    if not all_pods:
                        if not pods_found:
                            with self.output:
                                print(f"⚠️  No pods found for PyTorchJob '{job_name}'. Tried selectors: {possible_selectors}")
                            pods_found = True  # Avoid repeated messages
                        time.sleep(10)
                        continue
                    
                    for pod in all_pods:
                        if self.stop_monitoring:
                            break
                            
                        pod_name = pod.metadata.name
                        
                        try:
                            # Get recent logs
                            logs = v1.read_namespaced_pod_log(
                                name=pod_name,
                                namespace=namespace,
                                since_seconds=60,  # Increased to 60 seconds to catch more logs
                                timestamps=True
                            )
                            
                            if not logs_inspected:
                                with self.output:
                                    print(f"📋 Inspecting logs from pod '{pod_name}'...")
                                    logs_inspected = True
                            
                            # Check for checkpoint patterns
                            lines_checked = 0
                            for line in logs.split('\n'):
                                if self.stop_monitoring:
                                    break
                                
                                lines_checked += 1
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                for i, pattern in enumerate(checkpoint_patterns):
                                    match = re.search(pattern, line, re.IGNORECASE)
                                    if match:
                                        checkpoint_path = match.group(1)
                                        timestamp = datetime.now().strftime("%H:%M:%S")
                                        
                                        with self.output:
                                            print(f"✅ [Pattern {i+1}] Found checkpoint pattern in log line: {line}")
                                            print(f"📁 Extracted checkpoint path: {checkpoint_path}")
                                        
                                        # Avoid duplicate notifications for the same checkpoint
                                        if checkpoint_path not in self.last_checkpoint_time:
                                            self.last_checkpoint_time[checkpoint_path] = timestamp
                                            
                                            # Add to detected checkpoints for prioritized listing
                                            self.detected_checkpoints.add(checkpoint_path)
                                            
                                            # Update UI
                                            with self.output:
                                                print(f"🎉 [{timestamp}] New checkpoint detected: {checkpoint_path}")
                                            
                                            # Refresh checkpoints dropdown
                                            self.update_checkpoints_dropdown()
                                        else:
                                            with self.output:
                                                print(f"🔄 [{timestamp}] Checkpoint already known: {checkpoint_path}")
                                        break
                            
                            # Log inspection summary every 30 seconds
                            current_time = datetime.now()
                            if not hasattr(self, '_last_summary_time'):
                                self._last_summary_time = current_time
                            elif (current_time - self._last_summary_time).seconds >= 30:
                                with self.output:
                                    print(f"📊 Log inspection summary: {lines_checked} lines checked, {len(self.detected_checkpoints)} total checkpoints found")
                                self._last_summary_time = current_time
                                        
                        except client.ApiException as e:
                            # Pod might not be ready yet or logs not available
                            if e.status == 400:
                                with self.output:
                                    print(f"⚠️  Pod '{pod_name}' logs not available yet (container may be starting)")
                            pass
                        except Exception as e:
                            with self.output:
                                print(f"❌ Error reading logs from pod '{pod_name}': {e}")
                            
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    with self.output:
                        print(f"❌ Error in log monitoring: {e}")
                    time.sleep(10)  # Wait longer on error
                    
        except Exception as e:
            with self.output:
                print(f"💥 Fatal error in log monitoring: {e}")
        finally:
            if not self.stop_monitoring:
                self.log_status.value = '<span style="color: red;">Log monitoring: Error</span>'

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
    
    def __del__(self):
        """Cleanup when widget is destroyed."""
        self._stop_log_monitoring()
    
    def _ipython_display_(self):
        """Allows the object to be displayed directly in a notebook."""
        display(self.ui)